#!/usr/bin/env python3
"""
Power monitoring utility for tracking GPU and CPU power consumption during training.
Supports parallel jobs and provides detailed energy consumption reports.
"""
import time
import psutil
import subprocess
import csv
import json
import os
import threading
import argparse
from datetime import datetime
from typing import Callable, Dict, List, Optional
import signal
import sys


class PowerMonitor:
    """
    Monitor GPU and CPU power consumption during parallel training jobs.
    
    Tracks:
    - GPU power usage per device (via nvidia-smi)
    - CPU power estimation based on utilization
    - Total energy consumption
    - Per-process power allocation
    """
    
    def __init__(self, 
                 output_dir: str = "power_logs",
                 sample_interval: float = 1.0,
                 process_filter: Optional[str] = None,
                 on_sample: Optional[Callable[[Dict], None]] = None):
        """
        Initialize power monitor.
        
        Args:
            output_dir: Directory to save power logs
            sample_interval: Sampling interval in seconds
            process_filter: Filter processes by name (e.g., 'python')
        """
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.process_filter = process_filter
        self.on_sample = on_sample
        self.running = False
        self.monitoring_thread = None
        self._sample_index = 0
        
        # Data storage
        self.samples = []
        self.start_time = None
        self.end_time = None
        
        # System specs
        self.cpu_tdp = 105  # Watts Intel Xeon Gold 5120
        self.gpu_max_power = 250  # Watts per GTX 1080 Ti
        self.num_gpus = 7  # From your nvidia-smi output
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup signal handlers for shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\nReceived interrupt signal. Stopping monitoring...")
        self.stop()
        sys.exit(0)
    
    def get_gpu_power(self) -> List[Dict[str, float]]:
        """
        Get current power usage for all GPUs using nvidia-smi.
        
        Returns:
            List of dicts with GPU power info
        """
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,name,power.draw,power.limit,utilization.gpu,utilization.memory,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return []
            
            gpu_data = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpu_data.append({
                        'gpu_id': int(parts[0]),
                        'name': parts[1],
                        'power_draw': float(parts[2]) if parts[2] != 'N/A' else 0.0,
                        'power_limit': float(parts[3]) if parts[3] != 'N/A' else self.gpu_max_power,
                        'gpu_util': float(parts[4]) if parts[4] != 'N/A' else 0.0,
                        'mem_util': float(parts[5]) if parts[5] != 'N/A' else 0.0,
                        'temperature': float(parts[6]) if parts[6] != 'N/A' else 0.0
                    })
            return gpu_data
        except Exception as e:
            print(f"Error reading GPU power: {e}")
            return []
    
    def get_cpu_power_estimate(self) -> Dict[str, float]:
        """
        Estimate CPU power based on utilization.
        
        Uses linear scaling from idle to TDP based on CPU utilization.
        More accurate methods would require hardware counters (RAPL).
        
        Returns:
            Dict with CPU power estimation
        """
        try:
            # Get CPU utilization (percentage)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get per-core utilization for better estimate
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            avg_core_util = sum(per_core) / len(per_core)
            
            # Estimate power: assume idle is ~30% of TDP, scales linearly to 100% at full load
            idle_power = self.cpu_tdp * 0.3
            dynamic_power = (self.cpu_tdp - idle_power) * (avg_core_util / 100.0)
            total_power = idle_power + dynamic_power
            
            return {
                'cpu_utilization': avg_core_util,
                'cpu_power_estimate': total_power,
                'num_cores_active': sum(1 for u in per_core if u > 5.0),
                'max_core_util': max(per_core),
                'min_core_util': min(per_core)
            }
        except Exception as e:
            print(f"Error reading CPU power: {e}")
            return {
                'cpu_utilization': 0.0,
                'cpu_power_estimate': 0.0,
                'num_cores_active': 0,
                'max_core_util': 0.0,
                'min_core_util': 0.0
            }
    
    def get_process_info(self) -> List[Dict]:
        """
        Get information about running processes (filtered by name if specified).
        
        Returns:
            List of process info dicts
        """
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if self.process_filter and self.process_filter.lower() not in proc.info['name'].lower():
                        continue
                    
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error getting process info: {e}")
        
        return processes
    
    def collect_sample(self) -> Dict:
        """
        Collect a single power measurement sample.
        
        Returns:
            Dict containing all measurements
        """
        timestamp = time.time()
        
        gpu_data = self.get_gpu_power()
        total_gpu_power = sum(gpu['power_draw'] for gpu in gpu_data)
        
        cpu_data = self.get_cpu_power_estimate()
        
        processes = self.get_process_info()
        
        sample = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'gpu_data': gpu_data,
            'cpu_data': cpu_data,
            'total_gpu_power': total_gpu_power,
            'total_cpu_power': cpu_data['cpu_power_estimate'],
            'total_power': total_gpu_power + cpu_data['cpu_power_estimate'],
            'num_processes': len(processes),
            'processes': processes
        }
        
        return sample
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        print(f"Power monitoring started. Sampling every {self.sample_interval}s")
        print(f"Logs will be saved to: {self.output_dir}")
        
        while self.running:
            try:
                sample = self.collect_sample()
                self.samples.append(sample)
                self._sample_index += 1
                if self.on_sample:
                    try:
                        self.on_sample(sample)
                    except Exception as callback_error:
                        print(f"Power monitor callback error: {callback_error}")
                
                if len(self.samples) % 10 == 0:
                    print(f"[{len(self.samples)} samples] "
                          f"Total Power: {sample['total_power']:.1f}W "
                          f"(GPU: {sample['total_gpu_power']:.1f}W, "
                          f"CPU: {sample['total_cpu_power']:.1f}W) "
                          f"Processes: {sample['num_processes']}")
                
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        if self.running:
            print("Monitoring already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.samples = []
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop(self):
        """Stop monitoring and save results."""
        if not self.running:
            print("Monitoring not running")
            return
        
        self.running = False
        self.end_time = time.time()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.save_results()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.running:
            self.stop()
    
    def save_results(self):
        """Save monitoring results to files."""
        if not self.samples:
            print("No samples collected")
            return
        
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save detailed CSV
        csv_file = os.path.join(self.output_dir, f"power_log_{timestamp_str}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'datetime', 'total_power_w', 'total_gpu_power_w', 'total_cpu_power_w',
                'cpu_utilization_pct', 'num_active_gpus', 'num_processes'
            ])
            
            for sample in self.samples:
                num_active_gpus = sum(1 for gpu in sample['gpu_data'] if gpu['gpu_util'] > 5.0)
                writer.writerow([
                    sample['timestamp'],
                    sample['datetime'],
                    sample['total_power'],
                    sample['total_gpu_power'],
                    sample['total_cpu_power'],
                    sample['cpu_data']['cpu_utilization'],
                    num_active_gpus,
                    sample['num_processes']
                ])
        
        print(f"Detailed log saved to: {csv_file}")
        
        # Save summary JSON
        summary = self.calculate_summary()
        json_file = os.path.join(self.output_dir, f"power_summary_{timestamp_str}.json")
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {json_file}")
        
        # Print summary
        self.print_summary(summary)
    
    def calculate_summary(self) -> Dict:
        """Calculate summary statistics from collected samples."""
        if not self.samples:
            return {}
        
        duration = self.end_time - self.start_time
        
        # Power statistics
        total_powers = [s['total_power'] for s in self.samples]
        gpu_powers = [s['total_gpu_power'] for s in self.samples]
        cpu_powers = [s['total_cpu_power'] for s in self.samples]
        
        # Energy calculation (Watt-hours)
        # Energy = Average Power × Time
        avg_total_power = sum(total_powers) / len(total_powers)
        total_energy_wh = (avg_total_power * duration) / 3600
        
        avg_gpu_power = sum(gpu_powers) / len(gpu_powers)
        gpu_energy_wh = (avg_gpu_power * duration) / 3600
        
        avg_cpu_power = sum(cpu_powers) / len(cpu_powers)
        cpu_energy_wh = (avg_cpu_power * duration) / 3600
        
        # Per-GPU statistics
        gpu_stats = {}
        for gpu_id in range(self.num_gpus):
            gpu_samples = [s['gpu_data'][gpu_id] for s in self.samples if len(s['gpu_data']) > gpu_id]
            if gpu_samples:
                powers = [g['power_draw'] for g in gpu_samples]
                utils = [g['gpu_util'] for g in gpu_samples]
                temps = [g['temperature'] for g in gpu_samples]
                
                gpu_stats[f'gpu_{gpu_id}'] = {
                    'avg_power': sum(powers) / len(powers),
                    'max_power': max(powers),
                    'min_power': min(powers),
                    'avg_utilization': sum(utils) / len(utils),
                    'max_utilization': max(utils),
                    'avg_temperature': sum(temps) / len(temps),
                    'max_temperature': max(temps),
                    'energy_wh': (sum(powers) / len(powers) * duration) / 3600
                }
        
        summary = {
            'monitoring_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
                'duration_seconds': duration,
                'duration_hours': duration / 3600,
                'num_samples': len(self.samples),
                'sample_interval': self.sample_interval
            },
            'power_statistics': {
                'avg_total_power_w': avg_total_power,
                'max_total_power_w': max(total_powers),
                'min_total_power_w': min(total_powers),
                'avg_gpu_power_w': avg_gpu_power,
                'avg_cpu_power_w': avg_cpu_power
            },
            'energy_consumption': {
                'total_energy_wh': total_energy_wh,
                'total_energy_kwh': total_energy_wh / 1000,
                'gpu_energy_wh': gpu_energy_wh,
                'cpu_energy_wh': cpu_energy_wh,
                'gpu_percentage': (gpu_energy_wh / total_energy_wh * 100) if total_energy_wh > 0 else 0,
                'cpu_percentage': (cpu_energy_wh / total_energy_wh * 100) if total_energy_wh > 0 else 0
            },
            'cost_estimate': {
                'energy_kwh': total_energy_wh / 1000,
                'cost_usd_at_0_10_per_kwh': (total_energy_wh / 1000) * 0.10,
                'cost_usd_at_0_15_per_kwh': (total_energy_wh / 1000) * 0.15,
                'cost_usd_at_0_20_per_kwh': (total_energy_wh / 1000) * 0.20
            },
            'gpu_statistics': gpu_stats,
            'process_statistics': {
                'avg_num_processes': sum(s['num_processes'] for s in self.samples) / len(self.samples),
                'max_num_processes': max(s['num_processes'] for s in self.samples)
            }
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print formatted summary to console."""
        print("\n" + "="*80)
        print("POWER MONITORING SUMMARY")
        print("="*80)
        
        info = summary['monitoring_info']
        print(f"\nMonitoring Duration: {info['duration_hours']:.2f} hours ({info['duration_seconds']:.1f} seconds)")
        print(f"Samples Collected: {info['num_samples']}")
        
        power = summary['power_statistics']
        print(f"\nAverage Power Consumption:")
        print(f"  Total:  {power['avg_total_power_w']:.1f} W")
        print(f"  GPU:    {power['avg_gpu_power_w']:.1f} W")
        print(f"  CPU:    {power['avg_cpu_power_w']:.1f} W")
        
        energy = summary['energy_consumption']
        print(f"\nTotal Energy Consumption:")
        print(f"  Total:  {energy['total_energy_kwh']:.4f} kWh ({energy['total_energy_wh']:.2f} Wh)")
        print(f"  GPU:    {energy['gpu_energy_wh']:.2f} Wh ({energy['gpu_percentage']:.1f}%)")
        print(f"  CPU:    {energy['cpu_energy_wh']:.2f} Wh ({energy['cpu_percentage']:.1f}%)")
        
        cost = summary['cost_estimate']
        print(f"\nEstimated Cost:")
        print(f"  At $0.10/kWh: ${cost['cost_usd_at_0_10_per_kwh']:.4f}")
        print(f"  At $0.15/kWh: ${cost['cost_usd_at_0_15_per_kwh']:.4f}")
        print(f"  At $0.20/kWh: ${cost['cost_usd_at_0_20_per_kwh']:.4f}")
        
        print("\nPer-GPU Statistics:")
        for gpu_name, stats in summary['gpu_statistics'].items():
            print(f"  {gpu_name}:")
            print(f"    Avg Power: {stats['avg_power']:.1f} W, "
                  f"Avg Util: {stats['avg_utilization']:.1f}%, "
                  f"Energy: {stats['energy_wh']:.2f} Wh")
        
        print("="*80 + "\n")


def main():
    """Command-line interface for power monitoring."""
    parser = argparse.ArgumentParser(description='Monitor power consumption during training')
    parser.add_argument('--output-dir', default='power_logs', help='Output directory for logs')
    parser.add_argument('--interval', type=float, default=1.0, help='Sampling interval in seconds')
    parser.add_argument('--filter', default='python', help='Filter processes by name')
    parser.add_argument('--duration', type=float, default=None, help='Duration to monitor (seconds)')
    
    args = parser.parse_args()
    
    monitor = PowerMonitor(
        output_dir=args.output_dir,
        sample_interval=args.interval,
        process_filter=args.filter
    )
    
    monitor.start()
    
    try:
        if args.duration:
            print(f"Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("Monitoring until interrupted (Ctrl+C)...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
