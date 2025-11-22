#!/usr/bin/env python3
"""
Analyze power consumption logs and generate comprehensive reports.
"""
import json
import csv
import os
import glob
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


def load_power_logs(log_dir: str) -> Dict:
    """Load power monitoring logs from directory."""
    # Find summary JSON
    json_files = glob.glob(os.path.join(log_dir, "power_summary_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No power summary found in {log_dir}")
    
    with open(json_files[0], 'r') as f:
        summary = json.load(f)
    
    # Find detailed CSV
    csv_files = glob.glob(os.path.join(log_dir, "power_log_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No power log CSV found in {log_dir}")
    
    # Load CSV data
    samples = []
    with open(csv_files[0], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'timestamp': float(row['timestamp']),
                'datetime': row['datetime'],
                'total_power_w': float(row['total_power_w']),
                'total_gpu_power_w': float(row['total_gpu_power_w']),
                'total_cpu_power_w': float(row['total_cpu_power_w']),
                'cpu_utilization_pct': float(row['cpu_utilization_pct']),
                'num_active_gpus': int(row['num_active_gpus']),
                'num_processes': int(row['num_processes'])
            })
    
    return {
        'summary': summary,
        'samples': samples,
        'summary_file': json_files[0],
        'csv_file': csv_files[0]
    }


def plot_power_timeline(samples: List[Dict], output_file: str):
    """Generate timeline plot of power consumption."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Normalize timestamps to start at 0
    start_time = samples[0]['timestamp']
    times = [(s['timestamp'] - start_time) / 3600 for s in samples]  # Convert to hours
    
    # Plot 1: Total Power
    ax1 = axes[0]
    total_power = [s['total_power_w'] for s in samples]
    gpu_power = [s['total_gpu_power_w'] for s in samples]
    cpu_power = [s['total_cpu_power_w'] for s in samples]
    
    ax1.plot(times, total_power, label='Total Power', linewidth=2, color='black')
    ax1.fill_between(times, gpu_power, alpha=0.5, label='GPU Power', color='green')
    ax1.fill_between(times, cpu_power, alpha=0.5, label='CPU Power', color='blue')
    ax1.set_ylabel('Power (W)', fontsize=12)
    ax1.set_title('Power Consumption Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GPU Utilization
    ax2 = axes[1]
    num_active_gpus = [s['num_active_gpus'] for s in samples]
    ax2.plot(times, num_active_gpus, linewidth=2, color='orange')
    ax2.set_ylabel('Active GPUs', fontsize=12)
    ax2.set_title('GPU Utilization', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 7)  # 7 GPUs total
    
    # Plot 3: Number of Processes
    ax3 = axes[2]
    num_processes = [s['num_processes'] for s in samples]
    ax3.plot(times, num_processes, linewidth=2, color='purple')
    ax3.set_ylabel('Python Processes', fontsize=12)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_title('Active Python Processes', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Timeline plot saved to: {output_file}")
    plt.close()


def plot_power_distribution(samples: List[Dict], output_file: str):
    """Generate power distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    total_power = [s['total_power_w'] for s in samples]
    gpu_power = [s['total_gpu_power_w'] for s in samples]
    cpu_power = [s['total_cpu_power_w'] for s in samples]
    
    # Histogram of total power
    ax1 = axes[0]
    ax1.hist(total_power, bins=50, alpha=0.7, color='black', edgecolor='black')
    ax1.axvline(np.mean(total_power), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_power):.1f}W')
    ax1.set_xlabel('Total Power (W)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Total Power Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stacked histogram
    ax2 = axes[1]
    ax2.hist([gpu_power, cpu_power], bins=30, stacked=True, alpha=0.7, 
             label=['GPU', 'CPU'], color=['green', 'blue'], edgecolor='black')
    ax2.set_xlabel('Power (W)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('GPU vs CPU Power Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_file}")
    plt.close()


def generate_report(log_dir: str, output_dir: str = None):
    """Generate comprehensive power analysis report."""
    if output_dir is None:
        output_dir = log_dir
    
    print(f"Loading power logs from: {log_dir}")
    data = load_power_logs(log_dir)
    
    summary = data['summary']
    samples = data['samples']
    
    # Print summary to console
    print("\n" + "="*80)
    print("POWER CONSUMPTION ANALYSIS REPORT")
    print("="*80)
    
    info = summary['monitoring_info']
    print(f"\nMonitoring Period:")
    print(f"  Start:    {info['start_time']}")
    print(f"  End:      {info['end_time']}")
    print(f"  Duration: {info['duration_hours']:.2f} hours ({info['duration_seconds']:.0f} seconds)")
    print(f"  Samples:  {info['num_samples']}")
    
    power = summary['power_statistics']
    print(f"\nPower Statistics:")
    print(f"  Average Total: {power['avg_total_power_w']:.1f} W")
    print(f"  Maximum Total: {power['max_total_power_w']:.1f} W")
    print(f"  Minimum Total: {power['min_total_power_w']:.1f} W")
    print(f"  Average GPU:   {power['avg_gpu_power_w']:.1f} W")
    print(f"  Average CPU:   {power['avg_cpu_power_w']:.1f} W")
    
    energy = summary['energy_consumption']
    print(f"\nEnergy Consumption:")
    print(f"  Total:  {energy['total_energy_kwh']:.4f} kWh ({energy['total_energy_wh']:.1f} Wh)")
    print(f"  GPU:    {energy['gpu_energy_wh']:.1f} Wh ({energy['gpu_percentage']:.1f}%)")
    print(f"  CPU:    {energy['cpu_energy_wh']:.1f} Wh ({energy['cpu_percentage']:.1f}%)")
    
    cost = summary['cost_estimate']
    print(f"\nEstimated Electricity Cost:")
    print(f"  At $0.10/kWh: ${cost['cost_usd_at_0_10_per_kwh']:.4f}")
    print(f"  At $0.15/kWh: ${cost['cost_usd_at_0_15_per_kwh']:.4f}")
    print(f"  At $0.20/kWh: ${cost['cost_usd_at_0_20_per_kwh']:.4f}")
    
    print(f"\nCO2 Emissions (estimated):")
    # Average carbon intensity: ~0.4 kg CO2/kWh (varies by region)
    co2_kg = energy['total_energy_kwh'] * 0.4
    print(f"  ~{co2_kg:.3f} kg CO2")
    
    print(f"\nPer-GPU Breakdown:")
    for gpu_name, stats in summary['gpu_statistics'].items():
        print(f"  {gpu_name}:")
        print(f"    Avg Power: {stats['avg_power']:>6.1f} W  |  "
              f"Max Power: {stats['max_power']:>6.1f} W  |  "
              f"Avg Util: {stats['avg_utilization']:>5.1f}%")
        print(f"    Energy:    {stats['energy_wh']:>6.2f} Wh |  "
              f"Avg Temp:  {stats['avg_temperature']:>5.1f}°C  |  "
              f"Max Temp: {stats['max_temperature']:>5.1f}°C")
    
    # Generate plots
    print(f"\nGenerating visualizations...")
    timeline_file = os.path.join(output_dir, "power_timeline.png")
    plot_power_timeline(samples, timeline_file)
    
    distribution_file = os.path.join(output_dir, "power_distribution.png")
    plot_power_distribution(samples, distribution_file)
    
    # Save text report
    report_file = os.path.join(output_dir, "power_analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POWER CONSUMPTION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Source: {data['summary_file']}\n\n")
        
        f.write(json.dumps(summary, indent=2))
    
    print(f"\nText report saved to: {report_file}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze power consumption logs')
    parser.add_argument('log_dir', help='Directory containing power logs')
    parser.add_argument('--output', help='Output directory for reports (default: same as log_dir)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.log_dir):
        print(f"Error: {args.log_dir} is not a directory")
        return 1
    
    generate_report(args.log_dir, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
