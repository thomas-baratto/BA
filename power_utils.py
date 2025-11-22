import os
import logging
from contextlib import contextmanager

from torch.utils.tensorboard import SummaryWriter

from power_monitor import PowerMonitor


@contextmanager
def power_monitor_session(args, root_folder: str):
    if getattr(args, "disable_power_monitor", False):
        yield None
        return

    power_log_dir = args.power_log_dir or os.path.join(root_folder, "power_monitor")
    os.makedirs(power_log_dir, exist_ok=True)
    tensorboard_dir = os.path.join(power_log_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    state = {'count': 0}

    def _power_sample_callback(sample, writer=writer, state=state):
        state['count'] += 1
        step = state['count']
        writer.add_scalar("power/total_w", sample['total_power'], step)
        writer.add_scalar("power/gpu_w", sample['total_gpu_power'], step)
        writer.add_scalar("power/cpu_w", sample['total_cpu_power'], step)

    monitor = PowerMonitor(
        output_dir=power_log_dir,
        sample_interval=args.power_interval,
        process_filter=args.power_filter,
        on_sample=_power_sample_callback
    )

    logging.info(
        f"Starting power monitor (interval={args.power_interval}s, filter='{args.power_filter}', dir={power_log_dir})"
    )
    monitor.start()
    try:
        yield {"log_dir": power_log_dir}
    finally:
        logging.info("Stopping power monitor...")
        monitor.stop()
        writer.close()
