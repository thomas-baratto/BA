#!/usr/bin/env python3
"""
Parallel launcher for sweep jobs with Pareto maintenance.

Runs up to `max_parallel` training commands concurrently, each pinned to its
own GPU via a thread-safe GPU queue.  Summarize + pareto prune run only once
after ALL jobs finish, so there are zero concurrency issues with the frontier.
"""
import sys
import subprocess
import os
import time
import shlex
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Thread-safe GPU pool
gpu_queue: queue.Queue[int] = queue.Queue()


def run_cmd(cmd: str) -> tuple[int, str]:
    """Grab a free GPU, run the command, return the GPU."""
    gpu_id = gpu_queue.get()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        res = subprocess.run(
            cmd, shell=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=600,  # 10 min timeout per run
        )
        return res.returncode, cmd
    except subprocess.TimeoutExpired:
        return -1, cmd
    except Exception as e:
        print(f"Exception running {cmd}: {e}")
        return -1, cmd
    finally:
        gpu_queue.put(gpu_id)


def run_maintenance(base_dir: str) -> None:
    """Run summarize + pareto prune once."""
    env = os.environ.copy()
    try:
        subprocess.run(
            f"python scripts/analysis/summarize_results.py --run-dir \"{base_dir}\"",
            shell=True, env=env, capture_output=True, timeout=120,
        )
    except Exception as e:
        print(f"  summarize failed: {e}")

    try:
        subprocess.run(
            f"python scripts/analysis/pareto_manager.py --run-dir \"{base_dir}\" --prune",
            shell=True, env=env, capture_output=True, timeout=120, check=False,
        )
    except Exception as e:
        print(f"  pareto prune failed: {e}")


def detect_base_dir(cmd: str) -> str | None:
    """Extract --base-dir value from a command string."""
    parts = shlex.split(cmd)
    if "--base-dir" in parts:
        idx = parts.index("--base-dir")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def main():
    if len(sys.argv) < 3:
        print("Usage: launch_sweep_workers.py <max_parallel_gpus> <cmd1> [<cmd2> ...]")
        sys.exit(1)

    max_gpus = int(sys.argv[1])
    cmds = sys.argv[2:]
    total = len(cmds)

    # Fill GPU pool
    for g in range(max_gpus):
        gpu_queue.put(g)

    # Detect base_dir from first command
    base_dir = detect_base_dir(cmds[0]) if cmds else None

    print(f"=== Parallel sweep: {total} runs, {max_gpus} GPU workers ===")
    t0 = time.time()

    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_gpus) as pool:
        futures = {pool.submit(run_cmd, cmd): cmd for cmd in cmds}

        for fut in as_completed(futures):
            rc, cmd = fut.result()
            completed += 1

            if rc != 0:
                failed += 1
                short = cmd[:120] + "..." if len(cmd) > 120 else cmd
                print(f"  FAIL [{completed}/{total}] rc={rc}: {short}")

            if completed % 25 == 0 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600 if elapsed > 0 else 0
                eta = (total - completed) / (completed / elapsed) if completed else 0
                print(
                    f"  [{completed}/{total}] done "
                    f"({elapsed:.0f}s elapsed, ~{rate:.0f}/hr, ETA {eta:.0f}s)"
                )

    elapsed = time.time() - t0
    print(f"\n=== All {total} sweep runs completed in {elapsed:.0f}s "
          f"({elapsed/60:.1f}min). Failures: {failed} ===")

    # Final maintenance pass
    if base_dir:
        print("Running final summarize + pareto prune...")
        run_maintenance(base_dir)

    print("Done.")


if __name__ == "__main__":
    main()
