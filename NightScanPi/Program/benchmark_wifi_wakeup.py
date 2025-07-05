"""Benchmark wifi_wakeup tone detection CPU usage."""
from __future__ import annotations

import time
import os

import NightScanPi.Program.wifi_wakeup as wifi_wakeup


def benchmark(iterations: int = 100) -> float:
    """Return average CPU usage percentage for tone detection."""
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    for _ in range(iterations):
        wifi_wakeup.detect_tone()
    cpu_time = time.process_time() - start_cpu
    wall_time = time.perf_counter() - start_wall
    if wall_time == 0:
        return 0.0
    return 100.0 * cpu_time / wall_time


def main() -> None:
    iterations = int(os.getenv("NIGHTSCAN_BENCH_ITERS", "100"))
    usage = benchmark(iterations)
    print(f"Average CPU usage over {iterations} detections: {usage:.1f}%")


if __name__ == "__main__":
    main()
