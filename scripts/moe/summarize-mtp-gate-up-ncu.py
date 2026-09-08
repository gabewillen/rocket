#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize the two fixed SM121 gate/up Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re


CASES = {
    "c8-k7-r1": (292, 51),
    "c16-k7-r1": (1280, 256),
}
BYTES_PER_EXPERT = 2 * (2560 * 640 + (2560 // 128) * (640 // 128) * 2)
SECTOR_BYTES = 32


def number(value: str) -> float:
    return float(value.replace(",", "").replace("%", "").strip())


def read_metrics(path: pathlib.Path) -> dict[str, float]:
    lines = [line for line in path.read_text().splitlines() if line.startswith('"')]
    reader = csv.DictReader(lines)
    metrics: dict[str, float] = {}
    for row in reader:
        name = row.get("Metric Name", "")
        value = row.get("Metric Value", "")
        if name and value:
            try:
                metrics[name] = number(value)
            except ValueError:
                pass
    if not metrics:
        raise ValueError(f"no NCU metric rows in {path}")
    return metrics


def first(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        if name in metrics:
            return metrics[name]
    return float("nan")


def stall_summary(metrics: dict[str, float]) -> str:
    stalls = []
    for name, value in metrics.items():
        match = re.fullmatch(
            r"smsp__warp_issue_stalled_(.+)_per_warp_active\.pct", name
        )
        if match:
            stalls.append((value, match.group(1)))
    return ", ".join(f"{name}:{value:.1f}%" for value, name in sorted(stalls, reverse=True)[:3]) or "missing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incomplete", action="store_true")
    parser.add_argument("profiles", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    if args.incomplete:
        if len(args.profiles) != 1:
            parser.error("--incomplete requires one profile")
        print("INCOMPLETE_COUNTERS valid=false")
    elif len(args.profiles) != 2:
        parser.error("a complete summary requires two profiles")
    print("| case | duration ms | tensor issue % | compute/memory % | L2 % | occupancy % | top stalls | L2 hit % | backing read MB | active MB | expanded MB | backing/active |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for path in args.profiles:
        case = path.stem
        if case not in CASES:
            raise ValueError(f"unexpected profile case {case}")
        routes, experts = CASES[case]
        metrics = read_metrics(path)
        duration_ns = first(metrics, "gpu__time_duration.sum", "gpu__time_duration.avg")
        tensor = first(metrics, "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
        compute_memory = first(metrics, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
        l2 = first(metrics, "lts__throughput.avg.pct_of_peak_sustained_elapsed")
        occupancy = first(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active")
        hits = first(metrics, "lts__t_sectors_srcunit_tex_lookup_hit.sum")
        misses = first(metrics, "lts__t_sectors_srcunit_tex_lookup_miss.sum")
        device_reads = first(metrics, "lts__t_sectors_aperture_device_op_read.sum")
        sysmem_reads = first(metrics, "lts__t_sectors_aperture_sysmem_op_read.sum")
        backing_bytes = sum(value for value in (device_reads, sysmem_reads) if value == value) * SECTOR_BYTES
        active_bytes = experts * BYTES_PER_EXPERT
        expanded_bytes = routes * BYTES_PER_EXPERT
        hit_rate = 100.0 * hits / (hits + misses) if hits + misses > 0 else float("nan")
        print(
            f"| {case} | {duration_ns / 1e6:.3f} | {tensor:.2f} | {compute_memory:.2f} | {l2:.2f} | {occupancy:.2f} "
            f"| {stall_summary(metrics)} | {hit_rate:.2f} | {backing_bytes / 1e6:.3f} | {active_bytes / 1e6:.3f} "
            f"| {expanded_bytes / 1e6:.3f} | {backing_bytes / active_bytes:.3f}x |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
