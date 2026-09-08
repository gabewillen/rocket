#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize the two fixed SM121 gate/up Nsight Compute reports."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re


CASES = {
    "c8-k7-r1": (292, 51),
    "c16-k7-r1": (1280, 256),
}
BYTES_PER_EXPERT = 2 * (2560 * 640 + (2560 // 128) * (640 // 128) * 2)
SECTOR_BYTES = 32
KERNEL = "gate_up_silu"
PASSES = 19
STALL_METRICS = tuple(
    f"smsp__warp_issue_stalled_{reason}_per_warp_active.pct"
    for reason in (
        "barrier", "branch_resolving", "dispatch_stall", "drain",
        "lg_throttle", "long_scoreboard", "math_pipe_throttle", "membar",
        "mio_throttle", "misc", "no_instruction", "not_selected", "selected",
        "short_scoreboard", "sleeping", "tex_throttle", "wait",
    )
)
REQUIRED_UNITS = {
    "gpu__time_duration.sum": "ms",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "%",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
    "lts__t_sectors_srcunit_tex_lookup_hit.sum": "sector",
    "lts__t_sectors_srcunit_tex_lookup_miss.sum": "sector",
    "lts__t_sectors_aperture_device_op_read.sum": "sector",
    "lts__t_sectors_aperture_sysmem_op_read.sum": "sector",
    "profiler__replayer_passes": "pass",
    **{name: "%" for name in STALL_METRICS},
}


def number(value: str) -> float:
    stripped = value.strip()
    if "," in stripped or not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped
    ):
        raise ValueError(f"non-canonical numeric value {value!r}")
    result = float(stripped)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite numeric value {value!r}")
    return result


def read_metrics(
    path: pathlib.Path,
) -> tuple[dict[str, float], dict[str, str], str]:
    lines = [line for line in path.read_text().splitlines() if line.startswith('"')]
    metrics: dict[str, float] = {}
    units: dict[str, str] = {}
    rows = list(csv.reader(lines))
    if rows and "Metric Name" in rows[0]:
        schema = "long"
        required_columns = {"Metric Name", "Metric Unit", "Metric Value", "Kernel Name"}
        if not required_columns.issubset(rows[0]):
            raise ValueError(f"incomplete long schema in {path}")
        kernels = set()
        for row in csv.DictReader(lines):
            name = row.get("Metric Name", "")
            value = row.get("Metric Value", "")
            kernel = row.get("Kernel Name", "")
            if kernel:
                kernels.add(kernel)
            if name not in REQUIRED_UNITS:
                continue
            if name in metrics:
                raise ValueError(f"duplicate metric {name} in {path}")
            metrics[name] = number(value)
            units[name] = row.get("Metric Unit", "")
        if kernels != {KERNEL}:
            raise ValueError(f"wrong kernel set {sorted(kernels)} in {path}")
    elif len(rows) >= 3:
        schema = "wide"
        if len(rows) != 3:
            raise ValueError(f"wide schema must contain exactly three rows in {path}")
        if not (len(rows[0]) == len(rows[1]) == len(rows[2])):
            raise ValueError(f"wide schema row width mismatch in {path}")
        names = [name for name in rows[0] if name]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate wide-schema column in {path}")
        metadata = dict(zip(rows[0], rows[2], strict=True))
        if metadata.get("Kernel Name") != KERNEL:
            raise ValueError(f"wrong kernel {metadata.get('Kernel Name')!r} in {path}")
        for name, unit, value in zip(rows[0], rows[1], rows[2], strict=True):
            if name not in REQUIRED_UNITS:
                continue
            metrics[name] = number(value)
            units[name] = unit
    else:
        raise ValueError(f"unrecognized or mixed NCU CSV schema in {path}")
    missing = REQUIRED_UNITS.keys() - metrics.keys()
    if missing:
        raise ValueError(f"missing metrics {sorted(missing)} in {path}")
    drift = {
        name: (units[name], expected)
        for name, expected in REQUIRED_UNITS.items()
        if units[name] != expected
    }
    if drift:
        raise ValueError(f"metric unit drift {drift} in {path}")
    if metrics["profiler__replayer_passes"] != PASSES:
        raise ValueError(f"wrong replay pass count in {path}")
    for name, value in metrics.items():
        if value < 0 or (units[name] == "%" and value > 100):
            raise ValueError(f"out-of-range metric {name}={value} in {path}")
    return metrics, units, schema


def first(metrics: dict[str, float], *names: str) -> float:
    for name in names:
        if name in metrics:
            return metrics[name]
    return float("nan")


def duration_ms(value: float, unit: str) -> float:
    scale = {
        "second": 1e3,
        "msecond": 1.0,
        "usecond": 1e-3,
        "nsecond": 1e-6,
        "s": 1e3,
        "ms": 1.0,
        "us": 1e-3,
        "ns": 1e-6,
    }
    if unit not in scale:
        raise ValueError(f"unsupported duration unit {unit!r}")
    return value * scale[unit]


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
    profile_cases = [path.stem for path in args.profiles]
    expected_cases = set(CASES) if not args.incomplete else {profile_cases[0]}
    if set(profile_cases) != expected_cases or len(profile_cases) != len(set(profile_cases)):
        parser.error(f"wrong or duplicate profile cases: {profile_cases}")
    print("| case | duration ms | tensor issue % | compute/memory % | L2 % | occupancy % | top stalls | L2 hit % | backing read MB | active MB | expanded MB | backing/active |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    schemas: set[str] = set()
    for path in args.profiles:
        case = path.stem
        if case not in CASES:
            raise ValueError(f"unexpected profile case {case}")
        routes, experts = CASES[case]
        metrics, units, schema = read_metrics(path)
        schemas.add(schema)
        duration_name = next(
            name for name in ("gpu__time_duration.sum", "gpu__time_duration.avg")
            if name in metrics
        )
        elapsed_ms = duration_ms(metrics[duration_name], units[duration_name])
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
            f"| {case} | {elapsed_ms:.3f} | {tensor:.2f} | {compute_memory:.2f} | {l2:.2f} | {occupancy:.2f} "
            f"| {stall_summary(metrics)} | {hit_rate:.2f} | {backing_bytes / 1e6:.3f} | {active_bytes / 1e6:.3f} "
            f"| {expanded_bytes / 1e6:.3f} | {backing_bytes / active_bytes:.3f}x |"
        )
    if len(schemas) != 1:
        raise ValueError(f"mixed NCU CSV schemas: {sorted(schemas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
