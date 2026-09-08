#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reduce repeated routed-expert profile TSV streams to one Markdown table."""

from __future__ import annotations

import argparse
import pathlib
import statistics


STAGES = (
    "whole",
    "route_compact",
    "prepare",
    "quantize_hidden",
    "gate_up",
    "silu_quantize",
    "down_reduce",
)


def parse(paths: list[pathlib.Path]):
    measurements: dict[tuple[int, int, int, str], list[tuple[float, float]]] = {}
    metadata: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {}
    for path in paths:
        seen: set[tuple[int, int, int, str]] = set()
        for line in path.read_text().splitlines():
            fields = line.split("\t")
            if not fields or fields[0] != "RESULT":
                continue
            rank, concurrency, depth = map(int, fields[1:4])
            stage = fields[7]
            key = (rank, concurrency, depth, stage)
            if key in seen:
                raise ValueError(f"duplicate {key} in {path}")
            seen.add(key)
            measurements.setdefault(key, []).append(
                (float(fields[8]), float(fields[9]))
            )
            metadata[(rank, concurrency, depth)] = (
                int(fields[4]), int(fields[5]), int(fields[6]),
                int(fields[17]), int(fields[18])
            )
        if not seen:
            raise ValueError(f"no RESULT rows in {path}")
    expected_runs = len(paths)
    if any(len(values) != expected_runs for values in measurements.values()):
        raise ValueError("profile files do not contain the same workload rows")
    return measurements, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=pathlib.Path)
    parser.add_argument("--read-gbps", type=float, default=238.0)
    args = parser.parse_args()
    measurements, metadata = parse(args.profiles)
    print("| c | K | rank | rows | routes | experts | active MB | expanded MB | roof ms | whole p50/p95 ms | run CV | compact | prepare | quant | gate/up | SiLU | down |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, concurrency, depth in sorted(
        metadata, key=lambda item: (item[1], item[2], item[0])
    ):
        rows, routes, experts, active_bytes, expanded_bytes = metadata[
            (rank, concurrency, depth)
        ]
        medians = {
            stage: (
                statistics.median(
                    value[0] for value in measurements[(rank, concurrency, depth, stage)]
                ),
                statistics.median(
                    value[1] for value in measurements[(rank, concurrency, depth, stage)]
                ),
            )
            for stage in STAGES
        }
        whole_p50s = [
            value[0] for value in measurements[(rank, concurrency, depth, "whole")]
        ]
        run_cv = statistics.pstdev(whole_p50s) / statistics.mean(whole_p50s)
        roof_ms = active_bytes / (args.read_gbps * 1e9) * 1e3
        pair = lambda stage: f"{medians[stage][0]:.3f}/{medians[stage][1]:.3f}"
        print(
            f"| {concurrency} | {depth} | {rank} | {rows} | {routes} | {experts} "
            f"| {active_bytes / 1e6:.3f} | {expanded_bytes / 1e6:.3f} "
            f"| {roof_ms:.3f} | {pair('whole')} | {run_cv * 100:.2f}% "
            f"| {pair('route_compact')} | {pair('prepare')} "
            f"| {pair('quantize_hidden')} | {pair('gate_up')} "
            f"| {pair('silu_quantize')} | {pair('down_reduce')} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
