#!/usr/bin/env python3
"""Collect and summarize synchronized two-node Qwen3.8 GPU evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "rocket.qwen38.two-node-hardware.v1"
QUERY = "clocks.sm,power.draw,utilization.gpu"
QUERY_TIMEOUT_SECONDS = 10
stop = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def query(command: list[str], rank: int, runner=subprocess.run) -> dict:
    started = time.time_ns()
    try:
        result = runner(command, capture_output=True, text=True, check=False,
                        timeout=QUERY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"rank {rank} nvidia-smi timed out") from error
    finished = time.time_ns()
    if result.returncode:
        raise RuntimeError(f"rank {rank} nvidia-smi failed: {result.stderr.strip()}")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"rank {rank} expected one GPU, observed {len(rows)}")
    fields = [part.strip() for part in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"rank {rank} malformed nvidia-smi output")
    clock, power, utilization = map(float, fields)
    if clock < 0 or power < 0 or not 0 <= utilization <= 100:
        raise RuntimeError(f"rank {rank} nvidia-smi values outside valid ranges")
    return {
        "rank": rank, "observed_at": utc_now(),
        "started_unix_ns": started, "finished_unix_ns": finished,
        "clock_mhz": clock, "power_w": power,
        "utilization_percent": utilization,
    }


def commands(worker: str) -> tuple[list[str], list[str]]:
    base = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]
    return base, ["ssh", "-o", "BatchMode=yes", worker, *base]


def collect(worker: str, output: Path, interval: float) -> None:
    global stop
    local, remote = commands(worker)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        while not stop:
            pair_started = time.time_ns()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(query, command, rank)
                           for rank, command in enumerate((local, remote))]
                rows = [future.result() for future in futures]
            pair_finished = time.time_ns()
            for row in rows:
                row["pair_started_unix_ns"] = pair_started
                row["pair_finished_unix_ns"] = pair_finished
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            time.sleep(interval)


def distribution(values: list[float]) -> dict:
    return {"median": statistics.median(values), "max": max(values)}


def summarize(samples_path: Path, benchmark_path: Path) -> dict:
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
    cases = json.loads(benchmark_path.read_text())
    output_cases = []
    for case in cases:
        ranks = []
        for rank in (0, 1):
            selected = [row for row in samples if row["rank"] == rank
                        and row["finished_unix_ns"] >= case["started_unix_ns"]
                        and row["started_unix_ns"] <= case["finished_unix_ns"]]
            if not selected:
                raise ValueError(f"no rank {rank} samples for c{case['concurrency']}")
            ranks.append({
                "rank": rank, "samples": len(selected),
                "clock_mhz": distribution([row["clock_mhz"] for row in selected]),
                "power_w": distribution([row["power_w"] for row in selected]),
                "utilization_percent": distribution(
                    [row["utilization_percent"] for row in selected]),
            })
        output_cases.append({
            "concurrency": case["concurrency"],
            "started_at": case["started_at"], "finished_at": case["finished_at"],
            "started_unix_ns": case["started_unix_ns"],
            "finished_unix_ns": case["finished_unix_ns"], "ranks": ranks,
        })
    pair_spans = {
        (row["pair_started_unix_ns"], row["pair_finished_unix_ns"])
        for row in samples
    }
    return {
        "schema": SCHEMA, "query": QUERY,
        "max_pair_span_ms": max((end - start) / 1e6 for start, end in pair_spans),
        "cases": output_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    collector = subparsers.add_parser("collect")
    collector.add_argument("--worker", required=True)
    collector.add_argument("--output", required=True, type=Path)
    collector.add_argument("--interval-seconds", type=float, default=1.0)
    reducer = subparsers.add_parser("summarize")
    reducer.add_argument("--samples", required=True, type=Path)
    reducer.add_argument("--benchmark", required=True, type=Path)
    reducer.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "collect":
        if args.interval_seconds <= 0:
            parser.error("--interval-seconds must be positive")
        signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("stop", True))
        signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("stop", True))
        collect(args.worker, args.output, args.interval_seconds)
    else:
        payload = summarize(args.samples, args.benchmark)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
