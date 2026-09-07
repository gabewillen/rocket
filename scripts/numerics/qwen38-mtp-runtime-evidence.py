#!/usr/bin/env python3
"""Prove Qwen3.8 MTP execution from saved vLLM workload logs."""

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path


NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
METRIC = re.compile(
    rf"^\s*(?P<timestamp>\S+).*?SpecDecoding metrics:\s*"
    rf"Mean acceptance length:\s*(?P<mean_length>{NUMBER}),\s*"
    rf"Accepted throughput:\s*(?P<accepted_throughput>{NUMBER}) tokens/s,\s*"
    rf"Drafted throughput:\s*(?P<drafted_throughput>{NUMBER}) tokens/s,\s*"
    rf"Accepted:\s*(?P<accepted>\d+) tokens,\s*"
    rf"Drafted:\s*(?P<drafted>\d+) tokens,\s*"
    rf"Per-position acceptance rate:\s*(?P<positions>{NUMBER}(?:\s*,\s*{NUMBER})*),\s*"
    rf"Avg Draft acceptance rate:\s*(?P<average>{NUMBER})%\s*$"
)
MARKER = "SpecDecoding metrics:"
SCHEMA = "rocket.qwen38.mtp-runtime-evidence.v1"


def timestamp(value: str) -> dt.datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"invalid RFC3339 timestamp {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp has no UTC offset: {value!r}")
    return parsed.astimezone(dt.timezone.utc)


def parse_metrics(path: Path, not_before: dt.datetime, positions: int):
    fresh = []
    stale = 0
    marker_lines = 0
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if MARKER not in line:
            continue
        marker_lines += 1
        match = METRIC.match(line)
        if match is None:
            raise ValueError(f"malformed SpecDecoding metrics at line {line_number}")
        values = match.groupdict()
        observed_at = timestamp(values["timestamp"])
        per_position = [float(value.strip()) for value in values["positions"].split(",")]
        if len(per_position) != positions:
            raise ValueError(
                f"SpecDecoding metrics at line {line_number} has "
                f"{len(per_position)} positions, expected {positions}"
            )
        if any(not 0.0 <= value <= 1.0 for value in per_position):
            raise ValueError(
                f"SpecDecoding metrics at line {line_number} has an invalid position rate"
            )
        accepted = int(values["accepted"])
        drafted = int(values["drafted"])
        if drafted <= 0:
            raise ValueError(f"SpecDecoding metrics at line {line_number} drafted no tokens")
        if accepted > drafted:
            raise ValueError(
                f"SpecDecoding metrics at line {line_number}: "
                "accepted tokens exceed drafted tokens"
            )
        average = float(values["average"])
        calculated = accepted / drafted * 100.0
        if abs(average - calculated) > 0.11:
            raise ValueError(
                f"SpecDecoding metrics at line {line_number} has inconsistent "
                f"acceptance rate {average}% for {accepted}/{drafted} tokens"
            )
        record = {
            "timestamp": observed_at,
            "mean_length": float(values["mean_length"]),
            "accepted_throughput": float(values["accepted_throughput"]),
            "drafted_throughput": float(values["drafted_throughput"]),
            "accepted": accepted,
            "drafted": drafted,
            "per_position": per_position,
            "average": average,
        }
        if observed_at < not_before:
            stale += 1
        else:
            fresh.append(record)
    if marker_lines == 0:
        raise ValueError("no SpecDecoding metrics found")
    if not fresh:
        raise ValueError(
            f"SpecDecoding metrics are stale-only: {stale} record(s) before boundary"
        )
    return fresh, stale


def distribution(values):
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize(path: Path, not_before: dt.datetime, positions: int, min_records: int):
    records, stale = parse_metrics(path, not_before, positions)
    if len(records) < min_records:
        raise ValueError(
            f"only {len(records)} fresh SpecDecoding metric record(s), "
            f"require {min_records}"
        )
    accepted = sum(record["accepted"] for record in records)
    drafted = sum(record["drafted"] for record in records)
    return {
        "schema": SCHEMA,
        "verdict": "proven",
        "source_log": str(path),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence": {
            "records": len(records),
            "stale_records_ignored": stale,
            "not_before": not_before.isoformat().replace("+00:00", "Z"),
            "first_record": records[0]["timestamp"].isoformat().replace("+00:00", "Z"),
            "last_record": records[-1]["timestamp"].isoformat().replace("+00:00", "Z"),
            "expected_draft_positions": positions,
            "minimum_records": min_records,
        },
        "totals": {
            "accepted_tokens": accepted,
            "drafted_tokens": drafted,
            "acceptance_rate": accepted / drafted,
        },
        "mean_acceptance_length": distribution(
            [record["mean_length"] for record in records]
        ),
        "accepted_throughput_tokens_per_second": distribution(
            [record["accepted_throughput"] for record in records]
        ),
        "drafted_throughput_tokens_per_second": distribution(
            [record["drafted_throughput"] for record in records]
        ),
        "per_position_acceptance_rate": [
            distribution([record["per_position"][index] for record in records])
            for index in range(positions)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--not-before", required=True, type=timestamp)
    parser.add_argument("--min-records", type=int, default=2)
    parser.add_argument("--positions", type=int, default=3)
    args = parser.parse_args()
    if args.min_records < 1:
        parser.error("--min-records must be positive")
    if args.positions < 1:
        parser.error("--positions must be positive")
    try:
        result = summarize(
            args.log, args.not_before, args.positions, args.min_records
        )
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
