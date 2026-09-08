#!/usr/bin/env python3
"""Reduce variable-width Qwen router cohorts into rank-local slab traffic."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PREFIX = "ROCKET_NVFP4_TELEMETRY\t"
CHANNEL = re.compile(r"^layer\.(\d+)\.router\.topk\.output$")
GLOBAL_EXPERTS = 512
LOCAL_EXPERTS = 256
EXPECTED_LAYERS = 48
EXPECTED_TOP_K = 10
EXPECTED_CALLS = (1, 2, 3, 4)
SCHEMA = "rocket.qwen38.activation-telemetry.v4"
SUMMARY_SCHEMA = "rocket.qwen38.router-cohort-summary.v2"
HIDDEN = 2560
INTERMEDIATE = 640
SHARED_INTERMEDIATE = 640
MAX_RECORD_CHARS = 1 << 20


def routed_slab_bytes(unique_experts: int) -> int:
    per_expert = (
        2 * INTERMEDIATE * (HIDDEN // 2)
        + HIDDEN * (INTERMEDIATE // 2)
        + 2 * INTERMEDIATE * (HIDDEN // 16)
        + HIDDEN * (INTERMEDIATE // 16)
        + 16
    )
    return unique_experts * per_expert


def shared_slab_bytes() -> int:
    return 3 * (SHARED_INTERMEDIATE // 2) * HIDDEN * 2 + HIDDEN * 2


def records(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            marker = line.find(PREFIX)
            if marker < 0:
                continue
            payload = line[marker + len(PREFIX) :]
            if len(payload) > MAX_RECORD_CHARS:
                raise ValueError(f"{path}:{line_number}: router record is too large")
            try:
                record = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid router telemetry JSON"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: router telemetry is not an object"
                )
            channel = record.get("channel")
            match = CHANNEL.match(channel) if isinstance(channel, str) else None
            if match:
                yield int(match.group(1)), record


def validate_record(record: dict) -> None:
    cohort_sequences = record.get("cohort_sequences")
    sequences = record.get("sequences")
    verify_width = record.get("verify_width")
    widths = record.get("request_widths")
    offsets = record.get("row_offsets")
    rows = record.get("route_rows")
    if type(cohort_sequences) is not int or not 1 <= cohort_sequences <= 16:
        raise ValueError("router cohort configured sequence count is invalid")
    if type(sequences) is not int or sequences < 1:
        raise ValueError("router cohort sequences is invalid")
    if sequences > cohort_sequences:
        raise ValueError("router cohort active sequences exceed configured cohort")
    if verify_width != 5:
        raise ValueError("router cohort verify_width is not 5")
    if (
        not isinstance(widths, list)
        or len(widths) != sequences
        or any(
            type(width) is not int or not 1 <= width <= verify_width
            for width in widths
        )
    ):
        raise ValueError("router cohort request widths are invalid")
    expected_offsets = [0]
    for width in widths:
        expected_offsets.append(expected_offsets[-1] + width)
    if (
        not isinstance(offsets, list)
        or any(type(offset) is not int for offset in offsets)
        or offsets != expected_offsets
    ):
        raise ValueError("router cohort row offsets are invalid")
    if record.get("route_layout") != "sequence_major":
        raise ValueError("router cohort route layout is invalid")
    if not isinstance(rows, list) or len(rows) != expected_offsets[-1]:
        raise ValueError("router cohort row count is incomplete")
    if record.get("selected_expert_count") != len(rows) * EXPECTED_TOP_K:
        raise ValueError("router cohort selected_expert_count is incomplete")
    expected_rows = []
    for sequence, (start, width) in enumerate(zip(offsets, widths)):
        expected_rows.extend(
            (start + position, sequence, position)
            for position in range(width)
        )
    expert_counts = Counter()
    for row, (index, sequence, position) in zip(rows, expected_rows):
        expected_kind = "target" if position == 0 else "speculative"
        if (
            type(row.get("row")) is not int
            or type(row.get("sequence")) is not int
            or type(row.get("position")) is not int
            or row.get("row") != index
            or row.get("sequence") != sequence
            or row.get("position") != position
            or row.get("position_kind") != expected_kind
        ):
            raise ValueError("router cohort row-major metadata is invalid")
        expert_ids = row.get("expert_ids")
        weights = row.get("weights")
        if (
            not isinstance(expert_ids, list)
            or not isinstance(weights, list)
            or len(expert_ids) != EXPECTED_TOP_K
            or len(weights) != EXPECTED_TOP_K
        ):
            raise ValueError("router cohort top-k row is incomplete")
        if any(
            type(expert_id) is not int
            or not 0 <= expert_id < GLOBAL_EXPERTS
            for expert_id in expert_ids
        ):
            raise ValueError("global expert id is outside E512")
        if len(set(expert_ids)) != EXPECTED_TOP_K:
            raise ValueError("router cohort top-k expert ids are not unique")
        expert_counts.update(expert_ids)
        if any(
            type(weight) not in (int, float)
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights
        ):
            raise ValueError("router cohort route weight is invalid")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=2e-6):
            raise ValueError("router cohort route weights are not normalized")
    top_experts = record.get("top_experts")
    expected_top_count = min(16, len(expert_counts))
    if not isinstance(top_experts, list) or len(top_experts) != expected_top_count:
        raise ValueError("router cohort top expert counts are incomplete")
    top_ids = []
    top_counts = []
    for item in top_experts:
        if not isinstance(item, dict):
            raise ValueError("router cohort top expert count is invalid")
        expert_id = item.get("expert_id")
        selections = item.get("selections")
        if (
            type(expert_id) is not int
            or not 0 <= expert_id < GLOBAL_EXPERTS
            or type(selections) is not int
            or selections < 1
            or expert_counts[expert_id] != selections
        ):
            raise ValueError("router cohort top expert count is invalid")
        top_ids.append(expert_id)
        top_counts.append(selections)
    if len(set(top_ids)) != len(top_ids) or sorted(top_counts, reverse=True) != sorted(
        expert_counts.values(), reverse=True
    )[:expected_top_count]:
        raise ValueError("router cohort top expert counts do not match route rows")


def reduce(paths: list[Path]) -> dict:
    grouped = defaultdict(list)
    path_ranks = defaultdict(set)
    for path in paths:
        for layer, record in records(path):
            if record.get("schema") != SCHEMA:
                raise ValueError("router cohort requires telemetry schema v4")
            if (
                type(record.get("route_top_k")) is not int
                or record["route_top_k"] != EXPECTED_TOP_K
            ):
                raise ValueError("router cohort top-k is not 10")
            cohort = record.get("cohort")
            if not isinstance(cohort, str) or not cohort:
                raise ValueError("router cohort identity is invalid")
            rank = record.get("rank")
            if type(rank) is not int or rank not in (0, 1):
                raise ValueError("router cohort rank is invalid")
            validate_record(record)
            path_ranks[path].add(rank)
            records_for_layer = grouped[(cohort, rank, layer)]
            records_for_layer.append(record)
            if len(records_for_layer) > len(EXPECTED_CALLS):
                raise ValueError("router cohort layer exceeds four calls")
    if not grouped:
        raise ValueError("no router cohort records found")
    if any(len(ranks) != 1 for ranks in path_ranks.values()):
        raise ValueError("router cohort input contains mixed ranks")

    cases = []
    cohorts = sorted({cohort for cohort, _, _ in grouped})
    for cohort in cohorts:
        ranks = sorted({rank for name, rank, _ in grouped if name == cohort})
        if ranks != [0, 1]:
            raise ValueError(f"{cohort} does not authenticate ranks 0 and 1")
    cohort_ranks = sorted({(cohort, rank) for cohort, rank, _ in grouped})
    cohort_schedules = {}
    for cohort, rank in cohort_ranks:
        layer_rows = []
        layers = sorted(layer for c, r, layer in grouped if (c, r) == (cohort, rank))
        if layers != list(range(EXPECTED_LAYERS)):
            raise ValueError(f"{cohort} rank {rank} has {len(layers)}/48 layers")
        for layer in layers:
            layer_records = grouped[(cohort, rank, layer)]
            cohort_calls = [record.get("cohort_call") for record in layer_records]
            if any(
                type(call) is not int for call in cohort_calls
            ) or cohort_calls != list(EXPECTED_CALLS):
                raise ValueError(
                    f"{cohort} rank {rank} layer {layer} does not have calls 1..4"
                )
            raw_calls = [record.get("call") for record in layer_records]
            if any(type(call) is not int for call in raw_calls) or any(
                right <= left for left, right in zip(raw_calls, raw_calls[1:])
            ):
                raise ValueError("router cohort raw call ids are not monotonic")
            schedule = tuple(
                (
                    record["cohort_call"],
                    record["sequences"],
                    tuple(record["request_widths"]),
                    tuple(record["row_offsets"]),
                )
                for record in layer_records
            )
            expected_schedule = cohort_schedules.setdefault(cohort, schedule)
            if schedule != expected_schedule:
                raise ValueError(
                    "router cohort request schedule changed across ranks or layers"
                )
            metadata = {
                (record["cohort_sequences"], record["verify_width"])
                for record in layer_records
            }
            if len(metadata) != 1:
                raise ValueError("router cohort layer metadata changed within capture")
            samples = []
            for record in layer_records:
                first = rank * LOCAL_EXPERTS
                last = first + LOCAL_EXPERTS
                target_ids = set()
                speculative_ids = set()
                target_weight = 0.0
                speculative_weight = 0.0
                local_routes = 0
                for row in record["route_rows"]:
                    bucket = (
                        target_ids
                        if row["position_kind"] == "target"
                        else speculative_ids
                    )
                    for expert_id, weight in zip(row["expert_ids"], row["weights"]):
                        if not 0 <= expert_id < GLOBAL_EXPERTS:
                            raise ValueError("global expert id is outside E512")
                        if first <= expert_id < last:
                            bucket.add(expert_id - first)
                            local_routes += 1
                            if row["position_kind"] == "target":
                                target_weight += weight
                            else:
                                speculative_weight += weight
                union = target_ids | speculative_ids
                samples.append(
                    {
                        "cohort_call": record["cohort_call"],
                        "call": record["call"],
                        "sequences": record["sequences"],
                        "request_widths": record["request_widths"],
                        "row_offsets": record["row_offsets"],
                        "route_rows": len(record["route_rows"]),
                        "target_rows": len(record["request_widths"]),
                        "speculative_rows": (
                            len(record["route_rows"])
                            - len(record["request_widths"])
                        ),
                        "target_unique_local_experts": len(target_ids),
                        "speculative_unique_local_experts": len(speculative_ids),
                        "union_unique_local_experts": len(union),
                        "local_routes": local_routes,
                        "target_local_weight": target_weight,
                        "speculative_local_weight": speculative_weight,
                        "routed_slab_bytes": routed_slab_bytes(len(union)),
                        "shared_slab_bytes": shared_slab_bytes(),
                    }
                )
            layer_rows.append({"layer": layer, "samples": samples})
        unions = [
            sample["union_unique_local_experts"]
            for row in layer_rows
            for sample in row["samples"]
        ]
        bytes_values = [
            sample["routed_slab_bytes"]
            for row in layer_rows
            for sample in row["samples"]
        ]
        first_record = grouped[(cohort, rank, 0)][0]
        cases.append(
            {
                "cohort": cohort,
                "rank": rank,
                "sequences": first_record["cohort_sequences"],
                "verify_width": first_record["verify_width"],
                "layers": layer_rows,
                "union_unique_local_experts": {
                    "min": min(unions),
                    "median": statistics.median(unions),
                    "max": max(unions),
                    "pstdev": statistics.pstdev(unions),
                },
                "routed_slab_bytes": {
                    "min": min(bytes_values),
                    "median": statistics.median(bytes_values),
                    "max": max(bytes_values),
                },
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "inputs": [str(path) for path in paths],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(reduce(args.logs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
