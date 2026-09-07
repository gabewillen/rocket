#!/usr/bin/env python3
"""Reduce exact Qwen router cohort records into rank-local slab traffic."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PREFIX = "ROCKET_NVFP4_TELEMETRY\t"
CHANNEL = re.compile(r"^layer\.(\d+)\.router\.topk\.output$")
GLOBAL_EXPERTS = 512
LOCAL_EXPERTS = 256
EXPECTED_LAYERS = 48
EXPECTED_TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640
SHARED_INTERMEDIATE = 640


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
    for line in path.read_text().splitlines():
        marker = line.find(PREFIX)
        if marker < 0:
            continue
        record = json.loads(line[marker + len(PREFIX) :])
        match = CHANNEL.match(record.get("channel", ""))
        if match:
            yield int(match.group(1)), record


def reduce(paths: list[Path]) -> dict:
    grouped = defaultdict(list)
    for path in paths:
        for layer, record in records(path):
            if record.get("schema") != "rocket.qwen38.activation-telemetry.v3":
                raise ValueError("router cohort requires telemetry schema v3")
            if record.get("route_top_k") != EXPECTED_TOP_K:
                raise ValueError("router cohort top-k is not 10")
            rank = record.get("rank")
            if rank not in (0, 1):
                raise ValueError("router cohort rank is invalid")
            grouped[(record["cohort"], rank, layer)].append(record)
    if not grouped:
        raise ValueError("no exact router cohort records found")

    cases = []
    cohort_ranks = sorted({(cohort, rank) for cohort, rank, _ in grouped})
    for cohort, rank in cohort_ranks:
        layer_rows = []
        layers = sorted(layer for c, r, layer in grouped if (c, r) == (cohort, rank))
        if layers != list(range(EXPECTED_LAYERS)):
            raise ValueError(f"{cohort} rank {rank} has {len(layers)}/48 layers")
        for layer in layers:
            samples = []
            for record in grouped[(cohort, rank, layer)]:
                first = rank * LOCAL_EXPERTS
                last = first + LOCAL_EXPERTS
                target_ids = set()
                speculative_ids = set()
                target_weight = 0.0
                speculative_weight = 0.0
                local_routes = 0
                for row in record["route_rows"]:
                    bucket = target_ids if row["position_kind"] == "target" else speculative_ids
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
        unions = [sample["union_unique_local_experts"] for row in layer_rows for sample in row["samples"]]
        bytes_values = [sample["routed_slab_bytes"] for row in layer_rows for sample in row["samples"]]
        first_record = grouped[(cohort, rank, 0)][0]
        cases.append(
            {
                "cohort": cohort,
                "rank": rank,
                "sequences": first_record["sequences"],
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
        "schema": "rocket.qwen38.router-cohort-summary.v1",
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
