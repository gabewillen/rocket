#!/usr/bin/env python3
"""Deterministic adaptive lazy-MTP policy and offline evidence replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


SCHEMA = "rocket.qwen38.adaptive-mtp-offline.v1"
PROBE_DEPTHS = (1, 1, 2, 1, 2, 3, 2, 3) * 2
PHASES = ("prefill", "restoring", "early_decode", "steady_decode")
BUCKETS = ("c1", "c2_4", "c5_8", "c9_16", "c17_plus")
REASONS = ("phase_k0", "remaining_cap", "probe", "policy", "explore", "promote", "demote")
METRIC = re.compile(
    r"SpecDecoding metrics:.*?Accepted:\s*(\d+) tokens,\s*Drafted:\s*(\d+) tokens,"
    r"\s*Per-position acceptance rate:\s*([0-9., ]+),"
)


def concurrency_bucket(concurrency: int) -> str:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if concurrency == 1:
        return "c1"
    if concurrency <= 4:
        return "c2_4"
    if concurrency <= 8:
        return "c5_8"
    if concurrency <= 16:
        return "c9_16"
    return "c17_plus"


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    scale = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / scale
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / scale
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass
class Evidence:
    successes: list[int] = field(default_factory=lambda: [0, 0, 0])
    trials: list[int] = field(default_factory=lambda: [0, 0, 0])

    def add_rates(self, rates: list[float], opportunities: int) -> None:
        if len(rates) != 3 or opportunities < 1:
            raise ValueError("three position rates and positive opportunities required")
        for index, rate in enumerate(rates):
            if not 0 <= rate <= 1:
                raise ValueError("position rate outside [0, 1]")
            self.successes[index] += round(rate * opportunities)
            self.trials[index] += opportunities

    def expected(self, depth: int, bound: str = "mean") -> float:
        result = 1.0
        for index in range(depth):
            lo, hi = wilson(self.successes[index], self.trials[index])
            if bound == "lower":
                result += lo
            elif bound == "upper":
                result += hi
            else:
                result += self.successes[index] / self.trials[index]
        return result


@dataclass(frozen=True)
class CostModel:
    step_bytes: tuple[float, float, float, float]
    lazy_weight_bytes: float
    residency_rounds: int

    def bytes_per_token(self, depth: int, accepted: float, resident: bool) -> float:
        lazy = 0.0
        if depth and not resident:
            lazy = self.lazy_weight_bytes / self.residency_rounds
        return (self.step_bytes[depth] + lazy) / accepted


def recommendation(evidence: Evidence, costs: CostModel, resident: bool = False) -> dict:
    rows = []
    for depth in range(4):
        lower = evidence.expected(depth, "lower")
        mean = evidence.expected(depth)
        upper = evidence.expected(depth, "upper")
        rows.append({
            "depth": depth,
            "accepted_tokens": {"lower": lower, "mean": mean, "upper": upper},
            "bytes_per_accepted_token": {
                "lower": costs.bytes_per_token(depth, upper, resident),
                "mean": costs.bytes_per_token(depth, mean, resident),
                "upper": costs.bytes_per_token(depth, lower, resident),
            },
        })
    selected = 0
    for candidate in range(1, 4):
        # Promote only when the candidate's pessimistic cost beats the incumbent's
        # optimistic cost by 3%. This is intentionally harder than a mean comparison.
        if (rows[candidate]["bytes_per_accepted_token"]["upper"]
                <= rows[selected]["bytes_per_accepted_token"]["lower"] * 0.97):
            selected = candidate
        else:
            break
    return {"selected_depth": selected, "depths": rows}


@dataclass
class CohortState:
    depth: int = 0
    eligible_rounds: int = 0
    decode_rounds: int = 0
    consecutive_failing_windows: int = 0
    window_results: list[bool] = field(default_factory=list)

    def choose(self, phase: str, concurrency: int, remaining_tokens: int) -> dict:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        bucket = concurrency_bucket(concurrency)
        if remaining_tokens < 1:
            raise ValueError("remaining_tokens must be positive")
        if phase in ("prefill", "restoring"):
            depth, reason = 0, "phase_k0"
        else:
            cap = min(3, max(0, remaining_tokens - 1))
            if self.decode_rounds < 16:
                depth, reason = min(PROBE_DEPTHS[self.decode_rounds], cap), "probe"
            elif self.eligible_rounds and self.eligible_rounds % 64 == 0:
                depth, reason = min(self.depth + 1, 3, cap), "explore"
            else:
                depth, reason = min(self.depth, cap), "policy"
            if depth < (PROBE_DEPTHS[self.decode_rounds] if self.decode_rounds < 16 else self.depth):
                reason = "remaining_cap"
            self.decode_rounds += 1
            self.eligible_rounds += 1
        return {
            "depth": depth,
            "labels": {"phase": phase, "concurrency": bucket, "depth": f"k{depth}", "reason": reason},
        }

    def apply_window(self, current_passes: bool, promoted_depth: int | None = None) -> str:
        self.window_results.append(current_passes)
        if len(self.window_results) < 8:
            return "policy"
        failed = not all(self.window_results)
        self.window_results.clear()
        self.consecutive_failing_windows = self.consecutive_failing_windows + 1 if failed else 0
        if self.consecutive_failing_windows >= 2 and self.depth:
            self.depth -= 1
            self.consecutive_failing_windows = 0
            return "demote"
        if not failed and promoted_depth is not None and promoted_depth > self.depth:
            self.depth = min(3, promoted_depth)
            return "promote"
        return "policy"


@dataclass
class AdaptivePolicy:
    """Own independent state for each bounded phase/concurrency cohort."""

    cohorts: dict[tuple[str, str], CohortState] = field(default_factory=dict)

    def choose(self, phase: str, concurrency: int, remaining_tokens: int) -> dict:
        bucket = concurrency_bucket(concurrency)
        key = (phase, bucket)
        state = self.cohorts.setdefault(key, CohortState())
        return state.choose(phase, concurrency, remaining_tokens)


def parse_raw_log(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if "SpecDecoding metrics:" not in line:
            continue
        match = METRIC.search(line)
        if match is None:
            raise ValueError(f"malformed MTP metric at {path}:{line_number}")
        rates = [float(value.strip()) for value in match.group(3).split(",")]
        if len(rates) != 3:
            raise ValueError(f"expected three positions at {path}:{line_number}")
        records.append({"accepted": int(match.group(1)), "drafted": int(match.group(2)), "rates": rates})
    if not records:
        raise ValueError(f"no MTP metrics in {path}")
    return records


def replay(evidence_paths: list[Path], costs: CostModel) -> dict:
    combined = Evidence()
    sources = []
    for evidence_path in sorted(evidence_paths):
        payload = json.loads(evidence_path.read_text())
        log_path = Path(payload["source_log"])
        digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
        if digest != payload["input_sha256"]:
            raise ValueError(f"raw log digest mismatch for {evidence_path}")
        records = parse_raw_log(log_path)
        for record in records:
            opportunities = max(1, record["drafted"] // 3)
            combined.add_rates(record["rates"], opportunities)
        sources.append({
            "evidence": str(evidence_path), "raw_log": str(log_path),
            "raw_log_sha256": digest, "records": len(records),
            "accepted_tokens": payload["totals"]["accepted_tokens"],
            "drafted_tokens": payload["totals"]["drafted_tokens"],
        })
    result = recommendation(combined, costs)
    return {
        "schema": SCHEMA,
        "scope": "offline control proof; no live engine or matched K0-K3 throughput claim",
        "cohort": {"phase": "steady_decode", "concurrency": "c2_4"},
        "sources": sources,
        "observations": {"successes": combined.successes, "trials": combined.trials},
        "cost_model": {
            "formula": "(step_bytes[depth] + lazy_weight_bytes/residency_rounds when cold) / accepted_tokens",
            "source": "scripts/numerics/qwen38-roofline.py checkpoint-derived c16 uniform-route traffic",
            "checkpoint_revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
            "step_bytes": list(costs.step_bytes), "lazy_weight_bytes": costs.lazy_weight_bytes,
            "residency_rounds": costs.residency_rounds,
        },
        "decision": result,
        "policy": {
            "depths": [0, 1, 2, 3], "probe_depths": list(PROBE_DEPTHS),
            "promotion_margin": 0.03, "demotion": "two failing 8-round windows",
            "exploration": "one depth deeper every 64 eligible rounds",
            "forced_k0_phases": ["prefill", "restoring"],
            "remaining_token_cap": "depth <= remaining_tokens - 1",
        },
        "otel": {
            "metric": "rocket.scheduler.mtp.decision",
            "bounded_labels": {"phase": list(PHASES), "concurrency": list(BUCKETS),
                               "depth": ["k0", "k1", "k2", "k3"], "reason": list(REASONS)},
            "identifiers": "request and session IDs are forbidden on metrics; sampled spans/logs only",
        },
        "remaining_proof": "matched live K0, K1, K2, K3 ladder at each concurrency bucket",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step-bytes", default="27014478469.877693,43058601574.859886,55338212408.54245,64872150224.303696")
    parser.add_argument("--lazy-weight-bytes", type=float, default=2698026496)
    parser.add_argument("--residency-rounds", type=int, default=64)
    args = parser.parse_args()
    values = tuple(float(value) for value in args.step_bytes.split(","))
    if len(values) != 4 or any(value <= 0 for value in values):
        parser.error("--step-bytes requires four positive comma-separated values")
    if args.lazy_weight_bytes < 0 or args.residency_rounds < 1:
        parser.error("lazy weight bytes and residency rounds are invalid")
    payload = replay(args.evidence, CostModel(values, args.lazy_weight_bytes, args.residency_rounds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
