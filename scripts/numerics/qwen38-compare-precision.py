#!/usr/bin/env python3
"""Compare paired Qwen3.8 activation summaries and quality reports."""

import argparse
import json
import math
from pathlib import Path


def ratio(candidate, baseline):
    return candidate / baseline if baseline else (1.0 if candidate == 0 else math.inf)


def histogram_tv(candidate, baseline):
    ca, ba = sum(candidate), sum(baseline)
    if not ca or not ba:
        return 0.0 if ca == ba else 1.0
    return 0.5 * sum(abs(c / ca - b / ba) for c, b in zip(candidate, baseline))


def compare(baseline, candidate, baseline_quality, candidate_quality):
    left, right = baseline["telemetry"], candidate["telemetry"]
    if set(left) != set(right):
        raise ValueError(f"telemetry channel mismatch: missing={sorted(set(left)-set(right))}, extra={sorted(set(right)-set(left))}")
    rows = {}
    for channel in sorted(left):
        a, b = left[channel], right[channel]
        if a["kind"] != b["kind"] or len(a["histogram_log2"]) != len(b["histogram_log2"]):
            raise ValueError(f"incompatible channel: {channel}")
        rows[channel] = {
            "kind": a["kind"],
            "rms_ratio": ratio(b["rms"], a["rms"]),
            "p99_ratio": ratio(b["abs_p99"], a["abs_p99"]),
            "absmax_ratio": ratio(b["absmax"], a["absmax"]),
            "histogram_tv": histogram_tv(b["histogram_log2"], a["histogram_log2"]),
        }
    groups = {}
    for row in rows.values():
        group = groups.setdefault(row["kind"], {"channels": 0, "max_histogram_tv": 0.0, "max_rms_ratio": 0.0, "min_rms_ratio": math.inf})
        group["channels"] += 1
        group["max_histogram_tv"] = max(group["max_histogram_tv"], row["histogram_tv"])
        group["max_rms_ratio"] = max(group["max_rms_ratio"], row["rms_ratio"])
        group["min_rms_ratio"] = min(group["min_rms_ratio"], row["rms_ratio"])
    lb, rb = baseline_quality["results"], candidate_quality["results"]
    if set(lb) != set(rb):
        raise ValueError("quality case mismatch")
    parity = {name: lb[name].get("answer_tail") == rb[name].get("answer_tail") for name in sorted(lb)}
    regressions = [name for name in sorted(lb) if lb[name]["ok"] and not rb[name]["ok"]]
    return {"schema": "rocket.qwen38.precision-comparison.v1", "telemetry_channels": len(rows), "groups": groups, "channels": rows,
            "quality": {"cases": len(parity), "answer_tail_equal": sum(parity.values()), "parity": parity, "regressions": regressions}}


def main():
    parser = argparse.ArgumentParser()
    for name in ("baseline", "candidate", "baseline-quality", "candidate-quality", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    report = compare(load(args.baseline), load(args.candidate), load(args.baseline_quality), load(args.candidate_quality))
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"channels\t{report['telemetry_channels']}")
    print(f"answer_tail_equal\t{report['quality']['answer_tail_equal']}/{report['quality']['cases']}")
    print("regressions\t" + (",".join(report["quality"]["regressions"]) or "none"))


if __name__ == "__main__":
    main()
