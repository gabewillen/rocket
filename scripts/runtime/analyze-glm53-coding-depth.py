#!/usr/bin/env python3
"""Select a useful-throughput draft depth with latency and acceptance evidence."""

import argparse
import glob
import json
import pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True)
    p.add_argument("--output", type=pathlib.Path)
    a = p.parse_args()
    rows = []
    for name in sorted(glob.glob(a.glob)):
        summary = json.load(open(name))
        for run in summary["runs"]:
            accepted = run.get("accepted_drafts_by_stream", [])
            drafted = run.get("drafted_by_stream", [])
            ratios = [x / y for x, y in zip(accepted, drafted) if y]
            rows.append({
                "path": name,
                "spec_k": run["spec_k"],
                "aggregate_useful_tok_s": run["aggregate_useful_tok_s"],
                "completion_ms_p95": run["completion_ms_p95"],
                "acceptance_mean": sum(ratios) / len(ratios) if ratios else 0.0,
                "acceptance_p10": sorted(ratios)[max(0, int(len(ratios) * 0.1) - 1)] if ratios else 0.0,
                "stage_ms": run.get("stage_ms", {}),
            })
    if not rows:
        raise SystemExit("no runs matched")
    best = max(rows, key=lambda row: row["aggregate_useful_tok_s"])
    result = {"schema": "rocket.glm53.coding-depth.v1", "best": best, "rows": rows}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.output: a.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
