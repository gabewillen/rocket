#!/usr/bin/env python3
"""Aggregate repeated coding workload summaries without counting rejected drafts."""

import argparse
import json
import pathlib
import statistics


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * q)))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path)
    a = p.parse_args()
    summaries = [json.loads(path.read_text()) for path in a.runs]
    phases = [phase for summary in summaries for phase in summary["runs"]]
    useful = sum(x["useful_output_tokens"] for x in phases)
    decode_s = sum(x.get("decode_ms", 0) for x in phases) / 1000
    completion_s = sum(x.get("decode_ms", 0) + x.get("prefill_ms", 0) for x in phases) / 1000
    throughputs = [x["aggregate_useful_tok_s"] for x in phases]
    result = {
        "schema": "rocket.glm53.coding-summary.v1",
        "run_count": len(summaries),
        "phase_count": len(phases),
        "useful_output_tokens": useful,
        "aggregate_decode_tok_s": useful / decode_s,
        "aggregate_completion_tok_s": useful / completion_s,
        "phase_tok_s_median": statistics.median(throughputs),
        "phase_tok_s_min": min(throughputs),
        "phase_tok_s_max": max(throughputs),
        "phase_tok_s_population_stdev": statistics.pstdev(throughputs),
        "ttft_ms_p50": percentile([x["ttft_ms_p50"] for x in phases], 0.5),
        "ttft_ms_p95": percentile([x["ttft_ms_p95"] for x in phases], 0.95),
        "inter_token_ms_p50": percentile([x["inter_token_ms_p50"] for x in phases], 0.5),
        "inter_token_ms_p95": percentile([x["inter_token_ms_p95"] for x in phases], 0.95),
        "completion_ms_p50": percentile([x["completion_ms_p50"] for x in phases], 0.5),
        "completion_ms_p95": percentile([x["completion_ms_p95"] for x in phases], 0.95),
        "all_replacements_preserved": all(x.get("replacement_preserved_active_slots", True) for x in phases),
        "expert_cache_misses": sum(x.get("expert_cache_misses", 0) for x in phases),
        "prefix_record_hits": sum(x.get("prefix_record_hits", 0) for x in phases),
        "prefix_record_misses": sum(x.get("prefix_record_misses", 0) for x in phases),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.output: a.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
