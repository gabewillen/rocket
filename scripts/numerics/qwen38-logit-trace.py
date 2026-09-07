#!/usr/bin/env python3
"""Capture or compare deterministic Qwen3.8 token/logprob traces."""

import argparse
import importlib.util
import json
import time
import urllib.request
from pathlib import Path


def quality_tasks():
    path = Path(__file__).with_name("qwen38-quality.py")
    spec = importlib.util.spec_from_file_location("qwen38_quality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TASKS


def capture(url, model, max_tokens, top_logprobs, seed, timeout):
    cases = {}
    for index, (name, prompt, _) in enumerate(quality_tasks()):
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0, "seed": seed + index,
                           "logprobs": True, "top_logprobs": top_logprobs}).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        choice = result["choices"][0]
        cases[name] = {"seconds": round(time.monotonic() - started, 3),
                       "tokens": choice["logprobs"]["content"], "finish_reason": choice["finish_reason"]}
        print(f"{name}\t{len(cases[name]['tokens'])} tokens", flush=True)
    return {"schema": "rocket.qwen38.logit-trace.v1", "model": model, "seed": seed,
            "max_tokens": max_tokens, "top_logprobs": top_logprobs, "cases": cases}


def compare(left, right):
    if set(left["cases"]) != set(right["cases"]):
        raise ValueError("case mismatch")
    rows, total, equal = {}, 0, 0
    for name in sorted(left["cases"]):
        a, b = left["cases"][name]["tokens"], right["cases"][name]["tokens"]
        shared = min(len(a), len(b)); first = None; chosen_drift = []; common_drift = []
        for index in range(shared):
            if a[index]["bytes"] != b[index]["bytes"] and first is None:
                first = index
            if a[index]["bytes"] == b[index]["bytes"]:
                equal += 1
                chosen_drift.append(abs(a[index]["logprob"] - b[index]["logprob"]))
                amap = {tuple(x["bytes"]): x["logprob"] for x in a[index]["top_logprobs"]}
                for item in b[index]["top_logprobs"]:
                    key = tuple(item["bytes"])
                    if key in amap:
                        common_drift.append(abs(amap[key] - item["logprob"]))
        total += max(len(a), len(b))
        rows[name] = {"baseline_tokens": len(a), "candidate_tokens": len(b),
                      "first_divergence": first, "shared_prefix_tokens": shared if first is None else first,
                      "max_chosen_logprob_drift": max(chosen_drift, default=0.0),
                      "max_common_top_logprob_drift": max(common_drift, default=0.0)}
    return {"schema": "rocket.qwen38.logit-comparison.v1", "token_positions": total,
            "equal_token_positions": equal, "token_parity": equal / total if total else 1.0, "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3817)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.candidate):
        parser.error("--baseline and --candidate are required together")
    if args.baseline:
        report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
        print(f"token_parity\t{report['token_parity']:.6f}")
    else:
        report = capture(args.url, args.model, args.max_tokens, args.top_logprobs, args.seed, args.timeout)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
