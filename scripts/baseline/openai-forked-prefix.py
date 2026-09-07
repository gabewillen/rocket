#!/usr/bin/env python3
"""Measure fixed-length decode throughput for a shared agent prefix."""

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def request(endpoint, model, prefix, stream_id, tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"Worker {stream_id}: write continuous technical prose about memory systems."},
        ],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    started = time.perf_counter()
    first = finished = None
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            if event.get("choices"):
                first = first or time.perf_counter()
                finished = time.perf_counter()
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", tokens),
        "ttft_s": (first - started) if first else None,
        "decode_s": (finished - first) if first and finished else None,
        "first_at": first,
        "finished_at": finished,
    }


def summarize(concurrency, rows, wall_s):
    decoded = sum(max(0, row["completion_tokens"] - 1) for row in rows)
    decode_wall_s = max(row["finished_at"] for row in rows) - min(
        row["first_at"] for row in rows)
    return {
        "concurrency": concurrency,
        "aggregate_tok_s": decoded / decode_wall_s,
        "per_stream_tok_s": statistics.mean(
            max(0, row["completion_tokens"] - 1) / row["decode_s"] for row in rows),
        "mean_ttft_s": statistics.mean(row["ttft_s"] for row in rows),
        "wall_s": wall_s,
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--decode", type=int, default=256)
    parser.add_argument("--prefix-bytes", type=int, default=65536)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    prefix = ("Rocket agent session memory. " * (args.prefix_bytes // 29 + 1))[:args.prefix_bytes]
    results = []
    for concurrency in map(int, args.concurrency.split(",")):
        started_at = utc_now()
        started_unix_ns = time.time_ns()
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request, args.endpoint, args.model, prefix, i,
                                   args.decode, args.timeout) for i in range(concurrency)]
            rows = [future.result() for future in futures]
        result = summarize(concurrency, rows, time.perf_counter() - started)
        result.update({
            "started_at": started_at,
            "started_unix_ns": started_unix_ns,
            "finished_at": utc_now(),
            "finished_unix_ns": time.time_ns(),
        })
        results.append(result)
        if not args.json:
            print("c={concurrency:2d} aggregate={aggregate_tok_s:7.2f} tok/s "
                  "stream={per_stream_tok_s:6.2f} tok/s ttft={mean_ttft_s:.3f}s".format(**result),
                  flush=True)
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
