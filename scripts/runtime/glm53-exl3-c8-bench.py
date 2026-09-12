#!/usr/bin/env python3
"""c8 throughput bench for the EXL3 pair stack.

Fires B concurrent streaming completions of equal length and reports
aggregate tok/s, per-stream tok/s, and TTFT. Usage:

  python3 scripts/runtime/glm53-exl3-c8-bench.py --batch 8 --tokens 256
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"


STRUCTURED_PROMPT = (
    "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
)


def one(uid: int, prompt: str, tokens: int, thinking: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt if uid == 0 else f"[req{uid}] " + prompt}],
        "max_tokens": tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        # GLM-5.3 thinking off: the published decode receipts are measured in
        # this regime; thinking-on floods reasoning tokens and collapses
        # DFlash2 acceptance.
        "chat_template_kwargs": {"enable_thinking": thinking},
        # DFlash2 emits multi-token deltas (one event per verify step, all
        # accepted tokens); event counts are NOT token counts. Ask for usage.
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        ENDPOINT + "/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    first = None
    count = 0
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if first is None:
                first = time.monotonic() - t0
            chunk = json.loads(line[6:])
            if chunk.get("usage") is not None:
                count = chunk["usage"].get("completion_tokens", count)
                continue
            delta = chunk["choices"][0].get("delta", {}) if chunk.get("choices") else {}
            if delta.get("content") or delta.get("reasoning"):
                count += 1  # events; replaced by usage at stream end
    return {"uid": uid, "ttft": first, "tokens": count, "wall": time.monotonic() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--structured", action="store_true")
    args = ap.parse_args()
    prompt = "Explain paging. " * args.prompt_len if not args.structured else STRUCTURED_PROMPT

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.batch) as pool:
        results = list(pool.map(lambda u: one(u, prompt, args.tokens, args.thinking),
                                range(args.batch)))
    wall = time.monotonic() - t0

    total = sum(r["tokens"] for r in results)
    ttfts = sorted(r["ttft"] for r in results if r["ttft"] is not None)
    print(f"batch={args.batch} tokens/req={args.tokens}")
    print(f"aggregate : {total / wall:8.2f} tok/s  ({total} tokens in {wall:.1f} s)")
    print(f"per-stream: {total / wall / args.batch:8.2f} tok/s")
    if ttfts:
        print(f"ttft      : min {ttfts[0]:.2f}s  median {ttfts[len(ttfts)//2]:.2f}s  "
              f"max {ttfts[-1]:.2f}s")
    per = sorted(r["tokens"] / r["wall"] for r in results if r["wall"] > 0)
    print(f"stream rate: min {per[0]:.2f}  median {per[len(per)//2]:.2f}  max {per[-1]:.2f} tok/s")


if __name__ == "__main__":
    main()
