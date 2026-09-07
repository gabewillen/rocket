#!/usr/bin/env python3
"""Deterministic pre/post-quantization quality gate for Qwen3.8 Flash Next."""

import argparse
import json
import re
import time
import urllib.request

TASKS = [
    ("arith_chain", "Compute step by step: ((17 * 23) + 149) / 4 - 37. Give the final numeric answer last.", ["98"]),
    ("trains", "Train A leaves at 15:00 travelling 80 km/h. Train B leaves the same station at 16:00 travelling 100 km/h on the same track. At what clock time does B catch A? Give the time as HH:MM.", ["20:00", "8:00 pm", "8 pm"]),
    ("logic_grid", "Alice, Bob and Carol have a cat, a dog and a bird, in some order. Alice does not have the bird. Bob does not have the dog and does not have the bird. Who has the bird? Answer with just the name.", ["carol"]),
    ("sqrt2", "Is the square root of 2 rational or irrational? Answer with one word.", ["irrational"]),
    ("counting", "How many times does the letter 'r' appear in 'strawberry raspberry'? Give a single integer.", ["6"]),
    ("units", "A tank with a 240 litre capacity starts full. It drains at 3 litres per minute for 20 minutes, then is refilled at 8 litres per minute for 15 minutes, overflow being discarded. How many litres are in it now? Give a number.", ["240"]),
    ("date", "If today is Wednesday, what day of the week is it in 100 days? Answer with the weekday name only.", ["friday"]),
    ("modular", "What is 7^100 mod 13? Give a single integer.", ["9"]),
    ("ordering", "Sort these from smallest to largest and give only the sorted list: 0.9, 0.85, 0.099, 0.891, 0.1.", ["0.099, 0.1, 0.85, 0.891, 0.9"]),
    ("negation", "Every glorp is a frimp. Some frimps are not blicks. Does it follow that some glorps are not blicks? Answer yes or no, then one sentence.", ["no"]),
]
FILLER = "Record {i:05d}: the depot logged a routine variance in the northbound inventory reconciliation for quarter {q}.\n"


def needle(depth, secret):
    rows = [FILLER.format(i=i, q=i % 4 + 1) for i in range(depth // 16)]
    midpoint = len(rows) // 2
    rows.insert(midpoint, f"Record {midpoint:05d}: the archive passphrase is {secret}.\n")
    return "".join(rows) + "\nWhat is the archive passphrase? Answer with the passphrase only."


def ask(url, model, prompt, max_tokens, timeout):
    data = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]
    return message.get("content") or "", result.get("usage", {}), time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--needles", default="8000,64000")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--compare")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = list(TASKS)
    for index, raw_depth in enumerate(filter(None, args.needles.split(","))):
        secret = f"BRIGHT-HARBOR-{9174 + index}"
        tasks.append((f"needle_{raw_depth}", needle(int(raw_depth), secret), [secret.lower()]))
    results = {}
    for name, prompt, accepted in tasks:
        content, usage, seconds = ask(args.url, args.model, prompt, args.max_tokens, args.timeout)
        answer = re.sub(r"<think>.*?</think>", " ", content, flags=re.I | re.S).lower()
        results[name] = {"ok": any(value in answer for value in accepted),
                         "answer_tail": content.strip()[-160:], "seconds": round(seconds, 1),
                         "prompt_tokens": usage.get("prompt_tokens"),
                         "completion_tokens": usage.get("completion_tokens"),
                         "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        print(f"{name}\t{'PASS' if results[name]['ok'] else 'FAIL'}\t{seconds:.1f}s", flush=True)
    report = {"passed": sum(row["ok"] for row in results.values()), "total": len(results), "results": results}
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(f"score\t{report['passed']}/{report['total']}")
    if args.compare:
        with open(args.compare, encoding="utf-8") as stream:
            baseline = json.load(stream)
        regressions = [name for name, row in results.items()
                       if baseline["results"].get(name, {}).get("ok") and not row["ok"]]
        print("regressions\t" + (",".join(regressions) or "none"))
        if regressions:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
