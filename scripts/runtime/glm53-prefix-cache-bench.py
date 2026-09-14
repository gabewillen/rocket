#!/usr/bin/env python3
"""Generate an exact-length shared-prefix corpus and summarize cache runs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def make_prompt(path: Path, minimum_tokens: int, tokenizer_path: Path) -> int:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    unit = (
        "You are operating a repository maintenance agent. Read the current state, "
        "preserve every stated invariant, report measurements with units, and continue "
        "from the latest verified checkpoint. "
    )
    text = unit
    while len(tokenizer.encode(text).ids) < minimum_tokens + 32:
        text += unit
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return len(tokenizer.encode(text).ids)


def summarize(log: Path) -> dict[str, str]:
    text = log.read_text(errors="replace")
    fields: dict[str, str] = {}
    patterns = {
        "prompt_tokens": r"prompt\s+(\d+) tokens:",
        "restored_tokens": r"prefix\s+restored (\d+)/",
        "checkpoint_tokens": r"prefix\s+checkpointed (\d+)",
        "prefill_ms": r"prefill\s+([0-9.]+) ms",
        "aggregate_tps": r"\(([0-9.]+) agg tok/s",
        "stream_tps": r"([0-9.]+) tok/s/stream",
        "acceptance": r"\(([0-9.]+)% draft acceptance\)",
        "read_gib": r"prefix IO: read ([0-9.]+) GiB",
        "write_gib": r"prefix IO: read [0-9.]+ GiB write ([0-9.]+) GiB",
        "restore_ms": r"prefix IO:.* restore ([0-9.]+) ms",
        "writeback_ms": r"prefix IO:.* writeback ([0-9.]+) ms",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        fields[name] = match.group(1) if match else ""
    token_block = text.split("--- token ids", 1)[-1].split("--- timing", 1)[0]
    fields["token_ids"] = ",".join(re.findall(r"\d+:(\d+) \|", token_block))
    tokens = int(fields["restored_tokens"] or fields["checkpoint_tokens"] or 0)
    read_gib = float(fields["read_gib"] or 0)
    write_gib = float(fields["write_gib"] or 0)
    restore_s = float(fields["restore_ms"] or 0) / 1000
    write_s = float(fields["writeback_ms"] or 0) / 1000
    fields["bytes_per_token"] = f"{read_gib * (1 << 30) / tokens:.0f}" if tokens and read_gib else ""
    fields["read_gib_s"] = f"{read_gib / restore_s:.2f}" if restore_s else ""
    fields["write_gib_s"] = f"{write_gib / write_s:.2f}" if write_s else ""
    return fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-prompt", type=Path)
    parser.add_argument("--minimum-tokens", type=int, default=32768)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--summarize", type=Path, nargs="*")
    args = parser.parse_args()
    if args.make_prompt:
        if not args.tokenizer:
            parser.error("--tokenizer is required with --make-prompt")
        count = make_prompt(args.make_prompt, args.minimum_tokens, args.tokenizer)
        print(f"wrote {args.make_prompt} with {count} tokenizer tokens")
    for log in args.summarize or []:
        values = summarize(log)
        print(log, " ".join(f"{key}={value}" for key, value in values.items()))


if __name__ == "__main__":
    main()
