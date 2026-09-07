#!/usr/bin/env python3
"""Exercise Qwen3.8 attention mechanisms through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


SCHEMA = "rocket.qwen38.attention-calibration.v1"
DEFAULT_SEED = 380_121
CHECKPOINT_FACTS = {
    "text_layers": {
        "count": 48,
        "pattern": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_attention_count": 36,
        "full_attention_count": 12,
    },
    "ple": {"layer_ids": [2]},
    "qsa": {"applies_to": "full_attention", "indexer_n_heads": 4},
    "moe": {"experts": 512, "top_k": 10},
    "mtp": {
        "hybrid": True,
        "num_hidden_layers": 1,
        "layer_type": "full_attention",
    },
}
FILLER = (
    "Ledger {index:05d}: station {station} recorded crate {crate:04d}, "
    "checksum {checksum:06d}, during the routine inventory pass.\n"
)


@dataclass(frozen=True)
class Case:
    name: str
    mechanisms: tuple[str, ...]
    prompt: str
    max_tokens: int = 128


def deterministic_ledger(seed: int, rows: int) -> list[str]:
    rng = random.Random(seed)
    return [
        FILLER.format(
            index=index,
            station=chr(ord("A") + rng.randrange(26)),
            crate=rng.randrange(10_000),
            checksum=rng.randrange(1_000_000),
        )
        for index in range(rows)
    ]


def needle_prompt(seed: int, position: str, near_matches: bool = False) -> str:
    rows = deterministic_ledger(seed, 192)
    positions = {"beginning": 4, "middle": len(rows) // 2, "end": len(rows) - 5}
    secret = f"ORBIT-{seed % 100_000:05d}-{position.upper()}"
    rows.insert(positions[position], f"AUTHORITY RECORD: recovery code is {secret}.\n")
    if near_matches:
        for offset, decoy in ((23, "ORBIT-00000-MIDDLE"), (141, "ORBIT-99999-MIDDLE")):
            rows.insert(offset, f"VOID DRAFT: obsolete recovery code was {decoy}.\n")
    return (
        "Read the ledger. Return only the recovery code from the AUTHORITY RECORD.\n\n"
        + "".join(rows)
    )


def build_cases(seed: int, long_decode_tokens: int) -> list[Case]:
    repeated = " ".join(["amber cobalt amber cobalt"] * 96)
    induction = "A=17 B=29 C=41 D=53. A=17 B=29 C=41 D="
    overlap = "banana bandana ananas banana bandana ananas " * 48
    return [
        Case(
            "recurrent_linear_attention_long_decode",
            ("recurrent_linear_attention", "long_decode"),
            "Write a numbered technical log with one short, unique observation per line. "
            "Continue until the token budget ends and never emit an ending sentence.",
            long_decode_tokens,
        ),
        Case(
            "repeated_alternating_patterns",
            ("recurrent_linear_attention", "repetition", "alternation"),
            f"Study this sequence, then continue its exact alternating pattern for 64 terms:\n{repeated}",
            256,
        ),
        Case(
            "abrupt_topic_shift",
            ("recurrent_linear_attention", "state_transition"),
            "For 80 lines discuss only marine biology. Then the marker [SHIFT-TO-CUDA] "
            "changes the task: after it, discuss only CUDA shared-memory bank conflicts. "
            "[SHIFT-TO-CUDA] Begin the CUDA section now.",
            384,
        ),
        Case(
            "exact_copy_induction",
            ("full_attention", "exact_copy", "induction"),
            f"Copy the text after COPY exactly, then complete its final association. COPY: {induction}",
            96,
        ),
        Case(
            "distant_needle_beginning",
            ("full_attention", "distant_needle", "beginning_retrieval"),
            needle_prompt(seed + 11, "beginning"),
            48,
        ),
        Case(
            "distant_needle_middle_near_match",
            ("full_attention", "distant_needle", "middle_retrieval", "near_match_distractors"),
            needle_prompt(seed + 23, "middle", near_matches=True),
            48,
        ),
        Case(
            "distant_needle_end",
            ("full_attention", "distant_needle", "end_retrieval"),
            needle_prompt(seed + 37, "end"),
            48,
        ),
        Case(
            "ple_overlapping_ngrams",
            ("ple", "overlapping_ngrams"),
            f"Count every overlapping occurrence of 'ana' in the corpus and explain the scan:\n{overlap}",
            160,
        ),
        Case(
            "code_json_structure",
            ("ple", "code_structure", "json_structure"),
            "Return valid JSON with keys source, transform, and checks. source is an array of "
            "integers 1 through 8. transform is Python source for squaring them. checks contains "
            "the expected output and a nested object with booleans for sorted and unique.",
            256,
        ),
        Case(
            "rare_multilingual_tokens",
            ("ple", "rare_tokens", "multilingual"),
            "Preserve each token exactly and return a JSON mapping to its 1-based position: "
            "𓂀, ꙮ, ڜ, ᚠ, क़, 漢字, 한글, Ελληνικά, naïve, 🛰️.",
            192,
        ),
    ]


def concurrent_cases(seed: int, count: int) -> list[Case]:
    cases = []
    for index in range(count):
        rows = 12 + index * 19
        ledger = "".join(deterministic_ledger(seed + 1000 + index, rows))
        cases.append(
            Case(
                f"concurrent_variable_length_{index:02d}",
                ("concurrent_variable_length_streams", "scheduler_interaction"),
                f"Summarize this {rows}-row ledger in exactly {8 + index} bullets:\n{ledger}",
                96 + index * 32,
            )
        )
    return cases


def coverage_manifest(cases: list[Case], concurrent: list[Case], mtp_requested: bool) -> dict[str, Any]:
    all_cases = cases + concurrent
    mechanisms: dict[str, list[str]] = {}
    for case in all_cases:
        for mechanism in case.mechanisms:
            mechanisms.setdefault(mechanism, []).append(case.name)
    if mtp_requested:
        mechanisms["mtp_speculation"] = ["mtp_speculative_decode"]
    return {
        "schema": SCHEMA,
        "checkpoint_facts": CHECKPOINT_FACTS,
        "cases": [
            {"name": case.name, "mechanisms": list(case.mechanisms), "max_tokens": case.max_tokens}
            for case in all_cases
        ],
        "mechanisms": mechanisms,
    }


def endpoint_url(endpoint: str, suffix: str) -> str:
    base = endpoint.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        base = base[: -len("/v1/chat/completions")]
    return base + suffix


def http_json(url: str, timeout: float, api_key: str | None, body: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def http_text(url: str, timeout: float, api_key: str | None) -> str:
    headers = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def speculative_metrics(text: str) -> dict[str, float]:
    """Sum speculative draft/accept counters in a Prometheus exposition."""
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            line,
        )
        if not match:
            continue
        name = match.group(1)
        lowered = name.lower()
        if "spec" not in lowered or not any(word in lowered for word in ("draft", "accept")):
            continue
        totals[name] = totals.get(name, 0.0) + float(match.group(2))
    return totals


def snapshot_speculative_metrics(
    endpoint: str, timeout: float, api_key: str | None
) -> dict[str, float] | None:
    try:
        return speculative_metrics(
            http_text(endpoint_url(endpoint, "/metrics"), timeout, api_key)
        )
    except (OSError, UnicodeError, ValueError, urllib.error.HTTPError):
        return None


def metric_deltas(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> dict[str, float]:
    if before is None or after is None:
        return {}
    return {
        name: after[name] - before.get(name, 0.0)
        for name in after
        if after[name] - before.get(name, 0.0) > 0
    }


def response_speculation_signal(value: Any, under_speculation: bool = False) -> dict[str, float]:
    """Find positive per-request draft evidence in a response extension."""
    found: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            child_under_speculation = under_speculation or "speculat" in lowered
            if (
                isinstance(child, (int, float))
                and not isinstance(child, bool)
                and child > 0
                and (
                    "draft" in lowered
                    or (child_under_speculation and "accept" in lowered)
                )
            ):
                found[key] = float(child)
            found.update(response_speculation_signal(child, child_under_speculation))
    elif isinstance(value, list):
        for child in value:
            found.update(response_speculation_signal(child, under_speculation))
    return found


def find_speculative_config(value: Any) -> Any:
    if isinstance(value, dict):
        if "speculative_config" in value:
            return value["speculative_config"]
        for child in value.values():
            found = find_speculative_config(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_speculative_config(child)
            if found is not None:
                return found
    return None


def discover_speculative_config(endpoint: str, timeout: float, api_key: str | None) -> Any:
    for suffix in ("/server_info", "/v1/models"):
        try:
            found = find_speculative_config(http_json(endpoint_url(endpoint, suffix), timeout, api_key))
        except (OSError, ValueError, urllib.error.HTTPError):
            continue
        if found is not None:
            return found
    return None


def run_case(endpoint: str, model: str, case: Case, seed: int, timeout: float,
             api_key: str | None, extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": case.max_tokens,
        "temperature": 0,
        "seed": seed,
    }
    if extra_body:
        body.update(extra_body)
    started = time.monotonic()
    try:
        response = http_json(endpoint_url(endpoint, "/v1/chat/completions"), timeout, api_key, body)
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        usage = response.get("usage") or {}
        result = {
            "status": "completed",
            "mechanisms": list(case.mechanisms),
            "seed": seed,
            "seconds": round(time.monotonic() - started, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "response_characters": len(content),
        }
        signal = response_speculation_signal(response)
        if signal:
            result["speculation_response_signal"] = signal
        return result
    except (OSError, KeyError, IndexError, TypeError, ValueError, urllib.error.HTTPError) as error:
        return {
            "status": "failed",
            "mechanisms": list(case.mechanisms),
            "seed": seed,
            "seconds": round(time.monotonic() - started, 3),
            "error": f"{type(error).__name__}: {error}",
        }


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_cases(args.seed, args.long_decode_tokens)
    concurrent_workload = concurrent_cases(args.seed, args.concurrent_streams)
    manifest = coverage_manifest(cases, concurrent_workload, args.mtp)
    results: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        results[case.name] = run_case(
            args.endpoint, args.model, case, args.seed + index, args.timeout, args.api_key
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(concurrent_workload)) as pool:
        future_cases = {
            pool.submit(
                run_case,
                args.endpoint,
                args.model,
                case,
                args.seed + 10_000 + index,
                args.timeout,
                args.api_key,
            ): case
            for index, case in enumerate(concurrent_workload)
        }
        for future, case in future_cases.items():
            results[case.name] = future.result()

    runtime_speculative_config = None
    if args.mtp:
        runtime_speculative_config = discover_speculative_config(
            args.endpoint, args.timeout, args.api_key
        )
    if args.mtp:
        name = "mtp_speculative_decode"
        if runtime_speculative_config is None:
            results[name] = {
                "status": "unavailable",
                "mechanisms": ["mtp_speculation"],
                "reason": "speculative_config is None",
            }
        else:
            case = Case(
                name,
                ("mtp_speculation", "long_decode"),
                "Continue this deterministic sequence for 256 terms: 3, 6, 12, 24, 48,",
                args.mtp_tokens,
            )
            metrics_before = snapshot_speculative_metrics(
                args.endpoint, args.timeout, args.api_key
            )
            mtp_result = run_case(
                args.endpoint,
                args.model,
                case,
                args.seed + 20_000,
                args.timeout,
                args.api_key,
            )
            metrics_after = snapshot_speculative_metrics(
                args.endpoint, args.timeout, args.api_key
            )
            deltas = metric_deltas(metrics_before, metrics_after)
            if mtp_result["status"] == "completed":
                if deltas:
                    mtp_result["speculation_metric_deltas"] = deltas
                if not deltas and "speculation_response_signal" not in mtp_result:
                    mtp_result["status"] = "configured_but_unverified"
                    mtp_result["reason"] = (
                        "runtime speculative_config is non-null, but no draft-attempt "
                        "or accepted-draft signal changed during the request"
                    )
            results[name] = mtp_result

    statuses = {status: sum(row["status"] == status for row in results.values())
                for status in (
                    "completed", "failed", "unavailable", "configured_but_unverified"
                )}
    return {
        "schema": SCHEMA,
        "seed": args.seed,
        "endpoint": args.endpoint,
        "model": args.model,
        "coverage_manifest": manifest,
        "mtp": {
            "requested": args.mtp,
            "checkpoint_capable": CHECKPOINT_FACTS["mtp"],
            "caller_config_supplied": args.speculative_config is not None,
            "runtime_config_discovered": (
                runtime_speculative_config is not None if args.mtp else None
            ),
            "speculative_config_available": (
                runtime_speculative_config is not None if args.mtp else None
            ),
            "runtime_coverage": (
                results.get("mtp_speculative_decode", {}).get("status")
                if args.mtp else "not_requested"
            ),
        },
        "summary": {"total": len(results), **statuses},
        "results": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--endpoint", default="http://127.0.0.1:8888")
    result.add_argument("--model", default="qwen3.8-flash-next")
    result.add_argument("--seed", type=int, default=DEFAULT_SEED)
    result.add_argument("--timeout", type=float, default=900)
    result.add_argument("--long-decode-tokens", type=int, default=2048)
    result.add_argument("--concurrent-streams", type=int, default=4)
    result.add_argument("--mtp", action="store_true")
    result.add_argument("--mtp-tokens", type=int, default=512)
    result.add_argument(
        "--speculative-config",
        type=parse_json_object,
        help="expected config hint for reporting only; never proves server activation",
    )
    result.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    result.add_argument("--out", type=pathlib.Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.concurrent_streams < 1:
        raise SystemExit("--concurrent-streams must be at least 1")
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
