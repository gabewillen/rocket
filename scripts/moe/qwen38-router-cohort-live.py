#!/usr/bin/env python3
"""Bounded offline TP2 router-cohort capture for Qwen3.8 K4 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


SUPPORTED_CONCURRENCY = (1, 2, 4, 8, 16)
MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
ROUTER_METADATA = (
    "ROCKET_ROUTER_RANK",
    "ROCKET_ROUTER_VERIFY_WIDTH",
    "ROCKET_ROUTER_MAX_COHORTS",
    "ROCKET_ROUTER_COHORT",
    "ROCKET_ROUTER_SEQUENCES",
    "ROCKET_ROUTER_CACHE_BARRIER",
    "ROCKET_ROUTER_CACHE_BLOCK_SIZE",
)
VERIFY_WIDTH = 5
CAPTURE_CALLS = 4
# The pinned runtime completed c2 at 17 output tokens with only three exact
# width-5 verifier calls. Add a full K4 accepted-plus-bonus iteration and one
# token beyond that observed boundary before claiming four captured calls.
OBSERVED_THREE_CALL_TERMINAL_TOKENS = 17
CONSERVATIVE_ITERATION_MARGIN = VERIFY_WIDTH + 1
MIN_DECODE = (
    OBSERVED_THREE_CALL_TERMINAL_TOKENS + CONSERVATIVE_ITERATION_MARGIN + 1
)


def _publishable_cache_tokens(
    num_computed_tokens: int,
    block_size: int,
    num_reprefillable_tokens: int = 1,
) -> int:
    """Model pinned vLLM's finalized-token cache publication boundary."""
    finalized = max(0, num_computed_tokens - num_reprefillable_tokens)
    return finalized // block_size * block_size


class CohortContractError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _validate_c16_cache_counts(counts: list[int], expected: int) -> None:
    if len(counts) != 16 or any(count != expected for count in counts):
        raise CohortContractError(
            "cache_counts_mismatch",
            f"c16 cache barrier requires 16 cached counts equal to {expected}",
        )


def _observed_router_samples() -> list[dict]:
    for module in tuple(sys.modules.values()):
        samples = getattr(module, "_ROCKET_ROUTER_DIAGNOSTIC_SAMPLES", None)
        if isinstance(samples, list):
            return list(samples[:4])
    return []


_DIAGNOSTIC_PHASES = {
    "startup",
    "engine_initialization",
    "root_warmup",
    "prompt_construction",
    "cache_prime",
    "cache_barrier",
    "verifier",
    "complete",
}


def _failure_fields(error: BaseException | None, phase: str) -> tuple[str | None, str | None]:
    if error is None:
        return None, None
    failure_class = (
        "contract_error"
        if isinstance(error, CohortContractError)
        else "runtime_error"
        if isinstance(error, RuntimeError)
        else "argument_error"
        if isinstance(error, SystemExit)
        else "unexpected_error"
    )
    reason = getattr(error, "reason", None)
    failure_reason = (
        reason
        if reason == "cache_counts_mismatch"
        else f"{phase}_failed"
    )
    return failure_class, failure_reason


def _bounded_router_samples() -> list[dict]:
    bounded = []
    for sample in _observed_router_samples()[:4]:
        if not isinstance(sample, dict):
            continue
        bounded.append(
            {
                "channel": str(sample.get("channel", ""))[:96],
                "cohort_call": int(sample.get("cohort_call", 0)),
                "rank": int(sample.get("rank", -1)),
                "route_rows": int(sample.get("route_rows", 0)),
                "request_widths": [
                    int(item) for item in sample.get("request_widths", [])[:16]
                ],
                "row_offsets": [
                    int(item) for item in sample.get("row_offsets", [])[:17]
                ],
            }
        )
    return bounded


def _emit_diagnostic(state: dict, error: BaseException | None) -> None:
    failed = error is not None
    phase = state["phase"] if state["phase"] in _DIAGNOSTIC_PHASES else "startup"
    failure_class, failure_reason = _failure_fields(error, phase)
    counts = [
        int(item)
        for item in state.get("cached_prompt_tokens", [])[:16]
        if isinstance(item, int)
    ]
    token_text = state.get("continuation_token_text")
    print(
        "ROCKET_ROUTER_DIAGNOSTIC\t"
        + json.dumps(
            {
                "schema": "rocket.qwen38.router-diagnostic.v4",
                "phase": phase,
                "failure_class": failure_class,
                "failure_reason": failure_reason,
                "cached_prompt_tokens": counts,
                "continuation_token_id": state["continuation_token_id"],
                "continuation_token_sha256": state["continuation_token_sha256"],
                "continuation_token_text": (
                    token_text[:64]
                    if isinstance(token_text, str)
                    else None
                ),
                "router_samples": _bounded_router_samples(),
                "valid": not failed,
                "complete": not failed,
                "benchmark_accepted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _emit_diagnostic_without_masking(state: dict, error: BaseException | None) -> None:
    try:
        _emit_diagnostic(state, error)
    except BaseException:
        phase = state.get("phase", "startup")
        if phase not in _DIAGNOSTIC_PHASES:
            phase = "startup"
        failure_class, failure_reason = _failure_fields(error, phase)
        fallback = {
            "schema": "rocket.qwen38.router-diagnostic.v4",
            "phase": phase,
            "failure_class": failure_class or "diagnostic_error",
            "failure_reason": failure_reason or "diagnostic_serialization_failed",
            "cached_prompt_tokens": [],
            "continuation_token_id": None,
            "continuation_token_sha256": None,
            "continuation_token_text": None,
            "router_samples": [],
            "valid": False,
            "complete": False,
            "benchmark_accepted": False,
        }
        sys.stdout.write(
            "ROCKET_ROUTER_DIAGNOSTIC\t"
            + json.dumps(fallback, sort_keys=True)
            + "\n"
        )
        sys.stdout.flush()


def _run_with_diagnostics(operation) -> None:
    state = {
        "phase": "startup",
        "cached_prompt_tokens": [],
        "continuation_token_id": None,
        "continuation_token_sha256": None,
        "continuation_token_text": None,
    }
    try:
        operation(state)
    except BaseException as error:
        _emit_diagnostic_without_masking(state, error)
        raise
    _emit_diagnostic_without_masking(state, None)


def _run(state: dict) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concurrency", type=int, required=True, choices=SUPPORTED_CONCURRENCY
    )
    parser.add_argument("--decode", type=int, default=MIN_DECODE)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--divergence-tokens", type=int, default=128)
    parser.add_argument("--expected-cache-block-size", type=int, default=3216)
    args = parser.parse_args()
    if args.decode < MIN_DECODE:
        parser.error(
            f"--decode must be at least {MIN_DECODE} to guarantee four K4 calls"
        )
    if args.prefix_tokens < 1024 or args.prefix_tokens > 8192:
        parser.error("--prefix-tokens must be between 1024 and 8192")
    if args.divergence_tokens < 50 or args.divergence_tokens > 200:
        parser.error("--divergence-tokens must be between 50 and 200")
    if args.concurrency * args.divergence_tokens > 8192:
        parser.error("fresh cohort prefill exceeds max_num_batched_tokens")

    state["phase"] = "engine_initialization"
    from vllm import LLM, SamplingParams

    rank = int(os.environ["RANK"])
    inherited_metadata = [name for name in ROUTER_METADATA if name in os.environ]
    if inherited_metadata:
        raise RuntimeError(
            "router cohort metadata must be absent before initialization: "
            + ", ".join(inherited_metadata)
        )
    engine = LLM(
        model=MODEL,
        revision=REVISION,
        tensor_parallel_size=2,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
        speculative_config={"method": "mtp", "num_speculative_tokens": 4},
        enforce_eager=True,
        gpu_memory_utilization=0.835,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=262144,
        kv_cache_dtype="fp8",
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        enable_chunked_prefill=True,
        hf_overrides={"text_config": {"ple_embedding_dtype": "float8_e4m3fn"}},
    )
    tokenizer = engine.get_tokenizer()
    root_material = "Rocket agent session memory about GPU serving and systems. "
    root_ids = tokenizer.encode(
        root_material * (args.prefix_tokens // 8 + 2), add_special_tokens=False
    )[: args.prefix_tokens]
    if len(root_ids) != args.prefix_tokens:
        raise RuntimeError("failed to construct the requested root prefix")
    continuation_token_id = root_ids[0]
    continuation_token_text = tokenizer.decode([continuation_token_id])
    if continuation_token_id in set(tokenizer.all_special_ids):
        raise RuntimeError("c16 continuation token must not be an EOS or stop token")
    continuation_token_sha256 = hashlib.sha256(
        json.dumps([continuation_token_id], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    state.update(
        {
            "continuation_token_id": continuation_token_id,
            "continuation_token_sha256": continuation_token_sha256,
            "continuation_token_text": continuation_token_text,
        }
    )
    state["phase"] = "root_warmup"
    warm_sampling = SamplingParams(
        temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True
    )
    # Warm the shared long-context root while cohort metadata is wholly absent.
    # Finished prefix-cache blocks remain reusable by every fork in this process.
    warm_outputs = engine.generate(
        {"prompt_token_ids": root_ids}, warm_sampling, use_tqdm=False
    )
    if len(warm_outputs) != 1 or len(warm_outputs[0].outputs[0].token_ids) != 1:
        raise RuntimeError("root-prefix warmup did not finish exactly one token")

    concurrency = args.concurrency
    cohort = f"forked-prefix-c{concurrency}-k4"
    if concurrency == 16 and (
        args.prefix_tokens != 6304
        or args.divergence_tokens != 128
        or args.expected_cache_block_size != 3216
    ):
        raise RuntimeError(
            "c16 requires prefix=6304, divergence=128, and expected cache block size=3216"
        )
    prompts = []
    state["phase"] = "prompt_construction"
    for stream in range(concurrency):
        divergence = (
            f" Worker {stream} technical memory systems branch. "
            * (args.divergence_tokens // 6 + 2)
        )
        divergence_ids = tokenizer.encode(divergence, add_special_tokens=False)[
            : args.divergence_tokens
        ]
        if len(divergence_ids) != args.divergence_tokens:
            raise RuntimeError("failed to construct the requested divergence")
        prompts.append({"prompt_token_ids": root_ids + divergence_ids})

    cache_barrier = None
    cached_prompt_tokens = []
    if concurrency == 16:
        state["phase"] = "cache_prime"
        # Avoid the pinned hybrid-attention c16 failure on interleaved chunked
        # prefill and decode. Prime one complete prompt at a time while router
        # metadata is absent, then prove concurrent lookups hit each maximal
        # scheduler-owned prefix before opening the measured gate.
        measured_prompts = [
            {
                "prompt_token_ids": list(prompt["prompt_token_ids"])
                + [continuation_token_id]
            }
            for prompt in prompts
        ]
        print(
            "ROCKET_ROUTER_CACHE_PRIME_SETUP\t"
            + json.dumps(
                {
                    "continuation_token_id": continuation_token_id,
                    "continuation_token_sha256": continuation_token_sha256,
                    "continuation_token_text": continuation_token_text,
                    "continuation_is_special_or_stop": False,
                    "prime_prompt_tokens": 6433,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        prime_output_token_ids = []
        for prompt in measured_prompts:
            prime_outputs = engine.generate(prompt, warm_sampling, use_tqdm=False)
            if (
                len(prime_outputs) != 1
                or len(prime_outputs[0].outputs[0].token_ids) != 1
            ):
                raise RuntimeError("c16 full-prompt cache prime did not finish")
            prime_output_token_ids.append(prime_outputs[0].outputs[0].token_ids[0])
        prompts = measured_prompts
        state["phase"] = "cache_barrier"
        barrier_outputs = engine.generate(prompts, warm_sampling, use_tqdm=False)
        cached_prompt_tokens = [
            output.num_cached_tokens for output in barrier_outputs
        ]
        state["cached_prompt_tokens"] = cached_prompt_tokens
        print(
            "ROCKET_ROUTER_CACHE_BARRIER\t"
            + json.dumps(
                {
                    "cached_prompt_tokens": cached_prompt_tokens,
                    "expected_cache_block_size": args.expected_cache_block_size,
                    "expected_cached_tokens": 6432,
                    "prompt_tokens": 6433,
                    "requests": len(barrier_outputs),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _validate_c16_cache_counts(cached_prompt_tokens, 6432)
        for prompt, output in zip(prompts, barrier_outputs):
            prompt_tokens = len(prompt["prompt_token_ids"])
            expected_cached_tokens = (
                (prompt_tokens - 1)
                // args.expected_cache_block_size
                * args.expected_cache_block_size
            )
            if (
                len(output.outputs[0].token_ids) != 1
                or prompt_tokens != 6433
                or expected_cached_tokens != 6432
                or output.num_cached_tokens != expected_cached_tokens
            ):
                raise RuntimeError(
                    "c16 cache barrier requires the two-page prefix-cache hit: "
                    f"request_id={output.request_id} "
                    f"cached={output.num_cached_tokens} "
                    f"expected={expected_cached_tokens} "
                    f"block_size={args.expected_cache_block_size}"
                )
        cache_barrier = "two-cache-pages-v2"

    # Publish cohort metadata as one post-warmup transition. The c16 barrier
    # is included in the same transition, and any partial state is terminal in
    # the telemetry hook.
    cohort_metadata = {
        "ROCKET_ROUTER_RANK": str(rank),
        "ROCKET_ROUTER_VERIFY_WIDTH": str(VERIFY_WIDTH),
        "ROCKET_ROUTER_MAX_COHORTS": str(CAPTURE_CALLS),
        "ROCKET_ROUTER_COHORT": cohort,
        "ROCKET_ROUTER_SEQUENCES": str(concurrency),
    }
    if cache_barrier is not None:
        cohort_metadata["ROCKET_ROUTER_CACHE_BARRIER"] = cache_barrier
        cohort_metadata["ROCKET_ROUTER_CACHE_BLOCK_SIZE"] = str(
            args.expected_cache_block_size
        )
    os.environ.update(cohort_metadata)
    state["phase"] = "verifier"
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode,
        min_tokens=args.decode,
        ignore_eos=True,
    )
    if rank == 0:
        print(
            "ROCKET_ROUTER_RUN\t"
            + json.dumps(
                {
                    "schema": "rocket.qwen38.router-cohort-run.v3",
                    "telemetry_schema": "rocket.qwen38.activation-telemetry.v4",
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "revision": REVISION,
                    "concurrency": concurrency,
                    "verify_width": VERIFY_WIDTH,
                    "request_width_source": (
                        "vllm.forward_context.attn_metadata.query_start_loc"
                    ),
                    "top_k": 10,
                    "decode": args.decode,
                    "prompt_tokens": len(prompts[0]["prompt_token_ids"]),
                    "primed_continuation_tokens": 1 if concurrency == 16 else 0,
                    "pool_with_c1_c8_prompt_distribution": concurrency != 16,
                    "root_prefix_tokens": args.prefix_tokens,
                    "divergence_tokens": args.divergence_tokens,
                    "fresh_prefill_token_upper_bound": (
                        concurrency * args.divergence_tokens
                    ),
                    "cache_barrier": cache_barrier,
                    "attention_block_size": args.expected_cache_block_size,
                    "attention_block_size_proof": "all_request_cache_hit_counts",
                    "cache_pages": 2 if concurrency == 16 else None,
                    "cache_geometry": (
                        "two_cache_pages" if concurrency == 16 else None
                    ),
                    "cached_prompt_tokens": cached_prompt_tokens,
                    "continuation_token_id": continuation_token_id,
                    "continuation_token_sha256": continuation_token_sha256,
                    "continuation_token_text": continuation_token_text,
                    "prime_output_token_ids": (
                        prime_output_token_ids if concurrency == 16 else []
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    outputs = engine.generate(prompts, sampling, use_tqdm=False)
    if rank == 0:
        print(
            "ROCKET_ROUTER_COHORT_DONE\t"
            + json.dumps(
                {
                    "cohort": cohort,
                    "concurrency": concurrency,
                    "completion_tokens": [
                        len(item.outputs[0].token_ids) for item in outputs
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    state["phase"] = "complete"


if __name__ == "__main__":
    _run_with_diagnostics(_run)
