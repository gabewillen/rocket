#!/usr/bin/env python3
"""Bounded offline TP2 router-cohort capture for Qwen3.8 K4 verification."""

from __future__ import annotations

import argparse
import json
import os
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


def main() -> None:
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
        # Avoid the pinned hybrid-attention c16 failure on interleaved chunked
        # prefill and decode. Prime one complete prompt at a time while router
        # metadata is absent, then prove concurrent lookups hit each maximal
        # scheduler-owned prefix before opening the measured gate.
        measured_prompts = []
        for prompt in prompts:
            prime_outputs = engine.generate(prompt, warm_sampling, use_tqdm=False)
            if (
                len(prime_outputs) != 1
                or len(prime_outputs[0].outputs[0].token_ids) != 1
            ):
                raise RuntimeError("c16 full-prompt cache prime did not finish")
            # vLLM intentionally caps a new request's prefix-cache hit at
            # prompt_tokens - 1 so it can recompute logits. Append the sampled
            # continuation to make that cap land exactly after the complete
            # original prompt, leaving one target token for cached prefill.
            measured_prompts.append(
                {
                    "prompt_token_ids": prompt["prompt_token_ids"]
                    + list(prime_outputs[0].outputs[0].token_ids)
                }
            )
        prompts = measured_prompts
        barrier_outputs = engine.generate(prompts, warm_sampling, use_tqdm=False)
        cached_prompt_tokens = [
            output.num_cached_tokens for output in barrier_outputs
        ]
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
        if len(barrier_outputs) != concurrency:
            raise RuntimeError("c16 cache barrier returned the wrong request count")
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


if __name__ == "__main__":
    main()
