#!/usr/bin/env python3
"""Bounded offline TP2 router-cohort capture for Qwen3.8 K4 verification."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone


CONCURRENCY = (1, 2, 4, 8, 16)
MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--prefix-bytes", type=int, default=65536)
    args = parser.parse_args()
    if args.decode < 8:
        parser.error("--decode must be at least 8 to reach steady K4 verification")

    from vllm import LLM, SamplingParams

    rank = int(os.environ["RANK"])
    os.environ["ROCKET_ROUTER_RANK"] = str(rank)
    os.environ["ROCKET_ROUTER_VERIFY_WIDTH"] = "5"
    os.environ["ROCKET_ROUTER_MAX_COHORTS"] = "4"

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
    prefix = ("Rocket agent session memory. " * (args.prefix_bytes // 29 + 1))[
        : args.prefix_bytes
    ]
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
                    "schema": "rocket.qwen38.router-cohort-run.v1",
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "revision": REVISION,
                    "concurrency": CONCURRENCY,
                    "verify_width": 5,
                    "top_k": 10,
                    "decode": args.decode,
                    "prefix_bytes": args.prefix_bytes,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for concurrency in CONCURRENCY:
        cohort = f"forked-prefix-c{concurrency}-k4"
        os.environ["ROCKET_ROUTER_COHORT"] = cohort
        os.environ["ROCKET_ROUTER_SEQUENCES"] = str(concurrency)
        prompts = [
            prefix
            + f"\nWorker {stream}: write continuous technical prose about memory systems."
            for stream in range(concurrency)
        ]
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        if rank == 0:
            print(
                "ROCKET_ROUTER_COHORT_DONE\t"
                + json.dumps(
                    {
                        "cohort": cohort,
                        "concurrency": concurrency,
                        "completion_tokens": [len(item.outputs[0].token_ids) for item in outputs],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
