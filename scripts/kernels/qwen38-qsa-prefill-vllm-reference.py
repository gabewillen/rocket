#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the pinned vLLM QSA sparse-prefill control on Rocket's fixed tensors."""

from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path

import torch

from vllm import __version__ as vllm_version
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import (
    qsa_sparse_paged_attention,
)


HEADS = 12
DIM = 256
TOPK = 2051
PAGE = 64
VLLM_COMMIT = "8e685d198"


def formula(
    elements: int,
    multiplier: int,
    addend: int,
    modulus: int,
    center: int,
    divisor: int,
) -> torch.Tensor:
    result = torch.empty(elements, dtype=torch.bfloat16, device="cuda")
    chunk = 8 * 1024 * 1024
    for first in range(0, elements, chunk):
        end = min(first + chunk, elements)
        indices = torch.arange(first, end, dtype=torch.int64, device="cuda")
        values = ((indices * multiplier + addend) % modulus).to(torch.int32) - center
        result[first:end] = (values.float() / divisor).to(torch.bfloat16)
    return result


def make_indices(
    sequences: int, query_tokens: int, context_tokens: int, pattern: str
) -> torch.Tensor:
    result = torch.empty(
        (sequences * query_tokens, TOPK), dtype=torch.int32, device="cuda"
    )
    slots = torch.arange(TOPK, dtype=torch.int32, device="cuda")
    for sequence in range(sequences):
        for first_row in range(0, query_tokens, 256):
            count = min(256, query_tokens - first_row)
            local = torch.arange(
                first_row, first_row + count, dtype=torch.int32, device="cuda"
            )
            position = context_tokens - query_tokens + local
            first = torch.maximum(position - (TOPK - 1), torch.zeros_like(position))
            if pattern == "disjoint":
                first = torch.where((local & 1) != 0, 4096, 0)
            logical = first[:, None] + slots[None, :]
            logical = torch.where(
                (logical <= position[:, None]) & (logical < context_tokens),
                logical,
                -1,
            )
            begin = sequence * query_tokens + first_row
            result[begin : begin + count] = logical
    return result


def paged_state(
    flat: torch.Tensor, sequences: int, context_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pages_per_sequence = (context_tokens + PAGE - 1) // PAGE
    cache = torch.zeros(
        (sequences * pages_per_sequence, PAGE, 1, DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    logical = flat.view(sequences, context_tokens, DIM)
    for sequence in range(sequences):
        view = cache[
            sequence * pages_per_sequence : (sequence + 1) * pages_per_sequence
        ].view(-1, DIM)
        view[:context_tokens].copy_(logical[sequence])
    table = torch.arange(
        sequences * pages_per_sequence, dtype=torch.int32, device="cuda"
    ).view(sequences, pages_per_sequence)
    return cache, table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("query_tokens", type=int, choices=(300, 8192))
    parser.add_argument(
        "--pattern", choices=("overlap", "disjoint"), default="overlap"
    )
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if VLLM_COMMIT not in vllm_version:
        raise SystemExit(
            f"expected pinned vLLM {VLLM_COMMIT}, observed {vllm_version}"
        )
    if args.pattern == "disjoint" and args.query_tokens != 300:
        raise SystemExit("disjoint control is defined only for the 300-token burst")
    context_tokens = 8492 if args.query_tokens == 300 else 8192
    rows = args.sequences * args.query_tokens
    query = formula(rows * HEADS * DIM, 17, 3, 127, 63, 64).view(
        rows, HEADS, DIM
    )
    key_flat = formula(
        args.sequences * context_tokens * DIM, 13, 5, 113, 56, 57
    )
    value_flat = formula(
        args.sequences * context_tokens * DIM, 7, 11, 109, 54, 55
    )
    key, table = paged_state(key_flat, args.sequences, context_tokens)
    value, _ = paged_state(value_flat, args.sequences, context_tokens)
    indices = make_indices(
        args.sequences, args.query_tokens, context_tokens, args.pattern
    )
    token_to_request = torch.arange(
        args.sequences, dtype=torch.int32, device="cuda"
    ).repeat_interleave(args.query_tokens)
    output = torch.empty_like(query)

    qsa_sparse_paged_attention(
        query, key, value, indices, table, token_to_request, output
    )
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        qsa_sparse_paged_attention(
            query, key, value, indices, table, token_to_request, output
        )
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))

    host = output.cpu().contiguous()
    digest = hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()
    fields = {
        "sequences": args.sequences,
        "query_tokens": args.query_tokens,
        "context_tokens": context_tokens,
        "pattern": args.pattern,
        "vllm": vllm_version,
        "median_ms": statistics.median(samples),
        "p95_ms": max(samples),
        "sha256": digest,
        "samples_ms": ",".join(str(value) for value in samples),
    }
    if args.native_output:
        native = torch.from_file(
            str(args.native_output),
            shared=False,
            size=host.numel(),
            dtype=torch.bfloat16,
        ).view_as(host)
        difference = (native.float() - host.float()).abs()
        fields["native_max_abs"] = difference.max().item()
        fields["native_mean_abs"] = difference.mean().item()
    if args.output:
        args.output.write_bytes(host.view(torch.uint8).numpy().tobytes())
    print(" ".join(f"{key}={value}" for key, value in fields.items()))


if __name__ == "__main__":
    main()
