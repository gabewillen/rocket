#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile and measure Rocket's fixed Qwen QSA FlashInfer CUTE fork."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from flashinfer.msa_ops.sparse_prefill import (
    _cutlass_dtype,
    _fake,
)
from vllm import __version__ as vllm_version
from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import (
    qsa_sparse_paged_attention,
)


HEADS = 12
PADDED_HEADS = 16
DIM = 256
TOPK = 2051
N_TILE = 16
TOKENS_PER_TILE = 4
VLLM_COMMIT = "8e685d198"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_kernel(kernel_class):
    cdt = _cutlass_dtype(torch.bfloat16)
    i32 = _cutlass_dtype(torch.int32)
    u8 = _cutlass_dtype(torch.uint8)
    f32 = _cutlass_dtype(torch.float32)
    symbols = [cute.sym_int() for _ in range(19)]
    kernel = kernel_class()
    return cute.compile(
        kernel,
        _fake(cdt, (symbols[0], PADDED_HEADS, DIM)),
        _fake(cdt, (symbols[2], 1, DIM)),
        _fake(cdt, (symbols[2], 1, DIM)),
        _fake(u8, (1,), align=4),
        _fake(u8, (1,), align=4),
        _fake(cdt, (symbols[0], PADDED_HEADS, DIM)),
        _fake(f32, (1, 1), align=4),
        _fake(f32, (1, 1), align=4),
        _fake(i32, (symbols[6], symbols[7], N_TILE), align=4),
        _fake(i32, (symbols[6], symbols[7], N_TILE), align=4),
        _fake(i32, (symbols[6],), align=4),
        _fake(i32, (symbols[6], 3), align=4),
        _fake(i32, (1,), align=4),
        _fake(i32, (symbols[4],), align=4),
        _fake(i32, (symbols[4],), align=4),
        _fake(i32, (symbols[5],), align=4),
        _fake(i32, (1, 1), align=4),
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def compile_builder(builder_class):
    i32 = _cutlass_dtype(torch.int32)
    symbols = [cute.sym_int() for _ in range(9)]
    return cute.compile(
        builder_class(TOPK, TOKENS_PER_TILE),
        _fake(i32, (symbols[0], symbols[1], TOPK), align=4),
        _fake(i32, (symbols[2],), align=4),
        _fake(i32, (symbols[2],), align=4),
        _fake(i32, (symbols[2],), align=4),
        _fake(i32, (symbols[2],), align=4),
        _fake(i32, (symbols[3], symbols[4]), align=4),
        _fake(i32, (symbols[3], symbols[4]), align=4),
        _fake(i32, (symbols[3],), align=4),
        _fake(i32, (symbols[3],), align=4),
        _fake(i32, (symbols[3], 3), align=4),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument("query_tokens", type=int, choices=(300, 8192))
    parser.add_argument(
        "--pattern",
        choices=("overlap", "disjoint", "identical", "causal-zero"),
        default="overlap",
    )
    parser.add_argument("--source-root", type=Path, default=Path("/rocket"))
    args = parser.parse_args()
    if VLLM_COMMIT not in vllm_version:
        raise SystemExit(f"expected vLLM {VLLM_COMMIT}, observed {vllm_version}")

    reference = load_module(
        args.source_root / "scripts/kernels/qwen38-qsa-prefill-vllm-reference.py",
        "qwen38_prefill_reference",
    )
    fork = load_module(
        args.source_root
        / "engines/qwen38-flash-next-nvfp4-2b/src/attention/qsa_prefill_cute.py",
        "qwen38_prefill_cute",
    )
    context_tokens = 8492 if args.query_tokens == 300 else 8192
    rows = args.sequences * args.query_tokens
    query = reference.formula(rows * HEADS * DIM, 17, 3, 127, 63, 64).view(
        rows, HEADS, DIM
    )
    query_padded = torch.zeros(
        (rows, PADDED_HEADS, DIM), dtype=torch.bfloat16, device="cuda"
    )
    query_padded[:, :HEADS].copy_(query)
    key_flat = reference.formula(
        args.sequences * context_tokens * DIM, 13, 5, 113, 56, 57
    ).view(args.sequences * context_tokens, 1, DIM)
    value_flat = reference.formula(
        args.sequences * context_tokens * DIM, 7, 11, 109, 54, 55
    ).view(args.sequences * context_tokens, 1, DIM)
    key_before = key_flat.clone()
    value_before = value_flat.clone()
    indices = reference.make_indices(
        args.sequences, args.query_tokens, context_tokens, args.pattern
    )
    q2k = indices.view(args.sequences, args.query_tokens, TOPK).reshape(
        1, rows, TOPK
    )
    cu_q = torch.arange(
        args.sequences + 1, dtype=torch.int32, device="cuda"
    ) * args.query_tokens
    cu_k = torch.arange(
        args.sequences + 1, dtype=torch.int32, device="cuda"
    ) * context_tokens
    q_offset_value = (
        0 if args.pattern == "causal-zero" else context_tokens - args.query_tokens
    )
    q_offset = torch.full(
        (args.sequences,), q_offset_value,
        dtype=torch.int32, device="cuda"
    )
    output = torch.zeros_like(query_padded)
    dummy_u8 = torch.zeros(1, dtype=torch.uint8, device="cuda")
    dummy_f32 = torch.zeros((1, 1), dtype=torch.float32, device="cuda")
    dummy_page = torch.zeros((1, 1), dtype=torch.int32, device="cuda")

    compiled = compile_kernel(fork.Qwen38QsaPrefillSm121)
    compiled_builder = compile_builder(fork.BuildUnionMetaSm12x)

    # Qwen's admitted prompt buckets are uniform and divisible by four. Build
    # their immutable tile geometry once, outside capture and timing. The same
    # caller-owned-workspace lesson appears in TensorRT-LLM c426264bc4d0,
    # sparseAttentionKernels.cu plus xqaDispatcher.cpp. Its page-bitset/CUB scan
    # does not express Qwen's exact per-token top-k membership masks, so Rocket
    # retains the attributed FlashInfer merge body and removes wrapper dispatch.
    tiles_per_sequence = args.query_tokens // TOKENS_PER_TILE
    capacity = args.sequences * tiles_per_sequence
    tile_batch = torch.arange(
        args.sequences, dtype=torch.int32, device="cuda"
    ).repeat_interleave(tiles_per_sequence)
    tile_t = torch.arange(
        tiles_per_sequence, dtype=torch.int32, device="cuda"
    ).repeat(args.sequences)
    tile_qbase = tile_batch * args.query_tokens + tile_t * TOKENS_PER_TILE
    tile_ntok = torch.full(
        (capacity,), TOKENS_PER_TILE, dtype=torch.int32, device="cuda"
    )
    padded_union = ((TOKENS_PER_TILE * TOPK + N_TILE - 1) // N_TILE) * N_TILE
    union_tokens_flat = torch.empty(
        (capacity, padded_union), dtype=torch.int32, device="cuda"
    )
    union_masks_flat = torch.empty_like(union_tokens_flat)
    union_token_count = torch.empty(
        (capacity,), dtype=torch.int32, device="cuda"
    )
    union_count = torch.empty_like(union_token_count)
    work_meta = torch.empty((capacity, 3), dtype=torch.int32, device="cuda")
    union_tokens = union_tokens_flat.view(capacity, -1, N_TILE)
    union_masks = union_masks_flat.view(capacity, -1, N_TILE)
    work_count = torch.full((1,), capacity, dtype=torch.int32, device="cuda")

    def build_metadata():
        compiled_builder(
            q2k, tile_batch, tile_t, tile_qbase, tile_ntok,
            union_tokens_flat, union_masks_flat, union_token_count,
            union_count, work_meta, 1, capacity,
        )

    def launch_kernel():
        compiled(
            query_padded, key_flat, value_flat, dummy_u8, dummy_u8, output,
            dummy_f32, dummy_f32, union_tokens, union_masks, union_count,
            work_meta, work_count, cu_q, cu_k, q_offset, dummy_page,
            DIM**-0.5, 1.0, 1.0, capacity,
        )

    def launch_stage():
        build_metadata()
        launch_kernel()

    launch_stage()
    torch.cuda.synchronize()
    if not torch.equal(key_before, key_flat) or not torch.equal(
        value_before, value_flat
    ):
        raise RuntimeError("QSA CUTE mutated causal state before timing")

    key_pages, table = reference.paged_state(
        key_flat.view(-1), args.sequences, context_tokens
    )
    value_pages, _ = reference.paged_state(
        value_flat.view(-1), args.sequences, context_tokens
    )
    request = torch.arange(
        args.sequences, dtype=torch.int32, device="cuda"
    ).repeat_interleave(args.query_tokens)
    control = qsa_sparse_paged_attention(
        query, key_pages, value_pages, indices, table, request
    )
    torch.cuda.synchronize()
    difference = (output[:, :HEADS].float() - control.float()).abs()
    maximum_error = difference.max().item()
    mean_error = difference.mean().item()
    if maximum_error > 0.001953125 or mean_error > 2.0e-5:
        row_error = difference.view(rows, -1).amax(dim=1)
        head_error = difference.amax(dim=(0, 2))
        first_count = union_count[0].item()
        row0_indices = indices[0][indices[0] >= 0].to(torch.int64)
        row0_scores = (
            query[0].float() @ key_flat[row0_indices, 0].float().T
        ) * (DIM**-0.5)
        row0_oracle = torch.softmax(row0_scores, dim=-1) @ value_flat[
            row0_indices, 0
        ].float()
        row0_cute_fp32 = (output[0, :HEADS].float() - row0_oracle).abs().max().item()
        row0_vllm_fp32 = (control[0].float() - row0_oracle).abs().max().item()
        raise RuntimeError(
            f"QSA CUTE parity failed before timing: max={maximum_error} "
            f"mean={mean_error} worst_row={row_error.argmax().item()} "
            f"row_min={row_error.min().item()} row_median={row_error.median().item()} "
            f"head_max={','.join(str(value) for value in head_error.tolist())} "
            f"row0_native={output[0, 0, :8].float().tolist()} "
            f"row0_control={control[0, 0, :8].float().tolist()} "
            f"row0_cute_fp32_max={row0_cute_fp32} "
            f"row0_vllm_fp32_max={row0_vllm_fp32} "
            f"first_union_count={first_count} "
            f"first_union_tokens={union_tokens[0, 0, :8].tolist()} "
            f"first_union_masks={union_masks[0, 0, :8].tolist()}"
        )

    metadata_samples = []
    for _ in range(7):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        build_metadata()
        end.record()
        end.synchronize()
        metadata_samples.append(begin.elapsed_time(end))

    kernel_samples = []
    for _ in range(7):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch_kernel()
        end.record()
        end.synchronize()
        kernel_samples.append(begin.elapsed_time(end))

    stage_samples = []
    for _ in range(7):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        launch_stage()
        end.record()
        end.synchronize()
        stage_samples.append(begin.elapsed_time(end))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_stage()
    graph.replay()
    torch.cuda.synchronize()
    replay = output[:, :HEADS].clone()
    graph.replay()
    torch.cuda.synchronize()
    graph_exact = torch.equal(replay, output[:, :HEADS])
    if not graph_exact:
        raise RuntimeError("QSA CUTE graph replay changed output")
    if not torch.equal(key_before, key_flat) or not torch.equal(
        value_before, value_flat
    ):
        raise RuntimeError("QSA CUTE mutated causal state after replay")

    selected_tokens = (indices >= 0).sum().item()
    union_tokens_total = union_token_count.sum().item()
    kv_bytes = union_tokens_total * 2 * DIM * 2
    query_output_bytes = rows * HEADS * DIM * 4

    print(
        f"sequences={args.sequences} query_tokens={args.query_tokens} "
        f"pattern={args.pattern} kernel_median_ms={statistics.median(kernel_samples)} "
        f"kernel_p95_ms={max(kernel_samples)} "
        f"metadata_median_ms={statistics.median(metadata_samples)} "
        f"stage_median_ms={statistics.median(stage_samples)} "
        f"stage_p95_ms={max(stage_samples)} "
        f"max_abs={maximum_error} "
        f"mean_abs={mean_error} graph_exact={str(graph_exact).lower()} "
        f"state_unchanged=true "
        f"union_tokens={union_tokens_total} selected_tokens={selected_tokens} "
        f"kv_bytes={kv_bytes} query_output_bytes={query_output_bytes} "
        f"logical_hot_bytes={kv_bytes + query_output_bytes} kernel_samples_ms="
        + ",".join(str(value) for value in kernel_samples)
        + " metadata_samples_ms="
        + ",".join(str(value) for value in metadata_samples)
        + " stage_samples_ms="
        + ",".join(str(value) for value in stage_samples)
    )


if __name__ == "__main__":
    main()
