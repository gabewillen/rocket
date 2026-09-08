#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline-compile Rocket's fixed Qwen3.8 MTP FP8 MoE kernels for SM121."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


HIDDEN = tl.constexpr(2560)
LOGICAL_I = tl.constexpr(640)
PHYSICAL_I = tl.constexpr(768)
BLOCK_K = tl.constexpr(128)
BLOCK_N = tl.constexpr(128)


@triton.jit
def quantize_hidden(hidden, quantized, scales, rows: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    value = tl.load(hidden + row * HIDDEN + k).to(tl.float32)
    maximum = tl.max(tl.abs(value))
    scale = tl.maximum(maximum / 448.0, 1.0e-12)
    tl.store(quantized + row * HIDDEN + k, (value / scale).to(tl.float8e4nv))
    tl.store(scales + row * (HIDDEN // BLOCK_K) + block, scale)


@triton.jit
def gate_up_silu(
    quantized_hidden,
    hidden_scales,
    gate_up,
    consumer_active_routes,
    active_global_ids,
    local_to_active,
    owner_global_ids,
    owner_rows,
    grouped_permutation,
    gate_tables,
    gate_scale_tables,
    up_tables,
    up_scale_tables,
    rank: tl.constexpr,
):
    grouped = tl.program_id(0)
    tile_n = tl.program_id(1)
    active_routes = tl.load(consumer_active_routes)
    if grouped >= active_routes:
        return
    owner = tl.load(grouped_permutation + grouped)
    global_expert = tl.load(owner_global_ids + owner)
    local = global_expert - rank * 256
    active = tl.load(local_to_active + local)
    if active < 0 or tl.load(active_global_ids + active) != global_expert:
        return
    row = tl.load(owner_rows + owner)
    gate_base = tl.load(gate_tables + local).to(tl.pointer_type(tl.float8e4nv))
    up_base = tl.load(up_tables + local).to(tl.pointer_type(tl.float8e4nv))
    gate_sf = tl.load(gate_scale_tables + local).to(tl.pointer_type(tl.bfloat16))
    up_sf = tl.load(up_scale_tables + local).to(tl.pointer_type(tl.bfloat16))
    n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_gate = tl.zeros((1, BLOCK_N), tl.float32)
    acc_up = tl.zeros((1, BLOCK_N), tl.float32)
    for kb in tl.static_range(0, HIDDEN // BLOCK_K):
        k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(quantized_hidden + row * HIDDEN + k)[None, :]
        gate = tl.load(gate_base + n[:, None] * HIDDEN + k[None, :]).T
        up = tl.load(up_base + n[:, None] * HIDDEN + k[None, :]).T
        gate_scale = tl.load(gate_sf + tile_n * (HIDDEN // BLOCK_K) + kb)
        up_scale = tl.load(up_sf + tile_n * (HIDDEN // BLOCK_K) + kb)
        input_scale = tl.load(hidden_scales + row * (HIDDEN // BLOCK_K) + kb)
        acc_gate += tl.dot(a, gate) * (input_scale * gate_scale)
        acc_up += tl.dot(a, up) * (input_scale * up_scale)
    base = grouped * (2 * PHYSICAL_I)
    tl.store(gate_up + base + n, tl.reshape(acc_gate, (BLOCK_N,)).to(tl.bfloat16))
    tl.store(gate_up + base + PHYSICAL_I + n,
             tl.reshape(acc_up, (BLOCK_N,)).to(tl.bfloat16))


@triton.jit
def silu_mul_quantize(gate_up, activated, activated_scales, active_routes):
    grouped = tl.program_id(0)
    block = tl.program_id(1)
    if grouped >= tl.load(active_routes):
        return
    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    base = grouped * (2 * PHYSICAL_I)
    gate = tl.load(gate_up + base + k).to(tl.float32)
    up = tl.load(gate_up + base + PHYSICAL_I + k).to(tl.float32)
    value = (gate * tl.sigmoid(gate)) * up
    maximum = tl.max(tl.abs(value))
    scale = tl.maximum(maximum / 448.0, 1.0e-12)
    tl.store(activated + grouped * PHYSICAL_I + k,
             (value / scale).to(tl.float8e4nv))
    tl.store(activated_scales + grouped * (LOGICAL_I // BLOCK_K) + block, scale)


@triton.jit
def down_weighted_reduce(
    activated,
    activated_scales,
    output,
    consumer_active_routes,
    active_global_ids,
    local_to_active,
    owner_global_ids,
    owner_weights,
    owner_rows,
    grouped_permutation,
    down_tables,
    down_scale_tables,
    rank: tl.constexpr,
):
    grouped = tl.program_id(0)
    tile_n = tl.program_id(1)
    active_routes = tl.load(consumer_active_routes)
    if grouped >= active_routes:
        return
    owner = tl.load(grouped_permutation + grouped)
    global_expert = tl.load(owner_global_ids + owner)
    local = global_expert - rank * 256
    active = tl.load(local_to_active + local)
    if active < 0 or tl.load(active_global_ids + active) != global_expert:
        return
    row = tl.load(owner_rows + owner)
    route_weight = tl.load(owner_weights + owner)
    down_base = tl.load(down_tables + local).to(tl.pointer_type(tl.float8e4nv))
    down_sf = tl.load(down_scale_tables + local).to(tl.pointer_type(tl.bfloat16))
    n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((1, BLOCK_N), tl.float32)
    for kb in tl.static_range(0, LOGICAL_I // BLOCK_K):
        k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(activated + grouped * PHYSICAL_I + k)[None, :]
        weight = tl.load(down_base + n[:, None] * LOGICAL_I + k[None, :]).T
        scale = tl.load(down_sf + tile_n * (LOGICAL_I // BLOCK_K) + kb)
        input_scale = tl.load(
            activated_scales + grouped * (LOGICAL_I // BLOCK_K) + kb
        )
        accumulator += tl.dot(a, weight) * (input_scale * scale)
    tl.atomic_add(output + row * HIDDEN + n,
                  tl.reshape(accumulator, (BLOCK_N,)) * route_weight)


def compile_kernel(fn, signature: dict[str, str], constants: dict[str, int], out: Path) -> dict:
    compiled = triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", 121, 32),
        options={"num_warps": 4, "num_stages": 3},
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{fn.__name__}.cubin").write_bytes(compiled.asm["cubin"])
    metadata = dict(compiled.metadata._asdict())
    metadata["signature"] = signature
    metadata["constants"] = constants
    (out / f"{fn.__name__}.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_kernel(
        quantize_hidden,
        {"hidden": "*bf16", "quantized": "*fp8e4nv", "scales": "*fp32", "rows": "i32"},
        {"rows": 16},
        args.output,
    )
    compile_kernel(
        silu_mul_quantize,
        {"gate_up": "*bf16", "activated": "*fp8e4nv",
         "activated_scales": "*fp32", "active_routes": "*i32"},
        {},
        args.output,
    )
    pointer_signature = {
        "quantized_hidden": "*fp8e4nv",
        "hidden_scales": "*fp32",
        "gate_up": "*bf16",
        "consumer_active_routes": "*i32",
        "active_global_ids": "*i32",
        "local_to_active": "*i32",
        "owner_global_ids": "*i32",
        "owner_rows": "*i32",
        "grouped_permutation": "*i32",
        "gate_tables": "*u64",
        "gate_scale_tables": "*u64",
        "up_tables": "*u64",
        "up_scale_tables": "*u64",
        "rank": "i32",
    }
    first = compile_kernel(gate_up_silu, pointer_signature, {"rank": 0}, args.output / "rank0")
    compile_kernel(gate_up_silu, pointer_signature, {"rank": 1}, args.output / "rank1")
    down_signature = {
        "activated": "*fp8e4nv",
        "activated_scales": "*fp32",
        "output": "*fp32",
        "consumer_active_routes": "*i32",
        "active_global_ids": "*i32",
        "local_to_active": "*i32",
        "owner_global_ids": "*i32",
        "owner_weights": "*fp32",
        "owner_rows": "*i32",
        "grouped_permutation": "*i32",
        "down_tables": "*u64",
        "down_scale_tables": "*u64",
        "rank": "i32",
    }
    compile_kernel(down_weighted_reduce, down_signature, {"rank": 0}, args.output / "rank0")
    compile_kernel(down_weighted_reduce, down_signature, {"rank": 1}, args.output / "rank1")
    artifacts = {
        str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.rglob("*.cubin"))
    }
    manifest = {
        "schema": "rocket.qwen38-mtp-fp8-triton-aot.v1",
        "license": "Apache-2.0",
        "target": "cuda-sm121-warp32",
        "triton": triton.__version__,
        "source_checkpoint": {
            "schema": "rocket.qwen38-rank-slab.v1",
            "artifact": "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4",
            "revision": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
            "expert_abi": "fp8_e4m3_block_128x128",
        },
        "references": {
            "vllm_pinned": "8e685d198",
            "vllm_current_main": "869f78732b64454293d2ba42ae0008386fdaa6a6",
            "vllm_files": [
                "vllm/model_executor/layers/fused_moe/experts/triton_moe.py",
                "vllm/model_executor/layers/fused_moe/fused_moe.py",
            ],
            "pinned_triton_moe_sha256": "1ace6e60427c94fc17f16490475637f6988f88e7e4796f58a752a753dd7fd397",
            "pinned_fused_moe_sha256": "d5955a3460746b66740b024f470aeca79bbebd10872c327f765f2d7fcb805f28",
        },
        "artifacts": artifacts,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(first["name"], first["shared"], first["num_warps"])


if __name__ == "__main__":
    main()
