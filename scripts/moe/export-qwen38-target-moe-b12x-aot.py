#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the pinned fixed-c1 target MoE kernel with an explicit stream ABI.

Python, Torch, and CuTe DSL are build-time dependencies only.  The generated
header and relocatable object accept caller-owned device buffers and a borrowed
CUDA stream; the serving process does not load Python or TVM FFI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PINNED_FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"
PINNED_DISPATCH_SHA256 = (
    "61cb17593721d1bacc04c9a4dcc52238d3d99c531f3c6ae01c3d33971448a942"
)
PINNED_KERNEL_SHA256 = (
    "6dbd27625bc059c1246b3ef8445312e84ac76a8f1825ae3fb518f27919d8af64"
)
PINNED_TVM_FFI_OBJECT_SHA256 = (
    "8cc49bdb4163b07338818bb7db482aea812d91cc1cbf3ef1eeecf0ee2756fef4"
)


@dataclass(frozen=True)
class Plan:
    tokens: int = 1
    hidden: int = 2560
    logical_intermediate: int = 640
    physical_intermediate: int = 768
    top_k: int = 10
    max_rows: int = 10
    local_experts: int = 256
    state_experts: int = 257
    tile_m: int = 64
    tile_n: int = 128
    max_active_clusters: int = 20
    activation: str = "silu"
    fast_math: bool = True
    source_format: str = "modelopt"


PLAN = Plan()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def authenticate_exported_abi(header: Path, object_file: Path) -> None:
    header_text = header.read_text(encoding="utf-8")
    object_bytes = object_file.read_bytes()
    required = (
        "qwen38_target_moe_b12x_c1_Kernel_Module_t",
        "cute_dsl_qwen38_target_moe_b12x_c1_wrapper",
        "cudaStream_t stream",
    )
    if any(item not in header_text for item in required):
        raise RuntimeError("target MoE exported C ABI changed")
    if b"TVMFFIEnvGetStream" in object_bytes or b"__tvm_ffi" in object_bytes:
        raise RuntimeError("target MoE export retained the TVM-FFI runtime ABI")


def authenticate_sources() -> tuple[Path, Path]:
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_static_kernel

    dispatch_path = Path(moe_dispatch.__file__).resolve()
    kernel_path = Path(moe_static_kernel.__file__).resolve()
    observed = (digest(dispatch_path), digest(kernel_path))
    expected = (PINNED_DISPATCH_SHA256, PINNED_KERNEL_SHA256)
    if observed != expected:
        raise RuntimeError(
            "FlashInfer target MoE source drift: "
            f"expected {expected}, got {observed}"
        )
    return dispatch_path, kernel_path


def export(output_dir: Path, arch: str) -> dict[str, str]:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_kernel import (
        MoEStaticKernel,
    )

    authenticate_sources()
    p = PLAN
    kernel = MoEStaticKernel(
        sf_vec_size=16,
        mma_tiler_mn=(p.tile_m, p.tile_n),
        output_tile_count_n=p.physical_intermediate // p.tile_n,
        fast_math=p.fast_math,
        activation=p.activation,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=None,
        input_scales_are_reciprocal=False,
    )
    # The wheel cache uses TVMFFIEnvGetStream.  Export the identical static
    # kernel with a typed stream retained in the native signature instead.
    kernel.__call__.__func__.__annotations__["stream"] = cuda.CUstream
    tensor = cute.runtime.make_fake_compact_tensor
    pointer = lambda: make_ptr(
        cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    a_input = tensor(
        cutlass.BFloat16, (p.tokens, p.hidden), stride_order=(1, 0), assumed_align=16
    )
    topk_ids = tensor(cutlass.Int32, (p.top_k,), assumed_align=4)
    topk_weights = tensor(cutlass.Float32, (p.top_k,), assumed_align=4)
    packed_a = tensor(
        cutlass.Float4E2M1FN,
        (p.max_rows, p.hidden, p.state_experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    packed_a_storage = tensor(
        cutlass.Uint8,
        (p.state_experts * p.max_rows * (p.hidden // 2),),
        assumed_align=16,
    )
    retained_groups = p.physical_intermediate // 256
    route_scratch = tensor(
        cutlass.BFloat16,
        (p.max_rows * retained_groups * p.hidden,),
        assumed_align=16,
    )
    rows_pad = 128
    cols_pad = ((p.hidden // 16 + 3) // 4) * 4
    scale_storage = tensor(
        cutlass.Uint8,
        (p.state_experts * rows_pad * cols_pad,),
        assumed_align=16,
    )
    scalar_i32 = lambda: tensor(cutlass.Int32, (1,), assumed_align=4)
    w13 = tensor(
        cutlass.Float4E2M1FN,
        (2 * p.physical_intermediate, p.hidden, p.local_experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    down = tensor(
        cutlass.Float4E2M1FN,
        (p.hidden, p.physical_intermediate, p.local_experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    row_counts = tensor(cutlass.Int32, (p.state_experts,), assumed_align=4)
    weight_expert_ids = tensor(cutlass.Int32, (p.state_experts,), assumed_align=4)
    global_to_local = tensor(cutlass.Int32, (p.local_experts,), assumed_align=4)
    virt_scratch = tensor(
        cutlass.Int32, (p.local_experts * 2 + 8,), assumed_align=4
    )
    expert_f32 = lambda: tensor(
        cutlass.Float32, (p.local_experts,), assumed_align=16
    )
    output_tensor = tensor(
        cutlass.BFloat16,
        (p.tokens, p.hidden),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map = tensor(
        cutlass.Int32,
        (p.state_experts, p.max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights = tensor(
        cutlass.Float32,
        (p.state_experts, p.max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    compiled = cute.compile(
        kernel,
        a_input,
        topk_ids,
        topk_weights,
        packed_a,
        pointer(),
        packed_a_storage,
        route_scratch,
        scale_storage,
        scalar_i32(),
        scalar_i32(),
        w13,
        pointer(),
        down,
        pointer(),
        row_counts,
        scalar_i32(),
        weight_expert_ids,
        global_to_local,
        virt_scratch,
        expert_f32(),
        expert_f32(),
        expert_f32(),
        expert_f32(),
        output_tensor,
        token_map,
        token_weights,
        p.max_active_clusters,
        stream,
        options=f"--opt-level 2 --gpu-arch {arch} --host-target linux-aarch64",
    )
    compiled.export_to_c(
        output_dir.as_posix(), "target_moe_b12x_c1", "qwen38_target_moe_b12x_c1"
    )
    authenticate_exported_abi(
        output_dir / "target_moe_b12x_c1.h",
        output_dir / "target_moe_b12x_c1.o",
    )
    return {
        "header_sha256": digest(output_dir / "target_moe_b12x_c1.h"),
        "object_sha256": digest(output_dir / "target_moe_b12x_c1.o"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arch", default="sm_121a")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    manifest = {
        "abi": "rocket.qwen38.target-moe.b12x-aot.c1.v1",
        "arch": args.arch,
        "commit": PINNED_FLASHINFER_COMMIT,
        "dispatch_sha256": PINNED_DISPATCH_SHA256,
        "kernel_sha256": PINNED_KERNEL_SHA256,
        "measured_tvm_ffi_object_sha256": PINNED_TVM_FFI_OBJECT_SHA256,
        "plan": asdict(PLAN),
        "stream": "explicit_borrowed_cu_stream",
    }
    print(json.dumps(manifest, sort_keys=True))
    if args.plan_only:
        return
    if args.output is None or not args.output.is_dir():
        parser.error("--output must name an existing build directory")
    manifest["export"] = export(args.output, args.arch)
    (args.output / "target_moe_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
