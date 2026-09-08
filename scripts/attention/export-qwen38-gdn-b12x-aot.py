#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export pinned FlashInfer SM120 B12X GDN projection kernels to C objects.

Run this in the pinned Torch/CUDA image. The output headers and relocatable
objects contain native host launchers plus embedded cubins. Python and Torch
are build-time dependencies only and are absent from the serving hot path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PINNED_FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"
PINNED_KERNEL_SHA256 = (
    "6739702e27afad21767b71024678b35c6892e3f9c74aaff375fb3f6734399f86"
)


@dataclass(frozen=True)
class Shape:
    name: str
    tokens: int
    output_width: int
    input_width: int = 2560


@dataclass(frozen=True)
class Plan:
    name: str
    tokens: int
    output_width: int
    input_width: int
    tile_m: int
    tile_n: int
    tile_k: int
    swap_ab: bool
    use_prefetch: bool


SHAPES = (
    Shape("t300_qkvz", 300, 8192),
    Shape("t300_ba", 300, 48),
    Shape("t8192_qkvz", 8192, 8192),
    Shape("t8192_ba", 8192, 48),
)


def select_plan(shape: Shape, sm_count: int) -> Plan:
    """Mirror pinned _select_default_dense_gemm_plan for these four shapes."""
    coarse_tiles = ((shape.tokens + 127) // 128) * (
        (shape.output_width + 127) // 128
    )
    if shape.output_width > 1536 and (
        shape.tokens <= 64
        or (shape.tokens <= 256 and coarse_tiles < max(1, sm_count // 2))
    ):
        tile_m, tile_n = 64, 128
    elif shape.tokens <= 128 and coarse_tiles < max(1, sm_count // 2):
        medium_tiles = ((shape.tokens + 127) // 128) * (
            (shape.output_width + 63) // 64
        )
        tile_m, tile_n = (
            (64, 64)
            if medium_tiles < max(1, sm_count // 2)
            else (128, 64)
        )
    else:
        tile_m, tile_n = 128, 128
    return Plan(
        shape.name,
        shape.tokens,
        shape.output_width,
        shape.input_width,
        tile_m,
        tile_n,
        128,
        tile_n < 64,
        False,
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def export(plan: Plan, output: Path, arch: str, max_active_clusters: int) -> None:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x

    kernel_path = Path(dense_blockscaled_gemm_sm120_b12x.__file__).resolve()
    actual_digest = digest(kernel_path)
    if actual_digest != PINNED_KERNEL_SHA256:
        raise RuntimeError(
            "FlashInfer B12X source drift: "
            f"expected {PINNED_KERNEL_SHA256}, got {actual_digest} ({kernel_path})"
        )

    kernel = dense_blockscaled_gemm_sm120_b12x.DenseGemmKernel(
        16,
        (plan.tile_m, plan.tile_n),
        (1, 1),
        tile_k=plan.tile_k,
        use_prefetch=plan.use_prefetch,
        enable_pdl=True,
        swap_ab=plan.swap_ab,
    )
    # The pinned FlashInfer wrapper intentionally leaves the stream untyped
    # because its normal TVM-FFI path supplies the current stream through the
    # call environment.  CuTe's C exporter needs an explicit CUstream type to
    # retain the caller-owned stream in the generated native ABI.
    kernel.wrapper.__func__.__annotations__["current_stream"] = cuda.CUstream
    sym_m = cute.sym_int()
    sym_k = cute.sym_int()
    sym_n = cute.sym_int()
    a = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, sym_k), stride_order=(1, 0), assumed_align=32
    )
    b = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_n, sym_k), stride_order=(1, 0), assumed_align=32
    )
    c = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, sym_n),
        stride_order=(1, 0),
        assumed_align=16,
    )
    scale_pointer = lambda: make_ptr(
        cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, 16
    )
    alpha = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False)
    sf_m = (plan.tokens + 127) // 128
    sf_n = (plan.output_width + 127) // 128
    sf_k = (plan.input_width // 16 + 3) // 4
    compiled = cute.compile(
        kernel.wrapper,
        a,
        b,
        c,
        sf_m,
        sf_n,
        sf_k,
        1,
        scale_pointer(),
        scale_pointer(),
        alpha,
        max_active_clusters,
        stream,
        False,
        options=f"--opt-level 2 --gpu-arch {arch} --host-target linux-aarch64",
    )
    compiled.export_to_c(output.as_posix(), plan.name, f"qwen38_gdn_{plan.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--max-active-clusters", type=int)
    parser.add_argument("--arch", default="sm_121a")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.sm_count <= 0:
        parser.error("--sm-count must be positive")
    plans = tuple(select_plan(shape, args.sm_count) for shape in SHAPES)
    print(json.dumps({"commit": PINNED_FLASHINFER_COMMIT,
                      "kernel_sha256": PINNED_KERNEL_SHA256,
                      "plans": [asdict(plan) for plan in plans]}, sort_keys=True))
    if args.plan_only:
        return
    if args.output is None or not args.output.is_dir():
        parser.error("--output must name an existing build directory")
    if args.max_active_clusters is None or args.max_active_clusters <= 0:
        parser.error("--max-active-clusters must be positive when exporting")
    for plan in plans:
        export(plan, args.output, args.arch, args.max_active_clusters)
    manifest = {
        "abi": "rocket.qwen38.gdn.b12x-aot.v1",
        "arch": args.arch,
        "commit": PINNED_FLASHINFER_COMMIT,
        "kernel_sha256": PINNED_KERNEL_SHA256,
        "max_active_clusters": args.max_active_clusters,
        "plans": [asdict(plan) for plan in plans],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
