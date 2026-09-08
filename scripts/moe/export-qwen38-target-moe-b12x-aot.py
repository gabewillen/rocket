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
    "c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a"
)
PINNED_KERNEL_SHA256 = (
    "c7b6f24b94d7939cc0eb917ab15cef4c34f3dc12bf75e15fc9dc316ee7327f3f"
)
ARTIFACT_KEY_HEADER = "target_moe_artifact_key.h"
COMPACT_CONFIG_HEADER = "target_moe_compact_config.h"
COMPACT_CONFIG_SCHEMA = "rocket.qwen38.target-moe.compact-aot.v1"
SOURCE_ABI = "modelopt_nvfp4_group16_cutlass_sm121_sfb"
TRANSFORM_ABI = "rocket.qwen38.target-moe.device-stage.v1"
ROUTE_REMAP_ABI = "route_position_iota10_unique_positive_remote_zero_v1"
PINNED_COMPACT_CONFIG_SHA256 = (
    "2ec6180161706b6c4b6c3d6656d86279736865567e2456726451cc6c910dfe16"
)


@dataclass(frozen=True)
class Plan:
    tokens: int = 1
    hidden: int = 2560
    logical_intermediate: int = 640
    physical_intermediate: int = 768
    top_k: int = 10
    max_rows: int = 10
    weight_experts: int = 10
    state_experts: int = 11
    tile_m: int = 64
    tile_n: int = 128
    max_active_clusters: int = 20
    activation: str = "silu"
    fast_math: bool = True
    source_format: str = "modelopt"


PLAN = Plan()

EXPECTED_SHAPE = {
    "tokens": 1,
    "hidden": 2560,
    "logical_intermediate": 640,
    "physical_intermediate": 768,
    "top_k": 10,
    "weight_experts": 10,
    "state_experts": 11,
    "max_rows": 10,
}
EXPECTED_PLANES = [
    {"name": "w13_packed", "dtype": "uint8", "shape": [10, 1536, 1280], "bytes": 19_660_800},
    {"name": "w13_scale", "dtype": "e4m3_sfb", "shape": [10, 1536, 160], "bytes": 2_457_600},
    {"name": "down_packed", "dtype": "uint8", "shape": [10, 2560, 384], "bytes": 9_830_400},
    {"name": "down_scale", "dtype": "e4m3_sfb", "shape": [10, 2560, 48], "bytes": 1_228_800},
    {"name": "input_global_scale", "dtype": "float32", "shape": [10], "bytes": 40},
    {"name": "folded_w1_alpha", "dtype": "float32", "shape": [10], "bytes": 40},
    {"name": "w2_alpha", "dtype": "float32", "shape": [10], "bytes": 40},
    {"name": "down_input_scale", "dtype": "float32", "shape": [10], "bytes": 40},
]
EXPECTED_CONTROL = [
    {"name": "compact_expert_ids", "dtype": "int32", "shape": [10], "bytes": 40},
    {"name": "compact_routing_weights", "dtype": "float32", "shape": [10], "bytes": 40},
    {"name": "source_expert_ids", "dtype": "int32", "shape": [10], "bytes": 40},
    {"name": "active_experts", "dtype": "int32", "shape": [1], "bytes": 4},
    {"name": "generation", "dtype": "uint64", "shape": [1], "bytes": 8},
    {"name": "outcome", "dtype": "int32", "shape": [1], "bytes": 4},
]
EXPECTED_FC1_ROWS = [
    "up[0:640]",
    "zero[640:768]",
    "gate[768:1408]",
    "zero[1408:1536]",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def authenticate_compact_config(path: Path, artifact_key: str) -> dict[str, object]:
    """Load the closed E10 stage/consumer contract or reject before codegen."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("compact target MoE config is unavailable") from exc
    if not isinstance(config, dict) or set(config) != {
        "schema", "target_artifact_key", "source_abi", "transform_abi",
        "route_remap_abi", "shape", "planes", "control", "fc1_physical_rows",
        "ranks",
    }:
        raise RuntimeError("compact target MoE config schema changed")
    if (
        config["schema"] != COMPACT_CONFIG_SCHEMA
        or config["target_artifact_key"] != artifact_key
        or config["source_abi"] != SOURCE_ABI
        or config["transform_abi"] != TRANSFORM_ABI
        or config["route_remap_abi"] != ROUTE_REMAP_ABI
        or config["shape"] != EXPECTED_SHAPE
        or config["planes"] != EXPECTED_PLANES
        or config["control"] != EXPECTED_CONTROL
        or config["fc1_physical_rows"] != EXPECTED_FC1_ROWS
    ):
        raise RuntimeError("compact target MoE shape or layout changed")
    ranks = config["ranks"]
    if not isinstance(ranks, list) or len(ranks) != 2:
        raise RuntimeError("compact target MoE rank identities changed")
    for expected_rank, rank in enumerate(ranks):
        if (
            not isinstance(rank, dict)
            or set(rank) != {
                "rank", "descriptor_sha256", "binding_inventory_sha256",
                "publication_layout_sha256",
            }
            or rank["rank"] != expected_rank
            or not all(
                _valid_sha256(rank[field])
                for field in (
                    "descriptor_sha256", "binding_inventory_sha256",
                    "publication_layout_sha256",
                )
            )
        ):
            raise RuntimeError("compact target MoE rank identities changed")
    if canonical_digest(config) != PINNED_COMPACT_CONFIG_SHA256:
        raise RuntimeError("compact target MoE config identity changed")
    return config


def compact_layout_identity(config: dict[str, object]) -> str:
    return canonical_digest({
        "shape": config["shape"],
        "planes": config["planes"],
        "control": config["control"],
        "fc1_physical_rows": config["fc1_physical_rows"],
        "route_remap_abi": config["route_remap_abi"],
    })


def emit_compact_config_header(
    output_dir: Path, config: dict[str, object]
) -> tuple[Path, str]:
    config_sha256 = canonical_digest(config)
    layout_sha256 = compact_layout_identity(config)
    ranks = config["ranks"]
    assert isinstance(ranks, list)
    values = {
        "Config": config_sha256,
        "Layout": layout_sha256,
        "SourceAbi": canonical_digest(SOURCE_ABI),
        "TransformAbi": canonical_digest(TRANSFORM_ABI),
        "RouteRemapAbi": canonical_digest(ROUTE_REMAP_ABI),
        "Rank0Descriptor": ranks[0]["descriptor_sha256"],
        "Rank1Descriptor": ranks[1]["descriptor_sha256"],
        "Rank0BindingInventory": ranks[0]["binding_inventory_sha256"],
        "Rank1BindingInventory": ranks[1]["binding_inventory_sha256"],
        "Rank0PublicationLayout": ranks[0]["publication_layout_sha256"],
        "Rank1PublicationLayout": ranks[1]["publication_layout_sha256"],
    }
    declarations = "".join(
        f'inline constexpr char kRocketQwen38TargetMoeCompact{name}Sha256[] = "{value}";\n'
        for name, value in values.items()
    )
    path = output_dir / COMPACT_CONFIG_HEADER
    path.write_text(
        "// Generated from the authenticated compact E10 stage contract.\n"
        "#pragma once\n"
        "inline constexpr int kRocketQwen38TargetMoeWeightExperts = 10;\n"
        "inline constexpr int kRocketQwen38TargetMoeStateExperts = 11;\n"
        "inline constexpr int kRocketQwen38TargetMoeMaxRows = 10;\n"
        "inline constexpr int kRocketQwen38TargetMoePhysicalIntermediate = 768;\n"
        + declarations,
        encoding="utf-8",
    )
    return path, config_sha256


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


def authenticate_artifact_manifest(path: Path) -> str:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("target MoE artifact manifest is unavailable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("target MoE artifact manifest root changed")
    canonical = dict(manifest)
    claimed = canonical.pop("artifact_key", None)
    observed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        not isinstance(claimed, str)
        or len(claimed) != 64
        or any(ch not in "0123456789abcdef" for ch in claimed)
        or observed != claimed
    ):
        raise RuntimeError("target MoE artifact manifest identity changed")
    return claimed


def emit_artifact_key_header(output_dir: Path, artifact_key: str) -> Path:
    path = output_dir / ARTIFACT_KEY_HEADER
    path.write_text(
        "// Generated from the authenticated target slab manifest.\n"
        "#pragma once\n"
        f'inline constexpr char kRocketQwen38TargetMoeArtifactKey[] = "{artifact_key}";\n',
        encoding="utf-8",
    )
    return path


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
        (2 * p.physical_intermediate, p.hidden, p.weight_experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    down = tensor(
        cutlass.Float4E2M1FN,
        (p.hidden, p.physical_intermediate, p.weight_experts),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    row_counts = tensor(cutlass.Int32, (p.state_experts,), assumed_align=4)
    weight_expert_ids = tensor(cutlass.Int32, (p.state_experts,), assumed_align=4)
    global_to_local = tensor(cutlass.Int32, (p.weight_experts,), assumed_align=4)
    virt_scratch = tensor(
        cutlass.Int32, (p.weight_experts * 2 + 8,), assumed_align=4
    )
    expert_f32 = lambda: tensor(
        cutlass.Float32, (p.weight_experts,), assumed_align=16
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
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--compact-config", type=Path)
    parser.add_argument("--arch", default="sm_121a")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    args = parser.parse_args()
    manifest = {
        "abi": "rocket.qwen38.target-moe.b12x-aot.c1.v1",
        "arch": args.arch,
        "commit": PINNED_FLASHINFER_COMMIT,
        "dispatch_sha256": PINNED_DISPATCH_SHA256,
        "kernel_sha256": PINNED_KERNEL_SHA256,
        "plan": asdict(PLAN),
        "stream": "explicit_borrowed_cu_stream",
    }
    print(json.dumps(manifest, sort_keys=True))
    if args.plan_only:
        return
    if args.output is None or not args.output.is_dir():
        parser.error("--output must name an existing build directory")
    if args.artifact_manifest is None:
        parser.error("--artifact-manifest is required for an authenticated export")
    artifact_key = authenticate_artifact_manifest(args.artifact_manifest)
    if args.compact_config is None:
        parser.error("--compact-config is required for an authenticated E10 export")
    compact_config = authenticate_compact_config(args.compact_config, artifact_key)
    identity_header = emit_artifact_key_header(args.output, artifact_key)
    config_header, compact_config_sha256 = emit_compact_config_header(
        args.output, compact_config
    )
    manifest["artifact_key"] = artifact_key
    manifest["artifact_key_header_sha256"] = digest(identity_header)
    manifest["compact_config_sha256"] = compact_config_sha256
    manifest["compact_layout_sha256"] = compact_layout_identity(compact_config)
    manifest["compact_config_header_sha256"] = digest(config_header)
    if args.identity_only:
        print(json.dumps(manifest, sort_keys=True))
        return
    manifest["export"] = export(args.output, args.arch)
    (args.output / "target_moe_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
