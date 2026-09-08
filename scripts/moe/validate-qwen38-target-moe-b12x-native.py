#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validation-only graph capture and eager parity for native target MoE c1."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

ARTIFACT_SHA256 = "a9fcca026a87ad1285b94feef19448c51b42d97516f16211c61ae4c770c6f0f4"
CREATE_FAILURE_FIELDS = (
    "none", "device", "artifact_sha256", "layout_sha256", "rank", "layer",
    "w13_packed", "w13_scale", "down_packed", "down_scale",
    "input_global_scale", "folded_w1_alpha", "w2_alpha", "down_input_scale",
)


class Identity(ctypes.Structure):
    _fields_ = [
        ("artifact_sha256", ctypes.c_uint8 * 32),
        ("layout_sha256", ctypes.c_uint8 * 32),
        ("rank", ctypes.c_int),
        ("layer", ctypes.c_int),
    ]


class Weights(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "w13_packed", "w13_scale", "down_packed", "down_scale",
        "input_global_scale", "folded_w1_alpha", "w2_alpha", "down_input_scale",
    )]


class Workspace(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "packed_a", "packed_a_scale", "route_output_scratch_bf16",
        "barrier_count", "barrier_epoch", "row_counts",
        "active_expert_count", "weight_expert_ids",
        "global_to_local_expert", "virtual_route_scratch", "token_map",
        "token_weights",
    )]


class Launch(ctypes.Structure):
    _fields_ = [
        ("hidden_bf16", ctypes.c_void_p),
        ("local_expert_ids", ctypes.c_void_p),
        ("local_routing_weights", ctypes.c_void_p),
        ("output_bf16", ctypes.c_void_p),
        ("workspace", Workspace),
        ("stream", ctypes.c_void_p),
    ]


def pointer(tensor) -> int:
    return int(tensor.data_ptr())


def bind_library(path: Path):
    library = ctypes.CDLL(path)
    library.rocket_qwen38_target_moe_b12x_available.restype = ctypes.c_bool
    library.rocket_qwen38_target_moe_b12x_create.argtypes = [
        ctypes.c_int, ctypes.POINTER(Identity), ctypes.POINTER(Weights),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.rocket_qwen38_target_moe_b12x_create.restype = ctypes.c_int
    library.rocket_qwen38_target_moe_b12x_diagnose_create.argtypes = [
        ctypes.c_int, ctypes.POINTER(Identity), ctypes.POINTER(Weights),
    ]
    library.rocket_qwen38_target_moe_b12x_diagnose_create.restype = ctypes.c_int
    library.rocket_qwen38_target_moe_b12x_enqueue.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(Launch),
    ]
    library.rocket_qwen38_target_moe_b12x_enqueue.restype = ctypes.c_int
    library.rocket_qwen38_target_moe_b12x_destroy.argtypes = [ctypes.c_void_p]
    return library


def create_failure_record(*, status: int, diagnostic: int, rank: int, layer: int):
    """Return a fixed-cardinality diagnostic without pointer or hash values."""

    field = (
        CREATE_FAILURE_FIELDS[diagnostic]
        if 0 <= diagnostic < len(CREATE_FAILURE_FIELDS)
        else "unknown"
    )
    return {
        "abi": "rocket.qwen38.target-moe.native-failure.v1",
        "accepted": False,
        "failure_class": "contract" if status == 1 else "cuda",
        "first_invalid_field": field,
        "layer": layer,
        "phase": "create",
        "rank": rank,
        "status": status if status in (1, 3) else -1,
    }


def validate_artifact_mount_identity(
    artifact: Path, expected_artifact_key: str = ARTIFACT_SHA256,
) -> str:
    """Fail before CUDA when a bind mount erases the content-addressed name."""

    try:
        manifest = json.loads((artifact / "manifest.json").read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("target slab manifest is unavailable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("target slab manifest root changed")
    canonical = dict(manifest)
    claimed = canonical.pop("artifact_key", None)
    observed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != expected_artifact_key or observed != claimed:
        raise RuntimeError("target slab manifest artifact identity changed")
    if artifact.name != claimed:
        raise RuntimeError(
            "target slab mount basename must equal manifest artifact_key"
        )
    return claimed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--layer", type=int, choices=range(48), required=True)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    args = parser.parse_args()

    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from qwen38_slab.routed_moe import (
        FlashInferRoutedMoeBackend,
        MoeShape,
        load_owner_local_moe,
        materialize_flashinfer_weights,
    )

    if args.atol < 0.0 or args.rtol < 0.0:
        parser.error("tolerances must be nonnegative")
    slab = load_owner_local_moe(args.artifact, args.rank, args.layer)
    if slab.artifact_key != ARTIFACT_SHA256:
        raise RuntimeError("target slab artifact identity changed")
    source_weights = materialize_flashinfer_weights(slab)
    padded = moe_dispatch._pad_intermediate_to_tile(
        source_weights.w1_weight,
        source_weights.w1_scale,
        source_weights.w2_weight,
        source_weights.w2_scale,
        source_weights.fc2_input_scale,
        640,
        256,
        2560,
        256,
        True,
        "nvfp4",
    )
    w1, w1_sf, w2, w2_sf, down_scale, physical_n = padded
    if physical_n != 768:
        raise RuntimeError("target MoE padding geometry changed")
    views = moe_dispatch._get_weight_views(
        w1, w1_sf, w2, w2_sf, source_weights.w1_alpha,
        source_weights.w2_alpha, physical_n, 2560,
        activation_precision="fp4", quant_mode="nvfp4",
    )
    folded_w1_alpha = (
        views.w1_alpha * source_weights.input_scale
    ).contiguous()
    workspace = moe_dispatch.allocate_sm120_static_workspace(
        state_E=256, weight_E=256, max_rows=10, k=2560, n=physical_n,
        num_topk=10, device=source_weights.w1_weight.device,
        activation_precision="fp4", quant_mode="nvfp4",
    )

    torch.manual_seed(7)
    hidden = torch.randn(1, 2560, dtype=torch.bfloat16, device="cuda")
    global_ids = torch.tensor(
        [[0, 256, 1, 257, 2, 258, 3, 259, 4, 260]],
        dtype=torch.int32, device="cuda",
    )
    routing_weights = torch.tensor(
        [[0.19, 0.18, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05, 0.04, 0.02]],
        dtype=torch.float32, device="cuda",
    )
    first = args.rank * 256
    owned = (global_ids >= first) & (global_ids < first + 256)
    local_ids = torch.where(owned, global_ids - first, torch.zeros_like(global_ids))
    local_weights = torch.where(
        owned, routing_weights, torch.zeros_like(routing_weights)
    )
    output = torch.empty_like(hidden)
    stream = torch.cuda.Stream()

    native_weights = Weights(
        pointer(views.w13_fp4), pointer(views._w13_sf_storage),
        pointer(views.down_fp4), pointer(views._down_sf_storage),
        pointer(source_weights.input_scale), pointer(folded_w1_alpha),
        pointer(views.w2_alpha), pointer(down_scale),
    )
    native_workspace = Workspace(
        pointer(workspace.packed_a_flat), pointer(workspace.scale_flat),
        pointer(workspace.route_output_scratch), pointer(workspace.barrier_count),
        pointer(workspace.barrier_epoch), pointer(workspace.row_counts),
        pointer(workspace.active_expert_count), pointer(workspace.weight_expert_ids),
        pointer(workspace.global_to_local_expert),
        pointer(workspace.virt_route_scratch), pointer(workspace.token_map),
        pointer(workspace.token_weights),
    )
    launch = Launch(
        pointer(hidden), pointer(local_ids), pointer(local_weights), pointer(output),
        native_workspace, int(stream.cuda_stream),
    )
    identity = Identity(
        (ctypes.c_uint8 * 32).from_buffer_copy(bytes.fromhex(slab.artifact_key)),
        (ctypes.c_uint8 * 32).from_buffer_copy(bytes.fromhex(slab.layout_sha256)),
        args.rank,
        args.layer,
    )
    library = bind_library(args.native_library)
    if not library.rocket_qwen38_target_moe_b12x_available():
        raise RuntimeError("native target MoE AOT backend is unavailable")
    handle = ctypes.c_void_p()
    device = torch.cuda.current_device()
    diagnostic = library.rocket_qwen38_target_moe_b12x_diagnose_create(
        device, ctypes.byref(identity), ctypes.byref(native_weights)
    )
    status = library.rocket_qwen38_target_moe_b12x_create(
        device, ctypes.byref(identity),
        ctypes.byref(native_weights), ctypes.byref(handle),
    )
    if status != 0 or not handle.value:
        print(json.dumps(create_failure_record(
            status=status, diagnostic=diagnostic, rank=args.rank,
            layer=args.layer,
        ), sort_keys=True))
        raise RuntimeError(f"native target MoE create failed: outcome={status}")
    try:
        oracle_backend = FlashInferRoutedMoeBackend(source_weights)
        oracle = oracle_backend.routed_only(
            hidden, global_ids, routing_weights, rank=args.rank,
            shape=MoeShape(1), selector="static",
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            status = library.rocket_qwen38_target_moe_b12x_enqueue(
                handle, ctypes.byref(launch)
            )
            if status != 0:
                raise RuntimeError(
                    f"native target MoE capture failed: outcome={status}"
                )
        graph.replay()
        stream.synchronize()
        absolute = (output.float() - oracle.float()).abs()
        relative = absolute / oracle.float().abs().clamp_min(1.0e-12)
        accepted = bool(torch.allclose(
            output.float(), oracle.float(), atol=args.atol, rtol=args.rtol
        ))
        result = {
            "abi": "rocket.qwen38.target-moe.native-parity.c1.v1",
            "accepted": accepted,
            "artifact_sha256": slab.artifact_key,
            "graph_capture": True,
            "layer": args.layer,
            "layout_sha256": slab.layout_sha256,
            "logical_intermediate": 640,
            "max_abs": float(absolute.max().item()),
            "max_rel": float(relative.max().item()),
            "physical_intermediate": physical_n,
            "rank": args.rank,
            "route_slots": 10,
        }
        print(json.dumps(result, sort_keys=True))
        if not accepted:
            raise RuntimeError("native target MoE differs from eager B12x oracle")
    finally:
        library.rocket_qwen38_target_moe_b12x_destroy(handle)


if __name__ == "__main__":
    main()
