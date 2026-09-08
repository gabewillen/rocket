#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Physical graph-capture parity for one real target router+MoE layer3."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUTED_HARNESS = ROOT / "scripts/moe/validate-qwen38-target-moe-b12x-native.py"
SPEC = importlib.util.spec_from_file_location("routed_native_harness", ROUTED_HARNESS)
assert SPEC and SPEC.loader
ROUTED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTED)


class RouterWeights(ctypes.Structure):
    _fields_ = [
        ("packed_e2m1", ctypes.c_void_p),
        ("cutlass_sfb_e4m3", ctypes.c_void_p),
        ("weight_scale_2", ctypes.c_void_p),
    ]


class SharedWeights(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in ("gate", "up", "down", "shared_gate")]


class FullWeights(ctypes.Structure):
    _fields_ = [
        ("router", RouterWeights),
        ("routed", ROUTED.Weights),
        ("shared", SharedWeights),
    ]


class FullWorkspace(ctypes.Structure):
    _fields_ = [
        ("router_logits_f32", ctypes.c_void_p),
        ("global_ids_i32", ctypes.c_void_p),
        ("routing_weights_f32", ctypes.c_void_p),
        ("local_ids_i32", ctypes.c_void_p),
        ("local_weights_f32", ctypes.c_void_p),
        ("source_generation", ctypes.c_void_p),
        ("requested_generation", ctypes.c_void_p),
        ("route_summary", ctypes.c_void_p),
        ("routed", ROUTED.Workspace),
        ("shared_gate_scratch_f32", ctypes.c_void_p),
        ("shared_up_scratch_f32", ctypes.c_void_p),
        ("shared_gate_scalar_f32", ctypes.c_void_p),
    ]


class FullLaunch(ctypes.Structure):
    _fields_ = [
        ("hidden_bf16", ctypes.c_void_p),
        ("rank_local_partial_bf16", ctypes.c_void_p),
        ("workspace", FullWorkspace),
        ("stream", ctypes.c_void_p),
    ]


def bind(path: Path):
    library = ctypes.CDLL(path)
    library.rocket_qwen38_target_full_moe_c1_create.argtypes = [
        ctypes.c_int, ctypes.POINTER(ROUTED.Identity), ctypes.POINTER(FullWeights),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.rocket_qwen38_target_full_moe_c1_create.restype = ctypes.c_int
    library.rocket_qwen38_target_full_moe_c1_enqueue.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(FullLaunch),
    ]
    library.rocket_qwen38_target_full_moe_c1_enqueue.restype = ctypes.c_int
    library.rocket_qwen38_target_full_moe_c1_destroy.argtypes = [ctypes.c_void_p]
    return library


def emit_failure(*, phase: str, status: int, rank: int, layer: int) -> None:
    phases = {"create", "capture"}
    print(json.dumps({
        "abi": "rocket.qwen38.target-full-moe.c1.failure.v1",
        "accepted": False,
        "failure_class": "contract" if status == 1 else "cuda",
        "layer": layer,
        "phase": phase if phase in phases else "unknown",
        "rank": rank,
        "status": status if status in (1, 2) else -1,
    }, sort_keys=True))


def read_router_tensors(slab, torch):
    names = {item.name.rsplit(".gate.", 1)[-1]: item for item in slab.router}
    with slab.slab_path.open("rb", buffering=0) as source:
        def read(name):
            extent = names[name]
            source.seek(extent.offset)
            payload = source.read(extent.length)
            if len(payload) != extent.length:
                raise RuntimeError("short authenticated target router extent")
            return bytearray(payload)
        packed_bytes = read("weight")
        scale_bytes = read("weight_scale")
        alpha_bytes = read("weight_scale_2")
    packed = torch.frombuffer(packed_bytes, dtype=torch.uint8).clone().cuda()
    scales = torch.frombuffer(scale_bytes, dtype=torch.uint8).clone().cuda()
    alpha = torch.tensor(
        [struct.unpack("<f", alpha_bytes)[0]], dtype=torch.float32, device="cuda"
    )
    return packed, scales, alpha, packed_bytes, scale_bytes


def reconstruct_router(packed: bytearray, scales: bytearray, alpha: float, torch):
    from qwen38_slab.projection import _e2m1, _e4m3, _sfb_offset

    codes = torch.frombuffer(packed, dtype=torch.uint8).reshape(512, 1280)
    unpacked = torch.empty((512, 2560), dtype=torch.uint8)
    unpacked[:, 0::2] = codes & 15
    unpacked[:, 1::2] = codes >> 4
    e2 = torch.tensor([_e2m1(code) for code in range(16)], dtype=torch.float32)
    scale_indices = torch.tensor(
        [[_sfb_offset(row, group, 2560) for group in range(160)] for row in range(512)],
        dtype=torch.int64,
    )
    scale_codes = torch.frombuffer(scales, dtype=torch.uint8)
    e4 = torch.tensor([_e4m3(code) for code in range(256)], dtype=torch.float32)
    return e2[unpacked.long()] * e4[scale_codes[scale_indices].long()].repeat_interleave(16, 1) * alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--layer", type=int, choices=(3,), default=3)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.08)
    args = parser.parse_args()

    artifact_key = ROUTED.validate_artifact_mount_identity(args.artifact)
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from qwen38_slab.routed_moe import (
        FlashInferRoutedMoeBackend, MoeShape, load_owner_local_moe,
        materialize_flashinfer_weights,
    )

    slab = load_owner_local_moe(args.artifact, args.rank, args.layer)
    if slab.artifact_key != artifact_key:
        raise RuntimeError("target slab artifact identity changed")
    source_weights = materialize_flashinfer_weights(slab)
    padded = moe_dispatch._pad_intermediate_to_tile(
        source_weights.w1_weight, source_weights.w1_scale,
        source_weights.w2_weight, source_weights.w2_scale,
        source_weights.fc2_input_scale, 640, 256, 2560, 256, True, "nvfp4",
    )
    w1, w1_sf, w2, w2_sf, down_scale, physical_n = padded
    if physical_n != 768:
        raise RuntimeError("target MoE padding geometry changed")
    views = moe_dispatch._get_weight_views(
        w1, w1_sf, w2, w2_sf, source_weights.w1_alpha,
        source_weights.w2_alpha, physical_n, 2560,
        activation_precision="fp4", quant_mode="nvfp4",
    )
    routed_workspace = moe_dispatch.allocate_sm120_static_workspace(
        state_E=256, weight_E=256, max_rows=10, k=2560, n=physical_n,
        num_topk=10, device=source_weights.w1_weight.device,
        activation_precision="fp4", quant_mode="nvfp4",
    )
    packed, router_scales, router_alpha, packed_host, scales_host = read_router_tensors(slab, torch)

    torch.manual_seed(7)
    hidden = torch.randn(1, 2560, dtype=torch.bfloat16, device="cuda")
    reconstructed = reconstruct_router(
        packed_host, scales_host, float(router_alpha.cpu().item()), torch
    ).cuda()
    oracle_logits = hidden.float() @ reconstructed.T
    oracle_ids = torch.argsort(oracle_logits, dim=1, descending=True, stable=True)[:, :10].to(torch.int32)
    oracle_selected = oracle_logits.gather(1, oracle_ids.long())
    oracle_weights = torch.softmax(oracle_selected, dim=1)
    backend = FlashInferRoutedMoeBackend(source_weights)
    oracle = backend.routed_only(
        hidden, oracle_ids, oracle_weights, rank=args.rank,
        shape=MoeShape(1), selector="static",
    ).add_(backend.shared_partial(hidden, args.rank))

    logits = torch.empty((1, 512), dtype=torch.float32, device="cuda")
    global_ids = torch.empty((1, 10), dtype=torch.int32, device="cuda")
    routing_weights = torch.empty((1, 10), dtype=torch.float32, device="cuda")
    local_ids = torch.empty_like(global_ids)
    local_weights = torch.empty_like(routing_weights)
    output = torch.empty_like(hidden)
    source_generation = torch.empty((1,), dtype=torch.uint64, device="cuda")
    requested_generation = torch.tensor([7], dtype=torch.uint64, device="cuda")
    route_summary = torch.empty((16,), dtype=torch.uint8, device="cuda")
    shared_gate_scratch = torch.empty((160,), dtype=torch.float32, device="cuda")
    shared_up_scratch = torch.empty((160,), dtype=torch.float32, device="cuda")
    shared_gate_scalar = torch.empty((1,), dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()

    routed_weights = ROUTED.Weights(
        ROUTED.pointer(views.w13_fp4), ROUTED.pointer(views._w13_sf_storage),
        ROUTED.pointer(views.down_fp4), ROUTED.pointer(views._down_sf_storage),
        ROUTED.pointer(source_weights.input_scale), ROUTED.pointer(views.w1_alpha),
        ROUTED.pointer(views.w2_alpha), ROUTED.pointer(down_scale),
    )
    shared_weights = SharedWeights(
        ROUTED.pointer(source_weights.shared_gate), ROUTED.pointer(source_weights.shared_up),
        ROUTED.pointer(source_weights.shared_down),
        ROUTED.pointer(source_weights.shared_expert_gate),
    )
    full_weights = FullWeights(
        RouterWeights(ROUTED.pointer(packed), ROUTED.pointer(router_scales), ROUTED.pointer(router_alpha)),
        routed_weights, shared_weights,
    )
    native_routed_workspace = ROUTED.Workspace(
        ROUTED.pointer(routed_workspace.packed_a_flat), ROUTED.pointer(routed_workspace.scale_flat),
        ROUTED.pointer(routed_workspace.route_output_scratch), ROUTED.pointer(routed_workspace.barrier_count),
        ROUTED.pointer(routed_workspace.barrier_epoch), ROUTED.pointer(routed_workspace.row_counts),
        ROUTED.pointer(routed_workspace.active_expert_count), ROUTED.pointer(routed_workspace.weight_expert_ids),
        ROUTED.pointer(routed_workspace.global_to_local_expert), ROUTED.pointer(routed_workspace.virt_route_scratch),
        ROUTED.pointer(routed_workspace.token_map), ROUTED.pointer(routed_workspace.token_weights),
    )
    full_workspace = FullWorkspace(
        ROUTED.pointer(logits), ROUTED.pointer(global_ids), ROUTED.pointer(routing_weights),
        ROUTED.pointer(local_ids), ROUTED.pointer(local_weights),
        ROUTED.pointer(source_generation), ROUTED.pointer(requested_generation),
        ROUTED.pointer(route_summary), native_routed_workspace,
        ROUTED.pointer(shared_gate_scratch), ROUTED.pointer(shared_up_scratch),
        ROUTED.pointer(shared_gate_scalar),
    )
    launch = FullLaunch(
        ROUTED.pointer(hidden), ROUTED.pointer(output), full_workspace,
        int(stream.cuda_stream),
    )
    identity = ROUTED.Identity(
        (ctypes.c_uint8 * 32).from_buffer_copy(bytes.fromhex(slab.artifact_key)),
        (ctypes.c_uint8 * 32).from_buffer_copy(bytes.fromhex(slab.layout_sha256)),
        args.rank, args.layer,
    )
    library = bind(args.native_library)
    handle = ctypes.c_void_p()
    status = library.rocket_qwen38_target_full_moe_c1_create(
        torch.cuda.current_device(), ctypes.byref(identity), ctypes.byref(full_weights),
        ctypes.byref(handle),
    )
    if status != 0 or not handle.value:
        emit_failure(
            phase="create", status=status, rank=args.rank, layer=args.layer
        )
        raise RuntimeError(f"native full target MoE create failed: outcome={status}")
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            status = library.rocket_qwen38_target_full_moe_c1_enqueue(
                handle, ctypes.byref(launch)
            )
            if status != 0:
                emit_failure(
                    phase="capture", status=status, rank=args.rank,
                    layer=args.layer,
                )
                raise RuntimeError(f"native full target MoE capture failed: outcome={status}")
        graph.replay()
        stream.synchronize()
        absolute = (output.float() - oracle.float()).abs()
        relative = absolute / oracle.float().abs().clamp_min(1.0e-12)
        ids_match = bool(torch.equal(global_ids, oracle_ids))
        route_weight_max_abs = float((routing_weights - oracle_weights).abs().max().item())
        accepted = ids_match and route_weight_max_abs <= 1.0e-5 and bool(torch.allclose(
            output.float(), oracle.float(), atol=args.atol, rtol=args.rtol
        ))
        route_bytes = global_ids.cpu().numpy().tobytes() + routing_weights.cpu().numpy().tobytes()
        print(json.dumps({
            "abi": "rocket.qwen38.target-full-moe.c1.parity.v1",
            "accepted": accepted,
            "artifact_sha256": slab.artifact_key,
            "graph_capture": True,
            "ids_match": ids_match,
            "layer": args.layer,
            "max_abs": float(absolute.max().item()),
            "max_rel": float(relative.max().item()),
            "rank": args.rank,
            "route_sha256": hashlib.sha256(route_bytes).hexdigest(),
            "route_weight_max_abs": route_weight_max_abs,
            "shared_intermediate": [args.rank * 160, (args.rank + 1) * 160],
        }, sort_keys=True))
        if not accepted:
            raise RuntimeError("native full target MoE differs from eager oracle")
    finally:
        library.rocket_qwen38_target_full_moe_c1_destroy(handle)


if __name__ == "__main__":
    main()
