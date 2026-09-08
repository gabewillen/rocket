#!/usr/bin/env python3
"""Bitwise CPU gate for native versus pinned-FlashInfer target MoE padding."""

from __future__ import annotations

import argparse
import array
import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engines/qwen38-flash-next-nvfp4-2b"
sys.path.insert(0, str(ENGINE / "src"))

from qwen38_slab.routed_moe import load_owner_local_moe  # noqa: E402

E, H, N, NP = 256, 2560, 640, 768
FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"
FLASHINFER_DISPATCH_SHA256 = "c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a"
FLASHINFER_UTILS_SHA256 = "971faf2b712c8d973cf7b81fdb97b12d04d0fb9ff350c314981c1049b3816a25"
FLASHINFER_FP4_HELPERS_SHA256 = "3115cf3e73064cbf2c339d5fd2cec1c49206f34283c281faef2d1c2d290a6c73"
FLASHINFER_W4A16_HOST_SHA256 = "49622d643b9acc602715bfbfbfa4ffc75b000fbc26ffcb8247326554edec0e43"


class Logical(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "w13_packed", "w13_scale", "down_packed", "down_scale",
        "input_global_scale", "w1_alpha", "w2_alpha", "down_input_scale",
    )]


def _read(fd: int, extent) -> bytes:
    import os
    value = os.pread(fd, extent.length, extent.offset)
    if len(value) != extent.length:
        raise RuntimeError(f"short authenticated extent: {extent.name}")
    return value


def _logical(rank_slab):
    import os
    by_name = {x.name: x for x in (*rank_slab.routed, *rank_slab.shared)}
    w13, s13, down, sd = bytearray(), bytearray(), bytearray(), bytearray()
    input_scale, a1, a2, down_input = (array.array("f") for _ in range(4))
    fd = os.open(rank_slab.slab_path, os.O_RDONLY)
    try:
        for local, expert in enumerate(range(rank_slab.first_expert,
                                             rank_slab.last_expert + 1)):
            root = f"model.language_model.layers.3.mlp.experts.{expert}"
            up = _read(fd, by_name[f"{root}.up_proj.weight"])
            gate = _read(fd, by_name[f"{root}.gate_proj.weight"])
            w13.extend(up); w13.extend(gate)
            ups = _read(fd, by_name[f"{root}.up_proj.weight_scale"])
            gates = _read(fd, by_name[f"{root}.gate_proj.weight_scale"])
            s13.extend(ups); s13.extend(gates)
            down.extend(_read(fd, by_name[f"{root}.down_proj.weight"]))
            sd.extend(_read(fd, by_name[f"{root}.down_proj.weight_scale"]))
            gate_a = struct.unpack("<f", _read(fd, by_name[f"{root}.gate_proj.weight_scale_2"]))[0]
            up_a = struct.unpack("<f", _read(fd, by_name[f"{root}.up_proj.weight_scale_2"]))[0]
            gate_i = struct.unpack("<f", _read(fd, by_name[f"{root}.gate_proj.input_scale"]))[0]
            up_i = struct.unpack("<f", _read(fd, by_name[f"{root}.up_proj.input_scale"]))[0]
            if gate_a != up_a or gate_i != up_i:
                raise RuntimeError("fused FC1 scalar source changed")
            a1.append(gate_a); input_scale.append(gate_i)
            a2.append(struct.unpack("<f", _read(fd, by_name[f"{root}.down_proj.weight_scale_2"]))[0])
            down_input.append(struct.unpack("<f", _read(fd, by_name[f"{root}.down_proj.input_scale"]))[0])
    finally:
        os.close(fd)
    return w13, s13, down, sd, input_scale, a1, a2, down_input


def _flashinfer_physical(rank_slab):
    import torch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.cute_dsl import utils as cute_utils
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (
        moe_w4a16_fp4_helpers, moe_w4a16_host,
    )
    from qwen38_slab.routed_moe import materialize_flashinfer_weights

    for module, expected in (
        (moe_dispatch, FLASHINFER_DISPATCH_SHA256),
        (cute_utils, FLASHINFER_UTILS_SHA256),
        (moe_w4a16_fp4_helpers, FLASHINFER_FP4_HELPERS_SHA256),
        (moe_w4a16_host, FLASHINFER_W4A16_HOST_SHA256),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("pinned FlashInfer source identity changed")
    source = materialize_flashinfer_weights(
        rank_slab, torch_api=torch, device="cpu")
    padded = moe_dispatch._pad_intermediate_to_tile(
        source.w1_weight, source.w1_scale, source.w2_weight, source.w2_scale,
        source.fc2_input_scale, N, 256, H, E, True, "nvfp4")
    w1, w1_sf, w2, w2_sf, down_scale, physical_n = padded
    if physical_n != NP:
        raise RuntimeError("pinned FlashInfer physical intermediate changed")
    views = moe_dispatch._get_weight_views(
        w1, w1_sf, w2, w2_sf, source.w1_alpha, source.w2_alpha,
        physical_n, H, activation_precision="fp4", quant_mode="nvfp4")
    folded = (views.w1_alpha * source.input_scale).contiguous()
    # Hash physical backing storage, not the strided six-dimensional scale view.
    return (
        views.w1_storage, views._w13_sf_storage, views.w2_storage,
        views._down_sf_storage, source.input_scale, folded,
        views.w2_alpha, down_scale,
    )


def _digest(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        if hasattr(value, "detach"):
            import torch
            value = value.detach().contiguous().view(torch.uint8).view(-1).numpy()
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    args = parser.parse_args()
    native = ctypes.CDLL(str(args.native_library))
    native.rocket_qwen38_target_moe_n640_hash.argtypes = [
        ctypes.POINTER(Logical), ctypes.c_void_p, ctypes.c_void_p]
    native.rocket_qwen38_target_moe_n640_hash.restype = ctypes.c_int
    native.rocket_qwen38_target_moe_n640_plane_hashes.argtypes = [
        ctypes.POINTER(Logical), ctypes.c_void_p]
    native.rocket_qwen38_target_moe_n640_plane_hashes.restype = ctypes.c_int
    results = []
    for rank in (0, 1):
        slab = load_owner_local_moe(args.artifact, rank, 3)
        logical = _logical(slab)
        keepalive = [
            (ctypes.c_ubyte * len(value)).from_buffer(value)
            if isinstance(value, bytearray)
            else (ctypes.c_float * len(value)).from_buffer(value)
            for value in logical
        ]
        pointers = Logical(*(ctypes.cast(value, ctypes.c_void_p) for value in keepalive))
        source_out, physical_out = (ctypes.c_uint8 * 32)(), (ctypes.c_uint8 * 32)()
        status = native.rocket_qwen38_target_moe_n640_hash(
            ctypes.byref(pointers), source_out, physical_out)
        if status:
            raise RuntimeError("native N640 materializer hash failed")
        python_source = _digest(logical)
        flashinfer_physical = _digest(_flashinfer_physical(slab))
        native_source = bytes(source_out).hex()
        native_physical = bytes(physical_out).hex()
        plane_out = ((ctypes.c_uint8 * 32) * 8)()
        if native.rocket_qwen38_target_moe_n640_plane_hashes(
                ctypes.byref(pointers), plane_out):
            raise RuntimeError("native per-plane digest failed")
        reference_planes = _flashinfer_physical(slab)
        native_plane_hashes = [bytes(value).hex() for value in plane_out]
        reference_plane_hashes = [_digest((value,)) for value in reference_planes]
        if python_source != native_source or flashinfer_physical != native_physical:
            raise RuntimeError(
                f"rank{rank} transformed plane digest mismatch "
                f"native={native_physical} flashinfer={flashinfer_physical} "
                f"planes={[(i, n, r) for i, (n, r) in enumerate(zip(native_plane_hashes, reference_plane_hashes, strict=True)) if n != r]}")
        results.append({"rank": rank, "source_sha256": native_source,
                        "physical_sha256": native_physical,
                        "reference": f"flashinfer-{FLASHINFER_COMMIT}",
                        "dispatch_sha256": FLASHINFER_DISPATCH_SHA256,
                        "utils_sha256": FLASHINFER_UTILS_SHA256,
                        "fp4_helpers_sha256": FLASHINFER_FP4_HELPERS_SHA256,
                        "w4a16_host_sha256": FLASHINFER_W4A16_HOST_SHA256,
                        "match": True})
    print(json.dumps({"schema": "rocket.qwen38.target-moe-n768-bitwise.v1",
                      "valid": True, "complete": True, "results": results},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({
            "schema": "rocket.qwen38.target-moe-n768-bitwise.v1",
            "valid": False, "complete": False, "phase": "compare",
            "failure_class": "contract", "reason": str(error)[:512],
        }, sort_keys=True))
        raise SystemExit(1) from None
