#!/usr/bin/env python3
"""Materialize rank-local GLM-5.3 routed experts in final CUDA slot order.

Each output is an O_DIRECT-ready file: a 64 KiB header, 64 KiB-padded scalar
metadata, then gapless 64 KiB-aligned 15 MiB expert slots. The slot bytes match
WeightStore::ExpertDev exactly, including both linear and CUTLASS-swizzled scale
copies, so serving performs sequential reads and H2D copies without touching
safetensors or transforming weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
from pathlib import Path

import numpy as np

PAGE = 65536
CHUNK = 256 << 20
MAGIC = b"ROCKETEXPERT1\0\0\0"
VERSION = 1
HEADER = struct.Struct("<16sIIQQQQQIIIIIII32s32s")
SLOT_META = struct.Struct("<4f")  # gate global/input, up global/input


def align_up(value: int, alignment: int = PAGE) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.digest()


class Checkpoint:
    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "model.safetensors.index.json"
        self.weight_map = json.loads(self.index_path.read_text())["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}
        self.maps: dict[str, tuple[object, mmap.mmap]] = {}

    def _header(self, shard: str) -> tuple[int, dict]:
        if shard not in self.headers:
            with (self.root / shard).open("rb") as f:
                header_bytes = int.from_bytes(f.read(8), "little")
                self.headers[shard] = (header_bytes, json.loads(f.read(header_bytes)))
        return self.headers[shard]

    def tensor(self, name: str) -> memoryview:
        shard = self.weight_map[name]
        header_bytes, header = self._header(shard)
        if shard not in self.maps:
            f = (self.root / shard).open("rb")
            self.maps[shard] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        start, end = header[name]["data_offsets"]
        base = 8 + header_bytes
        return memoryview(self.maps[shard][1])[base + start:base + end]

    def optional_f32(self, name: str, default: float) -> float:
        if name not in self.weight_map:
            return default
        return struct.unpack("<f", self.tensor(name))[0]


def scale_permutation(rows: int, cols: int) -> np.ndarray:
    """Map row-major linear scale bytes to SfLayout byte offsets."""
    if rows % 128 or cols % 4:
        raise ValueError(f"scale shape {rows}x{cols} is not atom aligned")
    row = np.arange(rows, dtype=np.int64)[:, None]
    col = np.arange(cols, dtype=np.int64)[None, :]
    atom = (row // 128) * (cols // 4) + col // 4
    within = (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
    return (atom * 512 + within).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuel", type=Path, default=Path(os.environ.get(
        "ROCKET_FUEL_NVFP4_DIR", "~/.cache/rocket-fuels/glm-5.3-flash-nvfp4")).expanduser())
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = ap.parse_args()

    cfg_path = args.fuel / "config.json"
    cfg = json.loads(cfg_path.read_text())["text_config"]
    hidden = int(cfg["hidden_size"])
    intermediate = int(cfg["moe_intermediate_size"])
    experts = int(cfg["n_routed_experts"])
    layers = int(cfg["num_hidden_layers"])
    sparse_layers = [i for i, kind in enumerate(cfg["mlp_layer_types"]) if kind == "sparse"]
    if hidden != 4096 or intermediate != 2048 or experts != 288 or len(sparse_layers) != 42:
        raise SystemExit("checkpoint geometry does not match the GLM-5.3 slab contract")

    count = experts // 2
    first = args.rank * count
    slot_count = len(sparse_layers) * count
    gate_packed = intermediate * hidden // 2
    gate_scale = intermediate * hidden // 16
    down_packed = hidden * intermediate // 2
    down_scale = hidden * intermediate // 16
    slot_bytes = 2 * (gate_packed + gate_scale) + 2 * gate_scale + down_packed + 2 * down_scale
    if slot_bytes != 15 << 20 or slot_bytes % PAGE:
        raise SystemExit(f"unexpected slot size {slot_bytes}")

    payload_bytes = slot_count * slot_bytes
    down_meta_bytes = layers * experts * 2 * 4
    slot_meta_offset = down_meta_bytes
    chunk_count = (payload_bytes + CHUNK - 1) // CHUNK
    digest_offset = down_meta_bytes + slot_count * SLOT_META.size
    metadata_bytes = align_up(digest_offset + chunk_count * 32)
    payload_offset = PAGE + metadata_bytes
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"rank{args.rank}-experts.slab"
    tmp = out.with_suffix(".slab.tmp")
    ckpt = Checkpoint(args.fuel)
    gate_perm = scale_permutation(intermediate, hidden // 16)
    down_perm = scale_permutation(hidden, intermediate // 16)

    header = HEADER.pack(
        MAGIC, VERSION, PAGE, metadata_bytes, payload_offset, payload_bytes,
        slot_bytes, slot_count, hidden, intermediate, layers, experts,
        first, count, len(sparse_layers), sha256(cfg_path), sha256(ckpt.index_path))
    metadata = bytearray(metadata_bytes)
    for layer in range(layers):
        if layer not in sparse_layers:
            continue
        for expert in range(experts):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.down_proj."
            struct.pack_into("<2f", metadata, (layer * experts + expert) * 8,
                             ckpt.optional_f32(prefix + "weight_scale_2", 0.0),
                             ckpt.optional_f32(prefix + "input_scale", 1.0))

    with tmp.open("wb", buffering=0) as w:
        w.write(header)
        w.write(bytes(PAGE - len(header)))
        w.write(metadata)  # overwritten after scalar collection
        slot_index = 0
        chunk_hash = hashlib.sha256()
        chunk_fill = 0
        chunk_digests = []

        def write_payload(blob: memoryview) -> None:
            nonlocal chunk_hash, chunk_fill
            offset = 0
            while offset < len(blob):
                take = min(CHUNK - chunk_fill, len(blob) - offset)
                part = blob[offset:offset + take]
                w.write(part)
                chunk_hash.update(part)
                offset += take
                chunk_fill += take
                if chunk_fill == CHUNK:
                    chunk_digests.append(chunk_hash.digest())
                    chunk_hash = hashlib.sha256()
                    chunk_fill = 0

        for layer in sparse_layers:
            for expert in range(first, first + count):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                pieces = []
                scalars = []
                for projection, rows, cols, perm in (
                    ("gate_proj", intermediate, hidden, gate_perm),
                    ("up_proj", intermediate, hidden, gate_perm),
                    ("down_proj", hidden, intermediate, down_perm),
                ):
                    packed = ckpt.tensor(prefix + projection + ".weight")
                    scale_view = ckpt.tensor(prefix + projection + ".weight_scale")
                    scale = np.frombuffer(scale_view, dtype=np.uint8)
                    swizzled = np.empty_like(scale)
                    swizzled[perm] = scale
                    pieces.append((packed, scale_view, memoryview(swizzled)))
                    if projection != "down_proj":
                        scalars.extend((
                            ckpt.optional_f32(prefix + projection + ".weight_scale_2", 0.0),
                            ckpt.optional_f32(prefix + projection + ".input_scale", 1.0),
                        ))
                gate, up, down = pieces
                for blob in (gate[0], up[0], gate[1], up[1], gate[2], up[2],
                             down[0], down[1], down[2]):
                    write_payload(blob)
                SLOT_META.pack_into(metadata, slot_meta_offset + slot_index * SLOT_META.size,
                                    *scalars)
                slot_index += 1
                if slot_index % count == 0:
                    w.flush()
                    print(f"rank {args.rank}: packed layer {layer}, {slot_index}/{slot_count} slots",
                          flush=True)
        if chunk_fill:
            chunk_digests.append(chunk_hash.digest())
        if len(chunk_digests) != chunk_count:
            raise RuntimeError("slab chunk digest count changed")
        for i, digest in enumerate(chunk_digests):
            metadata[digest_offset + i * 32:digest_offset + (i + 1) * 32] = digest
        if w.tell() != payload_offset + payload_bytes:
            raise RuntimeError(f"slab size {w.tell()} != {payload_offset + payload_bytes}")
        w.seek(PAGE)
        w.write(metadata)
        w.flush()
        os.fsync(w.fileno())
    os.replace(tmp, out)
    print(f"rank {args.rank}: {payload_bytes / 2**30:.5f} GiB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
