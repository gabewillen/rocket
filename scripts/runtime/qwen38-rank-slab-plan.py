#!/usr/bin/env python3
"""Plan TP2 rank-local slabs for the pinned NVIDIA Qwen3.8 checkpoint.

The planner reads safetensors headers and configuration only. It never reads tensor
payload bytes. Output entries retain their source file, blob identity, header digest,
file offsets, full tensor metadata, consumer assignment, and exact source slices.

Supported input is the complete text portion, including routed experts, of revision
``fc694b54fb0174e0913e6adf86691ef85a4ead47``. Visual tensors may be omitted only
under the explicit text-only contract. Every other unknown tensor is a fatal error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence, TextIO

SCHEMA = "rocket.qwen38-rank-slab-plan.v2"
PINNED_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
TP_SIZE = 2
TENSOR_ALIGNMENT_BYTES = 256
IO_ALIGNMENT_BYTES = 65_536
IO_CHUNK_BYTES = 256 * 1024 * 1024
CUDA_USABLE_BYTES = 106 * 1024**3
MAX_LAYOUT_OVERHEAD_PERCENT = 1
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
PLE_LOGICAL_ROWS = 320_001_446
PLE_PADDED_ROWS = 320_001_536
PLE_ROWS_PER_CHECKPOINT_SHARD = 2_500_012
PINNED_VISUAL_TENSORS = 333
PINNED_VISUAL_BYTES = 897_862_112
PINNED_TENSOR_COUNT = 299_545
PINNED_SOURCE_BYTES = 132_639_846_394
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class PlanError(ValueError):
    """A caller-visible validation failure; no payload bytes have been read."""


@dataclass(frozen=True)
class TensorHeader:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: str
    source_blob: str
    source_file_size_bytes: int
    source_header_sha256: str
    source_payload_offset_bytes: int
    source_data_offsets: tuple[int, int]

    @property
    def source_absolute_offset_bytes(self) -> int:
        return self.source_payload_offset_bytes + self.source_data_offsets[0]

    @property
    def byte_count(self) -> int:
        return self.source_data_offsets[1] - self.source_data_offsets[0]


@dataclass(frozen=True)
class SliceRule:
    family: str
    state: str
    dimension: int | None = None
    custom: str | None = None


def _read_exact(file: BinaryIO, count: int, label: str) -> bytes:
    value = file.read(count)
    if len(value) != count:
        raise PlanError(f"short read of {label}: {len(value)} of {count} bytes")
    return value


def _checked_shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise PlanError(f"tensor {name!r} has invalid shape")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise PlanError(f"tensor {name!r} has invalid shape")
    return tuple(value)


def _read_bounded_json(path: Path, limit: int, label: str) -> object:
    try:
        size = path.stat().st_size
        if size > limit:
            raise PlanError(f"{label} exceeds {limit} bytes")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read {label}: {exc}") from exc


def read_safetensors_header(path: Path) -> list[TensorHeader]:
    """Borrow ``path`` and read its bounded header without touching its payload."""

    try:
        with path.open("rb") as file:
            identity = os.fstat(file.fileno())
            raw_length = _read_exact(file, 8, "safetensors header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= MAX_HEADER_BYTES:
                raise PlanError(f"invalid header length in {path.name}: {header_length}")
            if 8 + header_length > identity.st_size:
                raise PlanError(f"header exceeds file size in {path.name}")
            raw_header = _read_exact(file, header_length, "safetensors header")
    except OSError as exc:
        raise PlanError(f"cannot read {path}: {exc}") from exc
    try:
        decoded = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanError(f"invalid header JSON in {path.name}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PlanError(f"header in {path.name} is not an object")

    resolved = path.resolve()
    blob = (
        resolved.name
        if resolved != path.absolute()
        else hashlib.sha256(raw_header).hexdigest()
    )
    header_sha256 = hashlib.sha256(raw_header).hexdigest()
    payload_capacity = identity.st_size - 8 - header_length
    result: list[TensorHeader] = []
    for name, metadata in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise PlanError(f"invalid tensor metadata in {path.name}")
        dtype = metadata.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise PlanError(f"tensor {name!r} has unsupported dtype {dtype!r}")
        shape = _checked_shape(metadata.get("shape"), name)
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in offsets
            )
        ):
            raise PlanError(f"tensor {name!r} has invalid data offsets")
        start, end = offsets
        expected = DTYPE_BYTES[dtype]
        for extent in shape:
            expected *= extent
        if (
            start < 0
            or end < start
            or end > payload_capacity
            or end - start != expected
        ):
            raise PlanError(f"tensor {name!r} has inconsistent payload extent")
        result.append(
            TensorHeader(
                name=name,
                dtype=dtype,
                shape=shape,
                source_file=path.name,
                source_blob=blob,
                source_file_size_bytes=identity.st_size,
                source_header_sha256=header_sha256,
                source_payload_offset_bytes=8 + header_length,
                source_data_offsets=(start, end),
            )
        )
    return result


def _consumer(name: str) -> tuple[str, ...]:
    if name.startswith("mtp."):
        return ("mtp",)
    if name in {"lm_head.weight", "model.language_model.embed_tokens.weight"}:
        return ("target", "mtp")
    return ("target",)


def _suffix_rule(name: str) -> SliceRule | None:
    normalized = re.sub(r"^(?:model\.language_model\.)?layers\.\d+\.", "layers.N.", name)
    normalized = re.sub(r"^mtp\.layers\.\d+\.", "mtp.layers.N.", normalized)

    if name in {"lm_head.weight", "model.language_model.embed_tokens.weight"}:
        return SliceRule("token_embedding_or_head", "vocab", 0)
    if name in {"mtp.fc_embedding.weight", "mtp.fc_hidden.weight"}:
        return SliceRule("mtp_input_projection", "column", 0)
    if name in {"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"}:
        return SliceRule("mtp_input_norm", "replicated")

    if re.fullmatch(
        r"model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:down_proj|gate_proj|up_proj)\."
        r"(?:weight|weight_scale|weight_scale_2|input_scale)",
        name,
    ):
        return SliceRule("target_expert_nvfp4", "custom", custom="target_expert")
    if re.fullmatch(
        r"mtp\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:down_proj|gate_proj|up_proj)\.(?:weight|weight_scale_inv)",
        name,
    ):
        return SliceRule("mtp_expert_fp8_block", "custom", custom="mtp_expert")

    leaf = normalized.split("layers.N.", 1)[-1]
    if "hyper_connection" in leaf or normalized.startswith(
        ("model.language_model.hyper_connection_mixer.", "mtp.hyper_connection_mixer.")
    ):
        if leaf.endswith(
            (
                "hc_norm.weight",
                "input_mix_weight_down.weight",
                "input_mix_weight_up.weight",
                "block_inject_weight.weight",
            )
        ):
            return SliceRule("hyperconnection", "replicated")

    linear = {
        "linear_attn.in_proj_qkv.weight": SliceRule(
            "linear_attn.in_proj_qkv", "custom", custom="gdn_qkv"
        ),
        "linear_attn.in_proj_z.weight": SliceRule("linear_attn.in_proj_z", "column", 0),
        "linear_attn.in_proj_a.weight": SliceRule("linear_attn.in_proj_a", "column", 0),
        "linear_attn.in_proj_b.weight": SliceRule("linear_attn.in_proj_b", "column", 0),
        "linear_attn.out_proj.weight": SliceRule("linear_attn.out_proj", "row", 1),
        "linear_attn.conv1d.weight": SliceRule(
            "linear_attn.conv1d", "custom", custom="gdn_qkv"
        ),
        "linear_attn.A_log": SliceRule("linear_attn.A_log", "column", 0),
        "linear_attn.dt_bias": SliceRule("linear_attn.dt_bias", "column", 0),
        "linear_attn.norm.weight": SliceRule("linear_attn.norm", "replicated"),
    }
    if leaf in linear:
        return linear[leaf]

    if leaf.startswith("self_attn."):
        # index_qk_proj also ends in q_proj.weight. Match the replicated QSA
        # side branch before the sharded main-QKV suffixes.
        if leaf.endswith(
            (
                "indexer.index_qk_proj.weight",
                "indexer.q_layernorm.weight",
                "indexer.k_layernorm.weight",
            )
        ):
            return SliceRule("self_attn.indexer", "replicated")
        if leaf.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            return SliceRule("self_attn.qkv", "column", 0)
        if leaf.endswith("o_proj.weight"):
            return SliceRule("self_attn.o_proj", "row", 1)
        if leaf.endswith(("q_norm.weight", "k_norm.weight")):
            return SliceRule("self_attn.qk_norm", "replicated")

    if leaf.startswith("mlp."):
        if leaf == "mlp.gate.weight" or leaf == "mlp.shared_expert_gate.weight":
            return SliceRule("moe.router_or_shared_gate", "replicated")
        if leaf.endswith(
            ("shared_expert.gate_proj.weight", "shared_expert.up_proj.weight")
        ):
            return SliceRule("moe.shared_expert_up", "column", 0)
        if leaf.endswith("shared_expert.down_proj.weight"):
            return SliceRule("moe.shared_expert_down", "row", 1)

    if leaf.startswith("ple."):
        if leaf.endswith(
            (
                "key_proj.weight",
                "value_proj.weight",
                "conv1d.weight",
                "norm_key.weight",
                "norm_query.weight",
                "norm_conv.weight",
            )
        ):
            return SliceRule("ple_dense", "replicated")
        if leaf.endswith(
            (
                "ple_embedding.layer_multipliers",
                "ple_embedding.ngram_heads_offsets",
                "ple_embedding.ngram_heads_vocab_sizes",
            )
        ):
            return SliceRule("ple_metadata", "replicated")
        if re.search(r"ple_embedding\.ngram_embedding\.shard_\d+\.weight$", leaf):
            return SliceRule("ple_embedding", "custom", custom="ple_vocab_shard")
        if leaf.endswith("ple_embedding.ngram_embedding.weight_scale"):
            return SliceRule("ple_embedding.fp8_scale", "replicated")
    return None


def classify_tensor(name: str, shape: Sequence[int]) -> SliceRule | None:
    """Return the canonical rule, or ``None`` only for explicit exclusions."""

    if name.startswith("model.visual."):
        return None
    rule = _suffix_rule(name)
    if rule is None:
        if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            raise PlanError(
                f"unresolved quantization auxiliary tensor: {name} {tuple(shape)}"
            )
        raise PlanError(f"unresolved text non-expert tensor family: {name} {tuple(shape)}")
    return rule


def _axis_fragment(
    shape: Sequence[int], dimension: int, start: int, length: int
) -> dict[str, object]:
    sliced = list(shape)
    sliced[dimension] = length
    return {"dimension": dimension, "start": start, "length": length, "shape": sliced}


def _partition(extent: int, rank: int, label: str) -> tuple[int, int]:
    if extent % TP_SIZE:
        raise PlanError(f"{label} extent {extent} is not divisible by TP{TP_SIZE}")
    length = extent // TP_SIZE
    return rank * length, length


def _fragments(
    header: TensorHeader, rule: SliceRule, rank: int
) -> list[dict[str, object]]:
    shape = header.shape
    if rule.family == "ple_embedding.fp8_scale" and (
        header.dtype != "BF16" or shape != (1,)
    ):
        raise PlanError(
            f"PLE FP8 scale ABI drift for {header.name}: "
            f"expected ('BF16', (1,)), got {(header.dtype, shape)}"
        )
    if rule.state == "replicated":
        return [{"dimension": None, "start": 0, "length": None, "shape": list(shape)}]
    if rule.custom in {"target_expert", "mtp_expert"}:
        return _expert_fragments(header, rule, rank)
    if rule.custom == "gdn_qkv":
        if shape[0] != 10_240:
            raise PlanError(f"{header.name} expected GDN QKV extent 10240, got {shape[0]}")
        fragments = []
        target_offset = 0
        for logical, base, extent in (
            ("q", 0, 2048),
            ("k", 2048, 2048),
            ("v", 4096, 6144),
        ):
            local_start, local_length = _partition(extent, rank, header.name)
            fragment = _axis_fragment(shape, 0, base + local_start, local_length)
            fragment.update(
                {"logical_shard": logical, "target_axis0_start": target_offset}
            )
            target_offset += local_length
            fragments.append(fragment)
        return fragments
    if rule.custom == "ple_vocab_shard":
        match = re.search(r"\.shard_(\d+)\.weight$", header.name)
        if match is None:
            raise PlanError(f"invalid PLE shard name: {header.name}")
        shard_index = int(match.group(1))
        if not 0 <= shard_index < 128:
            raise PlanError(f"PLE shard index outside 0..127: {header.name}")
        if shape != (PLE_ROWS_PER_CHECKPOINT_SHARD, 160):
            raise PlanError(
                f"{header.name} has unexpected PLE checkpoint shard shape {shape}"
            )
        checkpoint_start = shard_index * PLE_ROWS_PER_CHECKPOINT_SHARD
        tp_rows = PLE_PADDED_ROWS // TP_SIZE
        owner = checkpoint_start // tp_rows
        if owner != rank:
            return []
        logical_rows = max(0, min(shape[0], PLE_LOGICAL_ROWS - checkpoint_start))
        return [
            {
                "dimension": 0,
                "start": 0,
                "length": logical_rows,
                "shape": [logical_rows, shape[1]],
                "checkpoint_shard": shard_index,
                "operation": "copy",
                "target_row_start": checkpoint_start - rank * tp_rows,
                "logical_global_row_start": checkpoint_start,
                "logical_global_row_end": checkpoint_start + logical_rows,
            }
        ]
    if rule.dimension is None:
        raise PlanError(f"internal slice rule has no dimension: {header.name}")
    start, length = _partition(shape[rule.dimension], rank, header.name)
    return [_axis_fragment(shape, rule.dimension, start, length)]


def _expert_fragments(
    header: TensorHeader, rule: SliceRule, rank: int
) -> list[dict[str, object]]:
    match = re.fullmatch(
        r"(?P<prefix>model\.language_model\.layers\.(?P<target_layer>\d+)|"
        r"mtp\.layers\.(?P<mtp_layer>\d+))\.mlp\.experts\."
        r"(?P<expert>\d+)\.(?P<projection>down_proj|gate_proj|up_proj)\."
        r"(?P<leaf>weight|weight_scale|weight_scale_2|input_scale|weight_scale_inv)",
        header.name,
    )
    if match is None:
        raise PlanError(f"invalid expert tensor name: {header.name}")
    expert = int(match.group("expert"))
    if not 0 <= expert < 512:
        raise PlanError(f"expert ID outside 0..511: {header.name}")
    owner = 0 if expert < 256 else 1
    if owner != rank:
        return []

    projection = match.group("projection")
    leaf = match.group("leaf")
    if rule.custom == "target_expert":
        layer = int(match.group("target_layer"))
        if not 0 <= layer < 48:
            raise PlanError(f"target expert layer outside 0..47: {header.name}")
        expected = {
            ("down_proj", "weight"): ("U8", (2560, 320)),
            ("down_proj", "weight_scale"): ("F8_E4M3", (2560, 40)),
            ("down_proj", "weight_scale_2"): ("F32", ()),
            ("down_proj", "input_scale"): ("F32", ()),
            ("gate_proj", "weight"): ("U8", (640, 1280)),
            ("gate_proj", "weight_scale"): ("F8_E4M3", (640, 160)),
            ("gate_proj", "weight_scale_2"): ("F32", ()),
            ("gate_proj", "input_scale"): ("F32", ()),
            ("up_proj", "weight"): ("U8", (640, 1280)),
            ("up_proj", "weight_scale"): ("F8_E4M3", (640, 160)),
            ("up_proj", "weight_scale_2"): ("F32", ()),
            ("up_proj", "input_scale"): ("F32", ()),
        }
        abi = "modelopt_nvfp4_group16"
    else:
        layer = int(match.group("mtp_layer"))
        if layer != 0:
            raise PlanError(f"MTP expert layer must be 0: {header.name}")
        expected = {
            ("down_proj", "weight"): ("F8_E4M3", (2560, 640)),
            ("down_proj", "weight_scale_inv"): ("BF16", (20, 5)),
            ("gate_proj", "weight"): ("F8_E4M3", (640, 2560)),
            ("gate_proj", "weight_scale_inv"): ("BF16", (5, 20)),
            ("up_proj", "weight"): ("F8_E4M3", (640, 2560)),
            ("up_proj", "weight_scale_inv"): ("BF16", (5, 20)),
        }
        abi = "fp8_e4m3_block_128x128"
    expected_metadata = expected.get((projection, leaf))
    if expected_metadata is None:
        raise PlanError(f"expert auxiliary is outside the proven ABI: {header.name}")
    if (header.dtype, header.shape) != expected_metadata:
        raise PlanError(
            f"expert tensor ABI drift for {header.name}: expected "
            f"{expected_metadata}, got {(header.dtype, header.shape)}"
        )
    return [
        {
            "dimension": None,
            "start": 0,
            "length": None,
            "shape": list(header.shape),
            "expert_id": expert,
            "expert_owner_rank": owner,
            "expert_abi": abi,
        }
    ]


def _element_count(shape: Sequence[int]) -> int:
    count = 1
    for extent in shape:
        count *= extent
    return count


def _local_shape(fragments: Sequence[dict[str, object]]) -> list[int]:
    first = list(fragments[0]["shape"])
    dimension = fragments[0]["dimension"]
    if len(fragments) == 1 or dimension is None:
        return first
    if any(fragment["dimension"] != dimension for fragment in fragments):
        raise PlanError(
            "one tensor cannot combine source slices from different dimensions"
        )
    if not isinstance(dimension, int):
        raise PlanError("combined tensor slice dimension is not an integer")
    first[dimension] = sum(int(fragment["shape"][dimension]) for fragment in fragments)
    return first


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _finalize_slab(slab: dict[str, object]) -> None:
    packed_bytes = int(slab["slab_bytes"])
    slab_bytes = _align(packed_bytes, IO_ALIGNMENT_BYTES)
    io_tail_padding = slab_bytes - packed_bytes
    if io_tail_padding:
        slab["padding_regions"].append(
            {
                "kind": "io_tail",
                "fill": "zero",
                "start_bytes": packed_bytes,
                "end_bytes": slab_bytes,
                "bytes": io_tail_padding,
            }
        )
    chunks = []
    for start in range(0, slab_bytes, IO_CHUNK_BYTES):
        end = min(start + IO_CHUNK_BYTES, slab_bytes)
        if start % IO_ALIGNMENT_BYTES or end % IO_ALIGNMENT_BYTES:
            raise PlanError("I/O chunk boundary is not 65536-byte aligned")
        chunks.append({"start_bytes": start, "end_bytes": end, "bytes": end - start})
    slab["packed_bytes_before_io_tail"] = packed_bytes
    slab["io_tail_padding_bytes"] = io_tail_padding
    slab["slab_bytes"] = slab_bytes
    slab["io_chunks"] = chunks


def _validate_capacity(slabs: dict[str, dict[str, object]]) -> dict[str, object]:
    ranks = []
    for rank in range(TP_SIZE):
        resident = sum(
            int(slabs[f"rank{rank}-{consumer}"]["slab_bytes"])
            for consumer in ("target", "mtp")
        )
        if resident > CUDA_USABLE_BYTES:
            raise PlanError(
                f"rank {rank} slabs exceed measured CUDA-usable memory: "
                f"{resident} > {CUDA_USABLE_BYTES}"
            )
        ranks.append(
            {
                "rank": rank,
                "resident_slab_bytes": resident,
                "cuda_usable_bytes": CUDA_USABLE_BYTES,
                "headroom_bytes": CUDA_USABLE_BYTES - resident,
                "fits": True,
            }
        )
    return {
        "measurement": "cudaMalloc granted and device-touched 106.00 GiB",
        "evidence": "blog/posts/hardware/2026-09-06-cuda-free-counts-page-cache/index.qmd",
        "ranks": ranks,
    }


def build_plan(
    headers: Iterable[TensorHeader], checkpoint: Path, *, text_only: bool = False
) -> dict[str, object]:
    """Build an immutable JSON-ready snapshot; inputs are borrowed and not retained."""

    ordered = sorted(headers, key=lambda item: (item.name, item.source_file))
    names: set[str] = set()
    classified: list[tuple[TensorHeader, SliceRule, tuple[str, ...]]] = []
    excluded = {"visual": {"tensors": 0, "bytes": 0}}
    for header in ordered:
        if header.name in names:
            raise PlanError(
                f"duplicate tensor name across checkpoint files: {header.name}"
            )
        names.add(header.name)
        if header.name.startswith("model.visual.") and not text_only:
            raise PlanError("visual tensors require explicit --text-only omission")
        rule = classify_tensor(header.name, header.shape)
        if rule is None:
            excluded["visual"]["tensors"] += 1
            excluded["visual"]["bytes"] += header.byte_count
            continue
        classified.append((header, rule, _consumer(header.name)))

    slabs = {
        f"rank{rank}-{consumer}": {
            "key": f"rank{rank}-{consumer}",
            "rank": rank,
            "consumer": consumer,
            "immutable": True,
            "slab_bytes": 0,
            "payload_bytes": 0,
            "tensor_alignment_bytes": TENSOR_ALIGNMENT_BYTES,
            "tensor_padding_bytes": 0,
            "padding_regions": [],
            "entries": [],
        }
        for rank in range(TP_SIZE)
        for consumer in ("target", "mtp")
    }
    payload_owners: dict[tuple[int, str], str] = {}

    def append_payload(
        *,
        slab_key: str,
        header: TensorHeader,
        rule: SliceRule,
        consumers: tuple[str, ...],
        fragments: list[dict[str, object]],
    ) -> None:
        slab = slabs[slab_key]
        local_shape = _local_shape(fragments)
        transforms = list(fragments)
        if rule.custom == "ple_vocab_shard":
            local_shape = list(header.shape)
            padding_rows = header.shape[0] - int(fragments[0]["length"])
            if padding_rows:
                transforms.append(
                    {
                        "operation": "zero_fill",
                        "dimension": 0,
                        "target_row_start": int(fragments[0]["target_row_start"])
                        + int(fragments[0]["length"]),
                        "length": padding_rows,
                        "shape": [padding_rows, header.shape[1]],
                    }
                )
        local_bytes = _element_count(local_shape) * DTYPE_BYTES[header.dtype]
        previous_end = int(slab["slab_bytes"])
        slab_offset = _align(previous_end, TENSOR_ALIGNMENT_BYTES)
        tensor_padding = slab_offset - previous_end
        if tensor_padding:
            slab["padding_regions"].append(
                {
                    "kind": "tensor_alignment",
                    "fill": "zero",
                    "start_bytes": previous_end,
                    "end_bytes": slab_offset,
                    "bytes": tensor_padding,
                }
            )
            slab["tensor_padding_bytes"] = (
                int(slab["tensor_padding_bytes"]) + tensor_padding
            )
        entry = {
            "entry_type": "payload",
            "name": header.name,
            "family": rule.family,
            "state": rule.state,
            "consumers": list(consumers),
            "physical_owner": slab_key,
            "dtype": header.dtype,
            "full_shape": list(header.shape),
            "local_shape": local_shape,
            "local_bytes": local_bytes,
            "slab_offset_bytes": slab_offset,
            "source": {
                "file": header.source_file,
                "blob": header.source_blob,
                "file_size_bytes": header.source_file_size_bytes,
                "header_sha256": header.source_header_sha256,
                "payload_offset_bytes": header.source_payload_offset_bytes,
                "data_offsets": list(header.source_data_offsets),
                "absolute_offset_bytes": header.source_absolute_offset_bytes,
                "byte_count": header.byte_count,
            },
            "source_slices": fragments,
            "transforms": transforms,
        }
        slab["entries"].append(entry)
        slab["payload_bytes"] = int(slab["payload_bytes"]) + local_bytes
        slab["slab_bytes"] = slab_offset + local_bytes

    for header, rule, consumers in classified:
        for rank in range(TP_SIZE):
            fragments = _fragments(header, rule, rank)
            if not fragments:
                continue
            physical_consumer = "target" if consumers == ("target", "mtp") else consumers[0]
            owner_key = f"rank{rank}-{physical_consumer}"
            traversal_key = (rank, header.name)
            if traversal_key in payload_owners:
                raise PlanError(
                    f"source tensor assigned to two consumer payloads: {header.name}"
                )
            payload_owners[traversal_key] = owner_key
            append_payload(
                slab_key=owner_key,
                header=header,
                rule=rule,
                consumers=consumers,
                fragments=fragments,
            )
            if consumers == ("target", "mtp"):
                slabs[f"rank{rank}-mtp"]["entries"].append(
                    {
                        "entry_type": "reference",
                        "name": header.name,
                        "family": rule.family,
                        "consumers": list(consumers),
                        "physical_owner": owner_key,
                        "payload_io": False,
                    }
                )

    for slab in slabs.values():
        _finalize_slab(slab)
        payload_bytes = int(slab["payload_bytes"])
        tensor_padding_bytes = int(slab["tensor_padding_bytes"])
        io_tail_padding_bytes = int(slab["io_tail_padding_bytes"])
        payload_entries = sum(
            entry["entry_type"] == "payload" for entry in slab["entries"]
        )
        if tensor_padding_bytes > payload_entries * (TENSOR_ALIGNMENT_BYTES - 1):
            raise PlanError(
                f"{slab['key']} tensor padding exceeds "
                f"{TENSOR_ALIGNMENT_BYTES - 1} bytes per payload entry"
            )
        if not 0 <= io_tail_padding_bytes < IO_ALIGNMENT_BYTES:
            raise PlanError(
                f"{slab['key']} I/O tail padding is outside the 64 KiB bound"
            )
        overhead_bytes = tensor_padding_bytes + io_tail_padding_bytes
        slab["layout_overhead_bytes"] = overhead_bytes
        slab["layout_overhead_percent"] = (
            0.0 if payload_bytes == 0 else 100.0 * overhead_bytes / payload_bytes
        )

    classified_source_bytes = sum(header.byte_count for header, _, _ in classified)
    accounted_source_bytes = classified_source_bytes + excluded["visual"]["bytes"]
    family_counts: dict[str, int] = {}
    for _, rule, _ in classified:
        family_counts[rule.family] = family_counts.get(rule.family, 0) + 1
    if checkpoint.name == PINNED_REVISION and text_only:
        visual = excluded["visual"]
        if visual != {"tensors": PINNED_VISUAL_TENSORS, "bytes": PINNED_VISUAL_BYTES}:
            raise PlanError(f"pinned visual inventory mismatch: {visual}")
        if len(classified) + visual["tensors"] != PINNED_TENSOR_COUNT:
            raise PlanError("pinned classified plus visual tensor count mismatch")
        if accounted_source_bytes != PINNED_SOURCE_BYTES:
            raise PlanError(
                "pinned accounted source byte mismatch: "
                f"{accounted_source_bytes} != {PINNED_SOURCE_BYTES}"
            )
        expected_experts = {
            "target_expert_nvfp4": 294_912,
            "mtp_expert_fp8_block": 3_072,
        }
        observed_experts = {
            family: family_counts.get(family, 0) for family in expected_experts
        }
        if observed_experts != expected_experts:
            raise PlanError(f"pinned expert inventory mismatch: {observed_experts}")
        for slab in slabs.values():
            payload_bytes = int(slab["payload_bytes"])
            overhead_bytes = int(slab["layout_overhead_bytes"])
            if payload_bytes and overhead_bytes > payload_bytes // 100:
                raise PlanError(
                    f"{slab['key']} pinned-checkpoint layout overhead exceeds "
                    f"{MAX_LAYOUT_OVERHEAD_PERCENT}%: {overhead_bytes} bytes"
                )

    return {
        "schema": SCHEMA,
        "checkpoint_revision": PINNED_REVISION,
        "checkpoint": str(checkpoint.absolute()),
        "tensor_parallel_size": TP_SIZE,
        "tensor_alignment_bytes": TENSOR_ALIGNMENT_BYTES,
        "io_alignment_bytes": IO_ALIGNMENT_BYTES,
        "io_chunk_bytes": IO_CHUNK_BYTES,
        "payload_io_performed": False,
        "assignment_phase": "complete-before-payload-io",
        "source_traversal_contract": (
            "at most once per rank; shared target/MTP tensors are target-owned "
            "and MTP-referenced"
        ),
        "inventory": {
            "checkpoint_tensors": len(ordered),
            "classified_entries": len(classified),
            "classified_source_bytes": classified_source_bytes,
            "accounted_source_bytes": accounted_source_bytes,
            "family_counts": dict(sorted(family_counts.items())),
        },
        "excluded_tensor_counts": excluded,
        "text_only_omission": (
            {
                "enabled": True,
                "prefix": "model.visual.",
                "construction_guard": "language_model_only=true",
                "input_guard": "reject multimodal inputs and image/video token IDs",
                "loader_guard": "skip model.visual.* before payload I/O",
            }
            if text_only
            else {"enabled": False}
        ),
        "expert_ownership": {
            "target": {
                "abi": "modelopt_nvfp4_group16",
                "rank_ranges": [[0, 256], [256, 512]],
            },
            "mtp": {
                "abi": "fp8_e4m3_block_128x128",
                "rank_ranges": [[0, 256], [256, 512]],
            },
        },
        "ple_transform": {
            "logical_rows": PLE_LOGICAL_ROWS,
            "padded_rows": PLE_PADDED_ROWS,
            "rank_ranges": [[0, 160_000_768], [160_000_768, PLE_LOGICAL_ROWS]],
            "rank_1_padding_rows": 90,
        },
        "capacity_proof": _validate_capacity(slabs),
        "slabs": slabs,
    }


def load_pinned_checkpoint(checkpoint: Path) -> list[TensorHeader]:
    """Read all pinned shard headers after validating the checkpoint identity."""

    root = checkpoint.absolute()
    if root.name != PINNED_REVISION:
        raise PlanError(f"checkpoint directory must be pinned revision {PINNED_REVISION}")
    config = _read_bounded_json(root / "config.json", MAX_CONFIG_BYTES, "config")
    text = config.get("text_config") if isinstance(config, dict) else None
    expected = {
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "hc_count": 4,
        "split_ngram_parts": 128,
        "vocab_size": 248320,
    }
    if not isinstance(text, dict):
        raise PlanError("checkpoint config has no text_config object")
    mismatches = {
        key: (text.get(key), value)
        for key, value in expected.items()
        if text.get(key) != value
    }
    if mismatches:
        raise PlanError(
            "checkpoint config does not match pinned Qwen3.8 shape contract: "
            f"{mismatches}"
        )
    files = sorted(root.glob("model-*-of-*.safetensors"))
    overlay = root / "model-fp8-mtp-ple.safetensors"
    if overlay.is_file():
        files.append(overlay)
    if len(files) != 11:
        raise PlanError(f"expected 10 base shards and one PLE overlay, found {len(files)}")
    headers: list[TensorHeader] = []
    for path in files:
        headers.extend(read_safetensors_header(path))
    index = _read_bounded_json(
        root / "model.safetensors.index.json", MAX_INDEX_BYTES, "safetensors index"
    )
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise PlanError("safetensors index has no weight_map object")
    observed = {header.name: header.source_file for header in headers}
    if weight_map != observed:
        missing = sorted(set(weight_map) - set(observed))[:3]
        extra = sorted(set(observed) - set(weight_map))[:3]
        misplaced = sorted(
            name
            for name in set(weight_map) & set(observed)
            if weight_map[name] != observed[name]
        )[:3]
        raise PlanError(
            "checkpoint headers disagree with safetensors index: "
            f"missing={missing}, extra={extra}, misplaced={misplaced}"
        )
    return headers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--text-only", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Write one stable JSON manifest to stdout; diagnostics go to stderr."""

    args = _parser().parse_args(argv)
    try:
        headers = load_pinned_checkpoint(args.checkpoint)
        plan = build_plan(headers, args.checkpoint, text_only=args.text_only)
    except PlanError as exc:
        print(f"error: {exc}", file=stderr)
        return 2
    json.dump(plan, stdout, indent=2, sort_keys=True)
    stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
