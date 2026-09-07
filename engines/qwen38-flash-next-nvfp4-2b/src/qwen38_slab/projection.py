"""Authenticated layer-3 rank-slab slice for a fixed K0 Q/K/V proof."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from .contract import MODEL_NVFP4_ABI, PINNED_CONTRACT, SCHEMA, canonical_bytes

PROJECTION_LAYER = 3
PROJECTION_K = 2560
PROJECTION_ROWS_PER_FAMILY = 4
PROJECTION_FAMILIES = ("q_proj", "k_proj", "v_proj")
PROJECTION_FAMILY_ROWS = (6144, 256, 256)
PROJECTION_OUTPUTS = len(PROJECTION_FAMILIES) * PROJECTION_ROWS_PER_FAMILY
PROJECTION_SCHEMA = "qwen3.8-flash-next:tp2:rank0:layer3:qkv-o-w4a4:v2"
OUTPUT_PROJECTION_K = 3072
OUTPUT_PROJECTION_N = 2560
OUTPUT_ACTIVATION_GLOBAL_SCALE = 1.0 / 256.0


class ProjectionError(RuntimeError):
    """Rank-slab identity, extent, or projection payload failure."""


@dataclass(frozen=True)
class ProjectionComponent:
    name: str
    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: str
    layout: str


@dataclass(frozen=True)
class RankSlabProjection:
    """Immutable authenticated offsets for one representative Q/K/V layer."""

    schema: str
    artifact_key: str
    rank: int
    layer: int
    slab_path: Path
    slab_bytes: int
    chunk_offset: int
    chunk_length: int
    chunk_sha256: str
    components: tuple[ProjectionComponent, ...]


@dataclass(frozen=True)
class QkvProjectionPayload:
    """Owned compact first-four-row Q/K/V data from an authenticated slab."""

    descriptor: RankSlabProjection
    packed_weights: bytes
    linear_scales: bytes
    global_scales: tuple[float, float, float]
    activations_bf16: bytes


@dataclass(frozen=True)
class FullQkvProjectionPayload:
    """Owned production-width Q/K/V tensors in the authenticated slab ABI."""

    descriptor: RankSlabProjection
    packed_weights: tuple[bytes, bytes, bytes]
    swizzled_scales: tuple[bytes, bytes, bytes]
    global_scales: tuple[float, float, float]
    activations_bf16: bytes
    output_weight: bytes
    output_scale: bytes
    output_global_scale: float


def load_rank0_layer3_projection(artifact: Path) -> RankSlabProjection:
    """Authenticate the manifest and source chunk, then bind exact Q/K/V extents."""

    if not isinstance(artifact, Path):
        raise ProjectionError("rank slab artifact must be a Path")
    try:
        raw = (artifact / "manifest.json").read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProjectionError("cannot load rank slab manifest") from exc
    if not isinstance(manifest, dict):
        raise ProjectionError("rank slab manifest root is invalid")
    digest_input = dict(manifest)
    claimed = digest_input.pop("artifact_key", None)
    observed = hashlib.sha256(canonical_bytes(digest_input)).hexdigest()
    if claimed != observed or artifact.name != observed:
        raise ProjectionError("rank slab manifest content address mismatch")
    expected = {
        "schema": SCHEMA,
        "revision": PINNED_CONTRACT.revision,
        "overlay_artifact_key": PINNED_CONTRACT.artifact_key,
        "overlay_sha256": PINNED_CONTRACT.overlay_sha256,
        "tp_size": 2,
        "tensor_alignment_bytes": PINNED_CONTRACT.tensor_alignment_bytes,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ProjectionError("rank slab manifest contract drift")
    slab = manifest.get("slabs", {}).get("rank0-target")
    if not isinstance(slab, dict) or not isinstance(slab.get("entries"), list):
        raise ProjectionError("rank0 target slab inventory is missing")
    by_name = {entry.get("name"): entry for entry in slab["entries"]}
    components = []
    for family, rows in (("q_proj", 6144), ("k_proj", 256), ("v_proj", 256)):
        prefix = f"model.language_model.layers.3.self_attn.{family}"
        contracts = (
            (
                "weight",
                (rows, PROJECTION_K // 2),
                "U8",
                "packed_e2m1_row_major",
                rows * PROJECTION_K // 2,
            ),
            (
                "weight_scale",
                (rows * PROJECTION_K // 16,),
                "F8_E4M3",
                "cutlass_sm121_sfb",
                rows * PROJECTION_K // 16,
            ),
            ("weight_scale_2", (1,), "F32", "scalar", 4),
        )
        for suffix, shape, dtype, layout, expected_length in contracts:
            name = f"{prefix}.{suffix}"
            entry = by_name.get(name)
            if (
                not isinstance(entry, dict)
                or entry.get("abi") != MODEL_NVFP4_ABI
                or tuple(entry.get("shape", ())) != shape
                or entry.get("dtype") != dtype
                or entry.get("layout") != layout
                or entry.get("length_bytes") != expected_length
            ):
                raise ProjectionError(f"rank slab projection contract drift: {name}")
            offset, length = entry.get("offset_bytes"), entry.get("length_bytes")
            if (
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or offset < 0
                or length <= 0
                or offset % PINNED_CONTRACT.tensor_alignment_bytes
            ):
                raise ProjectionError(f"rank slab projection extent is invalid: {name}")
            components.append(
                ProjectionComponent(name, offset, length, shape, dtype, layout)
            )
    prefix = "model.language_model.layers.3.self_attn.o_proj"
    for suffix, shape, dtype, layout, expected_length in (
        ("weight", (OUTPUT_PROJECTION_N, OUTPUT_PROJECTION_K // 2), "U8",
         "packed_e2m1_row_major", OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 2),
        ("weight_scale", (OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 16,),
         "F8_E4M3", "cutlass_sm121_sfb", OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 16),
        ("weight_scale_2", (1,), "F32", "scalar", 4),
    ):
        name = f"{prefix}.{suffix}"
        entry = by_name.get(name)
        if (
            not isinstance(entry, dict) or entry.get("abi") != MODEL_NVFP4_ABI
            or tuple(entry.get("shape", ())) != shape or entry.get("dtype") != dtype
            or entry.get("layout") != layout or entry.get("length_bytes") != expected_length
        ):
            raise ProjectionError(f"rank slab projection contract drift: {name}")
        offset, length = entry.get("offset_bytes"), entry.get("length_bytes")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0 or offset % 256:
            raise ProjectionError(f"rank slab projection extent is invalid: {name}")
        components.append(ProjectionComponent(name, offset, length, shape, dtype, layout))
    slab_path = artifact / str(slab.get("file", ""))
    slab_bytes = slab.get("bytes")
    try:
        observed_bytes = slab_path.stat().st_size
    except OSError as exc:
        raise ProjectionError("rank0 target slab file is missing") from exc
    if observed_bytes != slab_bytes:
        raise ProjectionError("rank0 target slab byte count drift")
    touched = [
        chunk
        for chunk in slab.get("chunks", [])
        if any(
            component.offset < chunk["offset_bytes"] + chunk["length_bytes"]
            and component.offset + component.length > chunk["offset_bytes"]
            for component in components
        )
    ]
    if len(touched) != 1:
        raise ProjectionError("projection components must occupy one authenticated chunk")
    chunk = touched[0]
    chunk_offset, chunk_length = chunk["offset_bytes"], chunk["length_bytes"]
    if _sha256_range(slab_path, chunk_offset, chunk_length) != chunk.get("sha256"):
        raise ProjectionError("rank slab projection chunk digest mismatch")
    return RankSlabProjection(
        PROJECTION_SCHEMA,
        claimed,
        0,
        PROJECTION_LAYER,
        slab_path,
        slab_bytes,
        chunk_offset,
        chunk_length,
        chunk["sha256"],
        tuple(components),
    )


def load_projection_payload(descriptor: RankSlabProjection) -> QkvProjectionPayload:
    """Read a fixed first-four-row projection slice from authenticated extents."""

    if not isinstance(descriptor, RankSlabProjection) or descriptor.schema != PROJECTION_SCHEMA:
        raise ProjectionError("projection descriptor is invalid")
    by_name = {component.name: component for component in descriptor.components}
    packed = bytearray()
    scales = bytearray()
    globals_ = []
    fd = os.open(descriptor.slab_path, os.O_RDONLY)
    try:
        if (
            os.fstat(fd).st_size != descriptor.slab_bytes
            or _sha256_fd_range(
                fd, descriptor.chunk_offset, descriptor.chunk_length
            )
            != descriptor.chunk_sha256
        ):
            raise ProjectionError("rank slab changed after descriptor authentication")
        for family in PROJECTION_FAMILIES:
            prefix = f"model.language_model.layers.3.self_attn.{family}"
            weight = by_name[f"{prefix}.weight"]
            scale = by_name[f"{prefix}.weight_scale"]
            global_scale = by_name[f"{prefix}.weight_scale_2"]
            row_bytes = PROJECTION_K // 2
            raw = os.pread(fd, PROJECTION_ROWS_PER_FAMILY * row_bytes, weight.offset)
            if len(raw) != PROJECTION_ROWS_PER_FAMILY * row_bytes:
                raise ProjectionError("short packed projection read")
            packed.extend(raw)
            scale_blob = os.pread(fd, scale.length, scale.offset)
            if len(scale_blob) != scale.length:
                raise ProjectionError("short SFB projection scale read")
            for row in range(PROJECTION_ROWS_PER_FAMILY):
                for sf_index in range(PROJECTION_K // 16):
                    scales.append(scale_blob[_sfb_offset(row, sf_index, PROJECTION_K)])
            raw_global = os.pread(fd, 4, global_scale.offset)
            if len(raw_global) != 4:
                raise ProjectionError("short global projection scale read")
            globals_.append(struct.unpack("<f", raw_global)[0])
    finally:
        os.close(fd)
    activation = bytearray()
    for row in range(16):
        for column in range(PROJECTION_K):
            value = ((column + row * 3) % 17 - 8) / 16.0
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            activation.extend(struct.pack("<H", bits >> 16))
    return QkvProjectionPayload(
        descriptor,
        bytes(packed),
        bytes(scales),
        tuple(globals_),
        bytes(activation),
    )


def load_full_projection_payload(
    descriptor: RankSlabProjection,
) -> FullQkvProjectionPayload:
    """Read full production widths without changing packed weight or SFB layout."""

    if not isinstance(descriptor, RankSlabProjection) or descriptor.schema != PROJECTION_SCHEMA:
        raise ProjectionError("projection descriptor is invalid")
    by_name = {component.name: component for component in descriptor.components}
    weights: list[bytes] = []
    scales: list[bytes] = []
    globals_: list[float] = []
    fd = os.open(descriptor.slab_path, os.O_RDONLY)
    try:
        if (
            os.fstat(fd).st_size != descriptor.slab_bytes
            or _sha256_fd_range(fd, descriptor.chunk_offset, descriptor.chunk_length)
            != descriptor.chunk_sha256
        ):
            raise ProjectionError("rank slab changed after descriptor authentication")
        for family in PROJECTION_FAMILIES:
            prefix = f"model.language_model.layers.3.self_attn.{family}"
            weight = by_name[f"{prefix}.weight"]
            scale = by_name[f"{prefix}.weight_scale"]
            global_scale = by_name[f"{prefix}.weight_scale_2"]
            weight_blob = os.pread(fd, weight.length, weight.offset)
            scale_blob = os.pread(fd, scale.length, scale.offset)
            raw_global = os.pread(fd, 4, global_scale.offset)
            if len(weight_blob) != weight.length or len(scale_blob) != scale.length:
                raise ProjectionError("short full projection read")
            if len(raw_global) != 4:
                raise ProjectionError("short global projection scale read")
            weights.append(weight_blob)
            scales.append(scale_blob)
            globals_.append(struct.unpack("<f", raw_global)[0])
        output_prefix = "model.language_model.layers.3.self_attn.o_proj"
        output_weight = os.pread(fd, by_name[f"{output_prefix}.weight"].length,
                                 by_name[f"{output_prefix}.weight"].offset)
        output_scale = os.pread(fd, by_name[f"{output_prefix}.weight_scale"].length,
                                by_name[f"{output_prefix}.weight_scale"].offset)
        raw_output_global = os.pread(fd, 4, by_name[f"{output_prefix}.weight_scale_2"].offset)
    finally:
        os.close(fd)
    activation = bytearray()
    for row in range(16):
        for column in range(PROJECTION_K):
            value = ((column + row * 3) % 17 - 8) / 16.0
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            activation.extend(struct.pack("<H", bits >> 16))
    if len(output_weight) != OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 2 or len(output_scale) != OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 16 or len(raw_output_global) != 4:
        raise ProjectionError("short output projection read")
    return FullQkvProjectionPayload(
        descriptor,
        tuple(weights),
        tuple(scales),
        tuple(globals_),
        bytes(activation),
        output_weight,
        output_scale,
        struct.unpack("<f", raw_output_global)[0],
    )


def reference_projection(
    payload: QkvProjectionPayload, active_rows: int
) -> tuple[float, ...]:
    """Return the scalar W4A4 definition for the fixed representative slice."""

    if (
        not isinstance(payload, QkvProjectionPayload)
        or isinstance(active_rows, bool)
        or not isinstance(active_rows, int)
        or not 0 <= active_rows <= 16
    ):
        raise ProjectionError("reference projection inputs are invalid")
    result = [0.0] * (16 * PROJECTION_OUTPUTS)
    packed_stride = PROJECTION_K // 2
    scale_stride = PROJECTION_K // 16
    for batch in range(active_rows):
        activation_base = batch * PROJECTION_K * 2
        quantized = []
        activation_scales = []
        for group in range(PROJECTION_K // 16):
            values = []
            for offset in range(16):
                column = group * 16 + offset
                raw = payload.activations_bf16[
                    activation_base + 2 * column : activation_base + 2 * column + 2
                ]
                bf16 = struct.unpack("<H", raw)[0]
                values.append(
                    struct.unpack("<f", struct.pack("<I", bf16 << 16))[0]
                )
            scale = max(abs(value) for value in values) / 6.0
            activation_scales.append(scale)
            for value in values:
                normalized = value / scale if scale else 0.0
                code = min(range(16), key=lambda item: abs(normalized - _e2m1(item)))
                quantized.append(_e2m1(code))
        for output in range(PROJECTION_OUTPUTS):
            total = 0.0
            packed_base = output * packed_stride
            scale_base = output * scale_stride
            for column in range(PROJECTION_K):
                pair = payload.packed_weights[packed_base + column // 2]
                nibble = pair >> 4 if column & 1 else pair & 15
                total += (
                    quantized[column]
                    * activation_scales[column // 16]
                    * _e2m1(nibble)
                    * _e4m3(payload.linear_scales[scale_base + column // 16])
                )
            result[batch * PROJECTION_OUTPUTS + output] = (
                total * payload.global_scales[output // PROJECTION_ROWS_PER_FAMILY]
            )
    return tuple(result)


def reference_cutlass_projection(
    payload: QkvProjectionPayload, active_rows: int
) -> tuple[float, ...]:
    """Scalar oracle with the E4M3 activation scales consumed by CUTLASS."""

    if (
        not isinstance(payload, QkvProjectionPayload)
        or isinstance(active_rows, bool)
        or not isinstance(active_rows, int)
        or not 0 <= active_rows <= 16
    ):
        raise ProjectionError("reference projection inputs are invalid")
    result = [0.0] * (16 * PROJECTION_OUTPUTS)
    packed_stride = PROJECTION_K // 2
    scale_stride = PROJECTION_K // 16
    for batch in range(active_rows):
        activation_base = batch * PROJECTION_K * 2
        quantized = []
        activation_scales = []
        for group in range(PROJECTION_K // 16):
            values = []
            for offset in range(16):
                column = group * 16 + offset
                bf16 = struct.unpack(
                    "<H",
                    payload.activations_bf16[
                        activation_base + 2 * column : activation_base + 2 * column + 2
                    ],
                )[0]
                values.append(struct.unpack("<f", struct.pack("<I", bf16 << 16))[0])
            scale_code = _float_to_e4m3(max(abs(value) for value in values) / 6.0)
            scale = _e4m3(scale_code)
            activation_scales.append(scale)
            for value in values:
                normalized = value / scale if scale else 0.0
                code = min(
                    range(16),
                    key=lambda item: (abs(normalized - _e2m1(item)), item & 1),
                )
                quantized.append(_e2m1(code))
        for output in range(PROJECTION_OUTPUTS):
            total = 0.0
            packed_base = output * packed_stride
            scale_base = output * scale_stride
            for column in range(PROJECTION_K):
                pair = payload.packed_weights[packed_base + column // 2]
                nibble = pair >> 4 if column & 1 else pair & 15
                total += (
                    quantized[column]
                    * activation_scales[column // 16]
                    * _e2m1(nibble)
                    * _e4m3(payload.linear_scales[scale_base + column // 16])
                )
            result[batch * PROJECTION_OUTPUTS + output] = (
                total * payload.global_scales[output // PROJECTION_ROWS_PER_FAMILY]
            )
    return tuple(result)


def reference_output_projection(
    payload: FullQkvProjectionPayload, attention_bf16: bytes, output_rows: int = 4
) -> tuple[float, ...]:
    """Scalar oracle for selected rows of the fixed rank-local output GEMM."""

    if (
        not isinstance(payload, FullQkvProjectionPayload)
        or not isinstance(attention_bf16, bytes)
        or len(attention_bf16) != 16 * OUTPUT_PROJECTION_K * 2
        or isinstance(output_rows, bool)
        or not 1 <= output_rows <= 4
    ):
        raise ProjectionError("output projection reference inputs are invalid")
    result = []
    activation_global = OUTPUT_ACTIVATION_GLOBAL_SCALE
    for batch in range(16):
        quantized, activation_scales = [], []
        for group in range(OUTPUT_PROJECTION_K // 16):
            values = []
            for offset in range(16):
                index = batch * OUTPUT_PROJECTION_K + group * 16 + offset
                bf16 = struct.unpack_from("<H", attention_bf16, index * 2)[0]
                values.append(struct.unpack("<f", struct.pack("<I", bf16 << 16))[0])
            scale_bits = _float_to_e4m3(
                max(abs(value) for value in values) / (6.0 * activation_global)
            )
            scale = _e4m3(scale_bits)
            activation_scales.append(scale)
            for value in values:
                normalized = value / (scale * activation_global) if scale else 0.0
                code = min(range(16), key=lambda item: abs(normalized - _e2m1(item)))
                quantized.append(_e2m1(code))
        for output in range(output_rows):
            total = 0.0
            packed_base = output * OUTPUT_PROJECTION_K // 2
            for column in range(OUTPUT_PROJECTION_K):
                pair = payload.output_weight[packed_base + column // 2]
                weight = _e2m1(pair >> 4 if column & 1 else pair & 15)
                sf = payload.output_scale[
                    _sfb_offset(output, column // 16, OUTPUT_PROJECTION_K)
                ]
                total += quantized[column] * activation_scales[column // 16] * weight * _e4m3(sf)
            result.append(total * activation_global * payload.output_global_scale)
    return tuple(result)


def _sfb_offset(row: int, sf_index: int, k: int) -> int:
    k_tiles = ((k // 16) + 3) // 4
    row_tile, row_in_tile = divmod(row, 128)
    k_tile, sf_in_tile = divmod(sf_index, 4)
    return (
        (row_tile * k_tiles + k_tile) * 512
        + (row_in_tile % 32) * 16
        + (row_in_tile // 32) * 4
        + sf_in_tile
    )


def _sha256_range(path: Path, offset: int, length: int) -> str:
    with path.open("rb", buffering=0) as stream:
        return _sha256_fd_range(stream.fileno(), offset, length)


def _sha256_fd_range(fd: int, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    position = offset
    while remaining:
        block = os.pread(fd, min(4 * 1024 * 1024, remaining), position)
        if not block:
            raise ProjectionError("short rank slab chunk read")
        digest.update(block)
        position += len(block)
        remaining -= len(block)
    return digest.hexdigest()


def _e2m1(value: int) -> float:
    magnitude = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[value & 7]
    return -magnitude if value & 8 else magnitude


def _e4m3(value: int) -> float:
    sign = -1.0 if value & 0x80 else 1.0
    exponent = (value >> 3) & 15
    mantissa = value & 7
    if exponent == 0:
        return sign * mantissa * (2.0**-9)
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def _float_to_e4m3(value: float) -> int:
    sign = 0x80 if value < 0.0 else 0
    magnitude = abs(value)
    if magnitude >= 464.0:
        return sign | 0x7E
    return sign | min(
        range(0x7F),
        key=lambda bits: (abs(magnitude - _e4m3(bits)), bits & 1),
    )
