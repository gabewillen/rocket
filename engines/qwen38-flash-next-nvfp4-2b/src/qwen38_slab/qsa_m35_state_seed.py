# SPDX-License-Identifier: Apache-2.0
"""Authenticated QSA M35 cache publication into the host-native c1 layout."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
import struct
from pathlib import Path


SCHEMA = "rocket.qwen38.qsa-prefill-boundary.v2"
ORACLE_MANIFEST_SHA256 = (
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b"
)
IMPLEMENTATION = "vllm:8e685d198:qwen38-qsa-prefill"
ROWS = 35

_LAYOUTS = {
    "positions": ("int64", (3, 35), 8),
    "query": ("bfloat16", (35, 12, 256), 2),
    "key": ("bfloat16", (35, 1, 256), 2),
    "value": ("bfloat16", (35, 1, 256), 2),
    "selected": ("int32", (35, 2051), 4),
    "main_slots": ("int64", (35,), 8),
    "raw_slots": ("int64", (35,), 8),
    "compressed_slots": ("int64", (35,), 8),
    "logical_positions": ("int64", (35,), 8),
    "token_to_req": ("int32", (35,), 4),
    # Pinned vLLM exposes FP8 KV cache storage as authenticated uint8 codes;
    # k_scale/v_scale carry the corresponding dequantization factors.
    "written_key": ("uint8", (35, 1, 256), 1),
    "written_value": ("uint8", (35, 1, 256), 1),
    "k_scale": ("float32", (1,), 4),
    "v_scale": ("float32", (1,), 4),
    "raw_state_slots": ("int64", (4,), 8),
    "raw_state": ("bfloat16", (4, 1, 140), 2),
    "compressed_state_slots": ("int64", (8,), 8),
    "compressed_state": ("bfloat16", (8, 1, 128), 2),
}


class QsaM35StateSeedError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def authenticate_qsa_m35_state_bundle(bundle: Path) -> dict[str, bytes]:
    try:
        manifest = json.loads((bundle / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QsaM35StateSeedError("QSA M35 state bundle unavailable") from exc
    if not isinstance(manifest, dict):
        raise QsaM35StateSeedError("QSA M35 state manifest changed")
    authenticated = dict(manifest)
    claimed = authenticated.pop("artifact_key", None)
    observed = hashlib.sha256(_canonical(authenticated)).hexdigest()
    if (
        claimed != observed
        or bundle.name != observed
        or manifest.get("schema") != SCHEMA
        or manifest.get("oracle_manifest_sha256") != ORACLE_MANIFEST_SHA256
        or manifest.get("implementation") != IMPLEMENTATION
        or manifest.get("rank") != 0
        or manifest.get("layer") != 3
        or manifest.get("rows") != ROWS
        or manifest.get("generation_index") != 0
        or not isinstance(manifest.get("tensors"), list)
    ):
        raise QsaM35StateSeedError("QSA M35 state identity changed")
    entries = {
        entry.get("name"): entry
        for entry in manifest["tensors"]
        if isinstance(entry, dict)
    }
    if set(entries) != set(_LAYOUTS) or len(entries) != len(manifest["tensors"]):
        raise QsaM35StateSeedError("QSA M35 state inventory changed")
    result: dict[str, bytes] = {}
    for name, (dtype, shape, element_bytes) in _LAYOUTS.items():
        entry = entries[name]
        filename = entry.get("file")
        expected_bytes = math.prod(shape) * element_bytes
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or entry.get("dtype") != dtype
            or tuple(entry.get("shape", ())) != shape
            or entry.get("bytes") != expected_bytes
        ):
            raise QsaM35StateSeedError("QSA M35 state layout changed")
        path = bundle / filename
        try:
            if path.is_symlink():
                raise QsaM35StateSeedError("QSA M35 state path changed")
            payload = path.read_bytes()
        except OSError as exc:
            raise QsaM35StateSeedError("QSA M35 state payload unavailable") from exc
        if (
            len(payload) != expected_bytes
            or hashlib.sha256(payload).hexdigest() != entry.get("sha256")
        ):
            raise QsaM35StateSeedError("QSA M35 state payload changed")
        result[name] = payload
    return result


class _StateView(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in (
            "main_key_cache", "main_value_cache", "raw_key_cache",
            "compressed_key_cache", "positions", "main_slot_mapping",
            "main_block_table", "raw_slot_mapping", "raw_block_table",
            "compressed_slot_mapping", "compressed_block_table",
            "query_start_locations", "logical_positions", "sequence_lengths",
            "token_to_request", "compression_work",
        )
    ] + [
        ("main_blocks", ctypes.c_int),
        ("compressed_blocks", ctypes.c_int),
        ("compression_work_items", ctypes.c_int),
        ("rows", ctypes.c_int),
        ("rank", ctypes.c_int),
        ("layer", ctypes.c_int),
        ("uses_mrope", ctypes.c_bool),
        ("main_kv_dtype", ctypes.c_uint8),
        ("side_cache_dtype", ctypes.c_uint8),
        ("generation", ctypes.c_uint64),
        ("expected_generation", ctypes.c_uint64),
    ]


class _Record(ctypes.Structure):
    _fields_ = [
        ("operation", ctypes.c_uint8), ("outcome", ctypes.c_uint8),
        ("rank", ctypes.c_int8), ("layer", ctypes.c_int8),
        ("rows", ctypes.c_uint8), ("success", ctypes.c_bool),
    ]


class _Config(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_int), ("rank", ctypes.c_int),
        ("layer", ctypes.c_int), ("rows", ctypes.c_int),
        ("publish", ctypes.c_void_p), ("publish_context", ctypes.c_void_p),
    ]


class _Bundle(ctypes.Structure):
    _fields_ = [
        ("main_key_fp8", ctypes.c_void_p),
        ("main_value_fp8", ctypes.c_void_p),
        ("main_slots", ctypes.c_void_p),
        ("key_scale", ctypes.c_float),
        ("value_scale", ctypes.c_float),
        ("raw_state", ctypes.c_void_p),
        ("raw_state_slots", ctypes.c_void_p),
        ("compressed_state", ctypes.c_void_p),
        ("compressed_state_slots", ctypes.c_void_p),
    ]


def _fp8_codes_to_bf16(payload: bytes, scale: float) -> bytes:
    result = bytearray(2 * len(payload))
    for index, code in enumerate(payload):
        sign = 0x80000000 if code & 0x80 else 0
        exponent = (code >> 3) & 0xF
        mantissa = code & 7
        if exponent == 0:
            value = (-1.0 if sign else 1.0) * mantissa / 512.0
        else:
            value = struct.unpack(
                "<f", struct.pack("<I", sign | ((exponent + 120) << 23) | (mantissa << 20))
            )[0]
        bits = struct.unpack("<I", struct.pack("<f", value * scale))[0]
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        struct.pack_into("<H", result, 2 * index, rounded & 0xFFFF)
    return bytes(result)


def execute_qsa_m35_state_seed(bundle: Path, library: Path) -> dict[str, object]:
    """Run one authenticated, caller-owned host-native state publication."""
    payloads = authenticate_qsa_m35_state_bundle(bundle)
    cudart_name = ctypes.util.find_library("cudart") or "libcudart.so.13"
    cudart = ctypes.CDLL(cudart_name)
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    allocations: list[ctypes.c_void_p] = []

    def check(status: int, operation: str) -> None:
        if status != 0:
            raise QsaM35StateSeedError(f"host-native CUDA {operation} failed")

    def allocate(size: int, payload: bytes | None = None) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        check(cudart.cudaMalloc(ctypes.byref(pointer), size), "allocation")
        allocations.append(pointer)
        if payload is None:
            check(cudart.cudaMemset(pointer, 0, size), "initialization")
        else:
            source = ctypes.create_string_buffer(payload)
            check(cudart.cudaMemcpy(pointer, ctypes.cast(source, ctypes.c_void_p), size, 1), "upload")
        return pointer

    def download(pointer: ctypes.c_void_p, size: int) -> bytes:
        target = ctypes.create_string_buffer(size)
        check(cudart.cudaMemcpy(ctypes.cast(target, ctypes.c_void_p), pointer, size, 2), "download")
        return target.raw

    inputs = {name: allocate(len(payloads[name]), payloads[name]) for name in (
        "written_key", "written_value", "main_slots", "raw_state",
        "raw_state_slots", "compressed_state", "compressed_state_slots",
    )}
    main_key = allocate(1600 * 256 * 2)
    main_value = allocate(1600 * 256 * 2)
    raw = allocate(8 * 140 * 2)
    compressed = allocate(400 * 128 * 2)
    positions, logical = allocate(8), allocate(8)
    main_slot, raw_slot, compressed_slot = allocate(4), allocate(4), allocate(4)
    main_table, compressed_table = allocate(164 * 4), allocate(164 * 4)
    raw_table, query_start, sequence, token, work = (allocate(4) for _ in range(5))
    state = _StateView(
        main_key, main_value, raw, compressed, positions, main_slot, main_table,
        raw_slot, raw_table, compressed_slot, compressed_table, query_start,
        logical, sequence, token, work, 1, 1, 1, 1, 0, 3, False, 0, 0, 0, 0,
    )
    k_scale = struct.unpack("<f", payloads["k_scale"])[0]
    v_scale = struct.unpack("<f", payloads["v_scale"])[0]
    records: list[tuple[int, int, int, int, int, bool]] = []
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, _Record)
    callback = callback_type(
        lambda _context, record: records.append((
            record.operation, record.outcome, record.rank, record.layer,
            record.rows, record.success,
        ))
    )
    native = ctypes.CDLL(str(library))
    native.qwen38_qsa_m35_state_seed_create.argtypes = [
        ctypes.POINTER(_Config), ctypes.POINTER(_StateView),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    native.qwen38_qsa_m35_state_seed_create.restype = ctypes.c_int
    native.qwen38_qsa_m35_state_seed_launch.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_Bundle), ctypes.c_void_p,
    ]
    native.qwen38_qsa_m35_state_seed_launch.restype = ctypes.c_int
    native.qwen38_qsa_m35_state_seed_destroy.argtypes = [ctypes.c_void_p]
    native.qwen38_qsa_m35_state_seed_destroy.restype = ctypes.c_int
    owner = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    config = _Config(0, 0, 3, 35, ctypes.cast(callback, ctypes.c_void_p), None)
    try:
        check(cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), 1), "stream creation")
        if native.qwen38_qsa_m35_state_seed_create(
            ctypes.byref(config), ctypes.byref(state), ctypes.byref(owner)
        ) != 0:
            raise QsaM35StateSeedError("native QSA state seed construction failed")
        publication = _Bundle(
            inputs["written_key"], inputs["written_value"], inputs["main_slots"],
            k_scale, v_scale, inputs["raw_state"], inputs["raw_state_slots"],
            inputs["compressed_state"], inputs["compressed_state_slots"],
        )
        if native.qwen38_qsa_m35_state_seed_launch(
            owner, ctypes.byref(publication), stream
        ) != 0:
            raise QsaM35StateSeedError("native QSA state seed launch failed")
        check(cudart.cudaDeviceSynchronize(), "synchronization")
        key_bytes = download(main_key, 35 * 256 * 2)
        value_bytes = download(main_value, 35 * 256 * 2)
        raw_bytes = download(raw, 4 * 140 * 2)
        compressed_bytes = download(compressed, 8 * 128 * 2)
        metadata = [
            struct.unpack("<q", download(positions, 8))[0],
            struct.unpack("<i", download(main_slot, 4))[0],
            struct.unpack("<i", download(raw_slot, 4))[0],
            struct.unpack("<i", download(compressed_slot, 4))[0],
            struct.unpack("<q", download(logical, 8))[0],
            struct.unpack("<i", download(sequence, 4))[0],
            struct.unpack("<i", download(token, 4))[0],
            struct.unpack("<i", download(work, 4))[0],
        ]
        tables = [
            struct.unpack("<i", download(main_table, 4))[0],
            struct.unpack("<i", download(raw_table, 4))[0],
            struct.unpack("<i", download(compressed_table, 4))[0],
        ]
    finally:
        if owner.value:
            native.qwen38_qsa_m35_state_seed_destroy(owner)
        if stream.value:
            cudart.cudaStreamDestroy(stream)
        for pointer in reversed(allocations):
            cudart.cudaFree(pointer)
    expected_key = _fp8_codes_to_bf16(payloads["written_key"], k_scale)
    expected_value = _fp8_codes_to_bf16(payloads["written_value"], v_scale)
    if (
        key_bytes != expected_key or value_bytes != expected_value
        or raw_bytes != payloads["raw_state"]
        or compressed_bytes != payloads["compressed_state"]
        or metadata != [35, 35, 3, 8, 35, 36, 0, 1]
        or tables != [0, 0, 0]
    ):
        raise QsaM35StateSeedError("native QSA state publication changed")
    return {
        "status": "success", "artifact_key": bundle.name,
        "rank": 0, "layer": 3, "rows_seeded": 35,
        "main_key_bf16_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "main_value_bf16_sha256": hashlib.sha256(value_bytes).hexdigest(),
        "raw_state_bf16_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "compressed_state_bf16_sha256": hashlib.sha256(compressed_bytes).hexdigest(),
        "next_row": 35, "telemetry": records,
        "c1_execution": "blocked_missing_authenticated_row35_hidden_and_index_query",
    }
