# SPDX-License-Identifier: Apache-2.0
"""One host-native lifetime for authenticated QSA c1 graph execution."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import struct
from pathlib import Path

from .native_qsa import (
    ARENA_FIELDS,
    NativeQsaGraphHandle,
    StateView,
    bind_slab_pointers,
)
from .qsa_m35_state_seed import (
    _Bundle,
    _Config,
    _Record,
    authenticate_qsa_c1_input_bundle,
)
from .qsa_weights import load_qsa_weights


ROPE_SHA256 = "6d1b5342ffb5f792f51c4599a8c3ac511b4a795877c2e21a113ac488195fda77"
GRAPH_STORAGE_BYTES = 812_544
ARENA_BYTES = {
    "qkv_packed": 1_280, "qkv_sfa": 20_480, "raw_main_qkv": 13_312,
    "index_projected_qk": 1_280, "main_query": 6_144,
    "attention_gate": 6_144, "index_query": 1_024,
    "index_logits": 262_144, "visible_blocks": 4,
    "selected_blocks": 2_048, "selected_tokens": 8_204,
    "attention_partial": 393_216, "attention_lse": 1_536,
    "attention_output": 6_144, "gated_attention": 6_144,
    "output_packed": 1_536, "output_sfa": 24_576,
    "projected_output": 5_120,
}
assert tuple(ARENA_BYTES) == ARENA_FIELDS


def arena_offsets() -> dict[str, tuple[int, int]]:
    cursor = 0
    result = {}
    for name, size in ARENA_BYTES.items():
        cursor = (cursor + 255) & ~255
        result[name] = (cursor, size)
        cursor += size
    if cursor > GRAPH_STORAGE_BYTES:
        raise QsaC1CompositeError("arena_layout")
    return result


def mismatch_summary(observed: bytes, expected: bytes) -> tuple[int, int]:
    if len(observed) != len(expected):
        raise QsaC1CompositeError("comparison_extent")
    offsets = [index for index, pair in enumerate(zip(observed, expected))
               if pair[0] != pair[1]]
    return len(offsets), offsets[0] if offsets else -1


class QsaC1CompositeError(RuntimeError):
    pass


class _DeviceBuffer:
    def __init__(self, pointer: int, size: int, dtype: str = "torch.uint8"):
        self.pointer = pointer
        self.size = size
        self.dtype = dtype
        self.device = "cuda:0"

    def data_ptr(self) -> int:
        return self.pointer

    def numel(self) -> int:
        return self.size


def execute_qsa_c1_composite(
    bundle: Path, seed_library: Path, graph_library: Path, rope_library: Path,
    artifact: Path, sidecar: Path,
) -> dict[str, object]:
    """Authenticate, seed, launch, fence, publish, then release one c1 graph."""

    stage = "bundle_authentication"
    records: list[dict[str, object]] = []
    allocations: list[ctypes.c_void_p] = []
    stream = ctypes.c_void_p()
    seed_owner = ctypes.c_void_p()
    rope_owner = ctypes.c_void_p()
    graph: NativeQsaGraphHandle | None = None
    cudart = None
    seed = None
    rope = None
    try:
        payloads = authenticate_qsa_c1_input_bundle(bundle)
        descriptor = load_qsa_weights(artifact, sidecar, 0, 3)
        records.append({"stage": stage, "outcome": "success"})

        cudart = ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.13")
        cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        cudart.cudaFree.argtypes = [ctypes.c_void_p]
        cudart.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        cudart.cudaDeviceSynchronize.argtypes = []

        def check(status: int) -> None:
            if status != 0:
                raise QsaC1CompositeError(stage)

        def allocate(size: int, *, zero: bool = False,
                     payload: bytes | None = None) -> _DeviceBuffer:
            pointer = ctypes.c_void_p()
            check(cudart.cudaMalloc(ctypes.byref(pointer), size))
            allocations.append(pointer)
            if zero:
                check(cudart.cudaMemset(pointer, 0, size))
            if payload is not None:
                source = ctypes.create_string_buffer(payload)
                check(cudart.cudaMemcpy(pointer, source, len(payload), 1))
            return _DeviceBuffer(int(pointer.value), size)

        def upload_at(target: _DeviceBuffer, offset: int, payload: bytes) -> None:
            source = ctypes.create_string_buffer(payload)
            check(cudart.cudaMemcpy(ctypes.c_void_p(target.pointer + offset),
                                    source, len(payload), 1))

        def download(pointer: int, size: int) -> bytes:
            target = ctypes.create_string_buffer(size)
            check(cudart.cudaMemcpy(target, ctypes.c_void_p(pointer), size, 2))
            return target.raw

        stage = "cuda_ownership"
        check(cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), 1))
        target_slab = allocate(descriptor.slab_bytes)
        for component in descriptor.components:
            if component.storage != "base-slab":
                continue
            with descriptor.slab_path.open("rb") as source:
                source.seek(component.offset)
                data = source.read(component.length)
            if len(data) != component.length:
                raise QsaC1CompositeError(stage)
            upload_at(target_slab, component.offset, data)
        sidecar_payload = descriptor.indexer_sidecar_path.read_bytes()
        sidecar_slab = allocate(len(sidecar_payload), payload=sidecar_payload)

        rope = ctypes.CDLL(str(rope_library))
        rope.qwen38_layer3_rope_c1_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        rope.qwen38_layer3_rope_c1_view.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        rope.qwen38_layer3_rope_c1_wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        rope.qwen38_layer3_rope_c1_destroy.argtypes = [ctypes.c_void_p]
        if rope.qwen38_layer3_rope_c1_create(0, 0, 3, ctypes.byref(rope_owner)):
            raise QsaC1CompositeError(stage)
        rope_pointer, rope_event = ctypes.c_void_p(), ctypes.c_void_p()
        rope_rows, rope_columns = ctypes.c_int(), ctypes.c_int()
        if rope.qwen38_layer3_rope_c1_view(
            rope_owner, ctypes.byref(rope_pointer), ctypes.byref(rope_event),
            ctypes.byref(rope_rows), ctypes.byref(rope_columns),
        ) or (rope_rows.value, rope_columns.value) != (36, 64):
            raise QsaC1CompositeError(stage)
        if rope.qwen38_layer3_rope_c1_wait(rope_owner, stream):
            raise QsaC1CompositeError(stage)

        arena_storage = allocate(GRAPH_STORAGE_BYTES, zero=True)
        arena = {
            name: _DeviceBuffer(arena_storage.pointer + offset, size)
            for name, (offset, size) in arena_offsets().items()
        }
        pointers = bind_slab_pointers(
            descriptor, target_slab, sidecar_slab,
            _DeviceBuffer(int(rope_pointer.value), 36 * 64, "torch.bfloat16"),
            arena,
        )
        records.append({"stage": stage, "outcome": "success"})

        stage = "state_ownership"
        inputs = {
            name: allocate(len(payloads[name]), payload=payloads[name])
            for name in (
                "written_key", "written_value", "main_slots", "raw_state",
                "raw_state_slots", "compressed_state", "compressed_state_slots",
            )
        }
        main_key = allocate(1600 * 256 * 2, zero=True)
        main_value = allocate(1600 * 256 * 2, zero=True)
        raw = allocate(8 * 140 * 2, zero=True)
        compressed = allocate(400 * 128 * 2, zero=True)
        positions, logical = allocate(8, zero=True), allocate(8, zero=True)
        main_slot = allocate(4, zero=True)
        raw_slot = allocate(4, zero=True)
        compressed_slot = allocate(4, zero=True)
        main_table = allocate(164 * 4, zero=True)
        compressed_table = allocate(164 * 4, zero=True)
        raw_table = allocate(4, zero=True)
        query_start = allocate(4, zero=True)
        sequence = allocate(4, zero=True)
        token = allocate(4, zero=True)
        work = allocate(4, zero=True)
        state = StateView(
            main_key.pointer, main_value.pointer, raw.pointer, compressed.pointer,
            positions.pointer, main_slot.pointer, main_table.pointer,
            raw_slot.pointer, raw_table.pointer, compressed_slot.pointer,
            compressed_table.pointer, query_start.pointer, logical.pointer,
            sequence.pointer, token.pointer, work.pointer,
            1, 1, 1, 1, 0, 3, False, 0, 0, 1, 1,
        )
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, _Record)
        callback = callback_type(lambda _context, record: records.append({
            "stage": "state_seed", "operation": int(record.operation),
            "outcome": int(record.outcome), "success": bool(record.success),
        }))
        seed = ctypes.CDLL(str(seed_library))
        seed.qwen38_qsa_m35_state_seed_create.argtypes = [
            ctypes.POINTER(_Config), ctypes.POINTER(StateView),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        seed.qwen38_qsa_m35_state_seed_launch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Bundle), ctypes.c_void_p,
        ]
        seed.qwen38_qsa_m35_state_seed_destroy.argtypes = [ctypes.c_void_p]
        config = _Config(0, 0, 3, 35, ctypes.cast(callback, ctypes.c_void_p), None)
        if seed.qwen38_qsa_m35_state_seed_create(
            ctypes.byref(config), ctypes.byref(state), ctypes.byref(seed_owner)
        ):
            raise QsaC1CompositeError(stage)
        publication = _Bundle(
            inputs["written_key"].pointer, inputs["written_value"].pointer,
            inputs["main_slots"].pointer,
            struct.unpack("<f", payloads["k_scale"])[0],
            struct.unpack("<f", payloads["v_scale"])[0],
            inputs["raw_state"].pointer, inputs["raw_state_slots"].pointer,
            inputs["compressed_state"].pointer,
            inputs["compressed_state_slots"].pointer,
        )
        if seed.qwen38_qsa_m35_state_seed_launch(
            seed_owner, ctypes.byref(publication), stream
        ):
            raise QsaC1CompositeError(stage)
        records.append({"stage": stage, "outcome": "success"})

        stage = "graph_construction"
        graph = NativeQsaGraphHandle(graph_library, descriptor, pointers, 0)
        records.append({"stage": stage, "outcome": "success"})
        hidden = allocate(len(payloads["row35_hidden"]),
                          payload=payloads["row35_hidden"])
        hidden.dtype = "torch.bfloat16"
        stage = "graph_launch"
        output_pointer = graph.launch(hidden, state, 1, int(stream.value))
        records.append({"stage": stage, "outcome": "success"})
        stage = "graph_fence"
        check(cudart.cudaDeviceSynchronize())
        records.append({"stage": stage, "outcome": "success"})
        projected = download(output_pointer, ARENA_BYTES["projected_output"])
        index_query = download(arena["index_query"].pointer,
                               ARENA_BYTES["index_query"])
        selected = download(arena["selected_tokens"].pointer,
                            ARENA_BYTES["selected_tokens"])
        attention = download(arena["attention_output"].pointer,
                             ARENA_BYTES["attention_output"])
        index_mismatch_bytes, index_first_mismatch = mismatch_summary(
            index_query, payloads["row35_index_query"]
        )
        records.append({"stage": "publication", "outcome": "success"})
        return {
            "status": "success", "artifact_key": bundle.name,
            "rank": 0, "layer": 3, "generation": 1,
            "rope_payload_sha256": ROPE_SHA256,
            "projected_output_bf16_sha256": hashlib.sha256(projected).hexdigest(),
            "index_query_bf16_sha256": hashlib.sha256(index_query).hexdigest(),
            "reference_index_query_bf16_sha256": hashlib.sha256(
                payloads["row35_index_query"]
            ).hexdigest(),
            "index_query_matches_reference":
                index_query == payloads["row35_index_query"],
            "index_query_mismatch_bytes": index_mismatch_bytes,
            "index_query_first_mismatch_byte": index_first_mismatch,
            "selected_tokens_sha256": hashlib.sha256(selected).hexdigest(),
            "attention_output_bf16_sha256": hashlib.sha256(attention).hexdigest(),
            "telemetry": records,
        }
    except Exception as exc:
        records.append({"stage": stage, "outcome": "failure"})
        return {"status": "failure", "stage": stage,
                "failure_class": type(exc).__name__, "telemetry": records}
    finally:
        if graph is not None:
            graph.close()
        if seed_owner.value and seed is not None:
            seed.qwen38_qsa_m35_state_seed_destroy(seed_owner)
        if rope_owner.value and rope is not None:
            rope.qwen38_layer3_rope_c1_destroy(rope_owner)
        if stream.value and cudart is not None:
            cudart.cudaStreamDestroy(stream)
        if cudart is not None:
            for pointer in reversed(allocations):
                cudart.cudaFree(pointer)


__all__ = ["ARENA_BYTES", "GRAPH_STORAGE_BYTES", "QsaC1CompositeError",
           "arena_offsets", "execute_qsa_c1_composite", "mismatch_summary"]
