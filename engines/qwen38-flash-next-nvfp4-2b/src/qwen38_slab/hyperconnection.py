"""Authenticated layer-3 rank-0 HyperConnection weights."""

from __future__ import annotations

import hashlib
import ctypes
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .device_decode import DEFAULT_CUDART, DeviceDecodeError, _Cuda13Api
from .projection import RankSlabProjection, load_rank0_layer3_projection

HC_STREAMS = 4
HC_HIDDEN = 2_560
HC_WIDTH = HC_STREAMS * HC_HIDDEN
HC_LOW_RANK = 320
HC_SCHEMA = "qwen3.8-flash-next:tp2:rank0:layer3:hyperconnection:v1"


class HyperConnectionError(RuntimeError):
    """Authenticated HyperConnection contract failure."""


@dataclass(frozen=True)
class HyperConnectionFamily:
    norm_bf16: bytes
    down_bf16: bytes
    injection_bf16: bytes
    up_bf16: bytes


@dataclass(frozen=True)
class HyperConnectionPayload:
    schema: str
    descriptor: RankSlabProjection
    attention: HyperConnectionFamily
    mlp: HyperConnectionFamily


def load_layer3_hyperconnection(artifact: Path) -> HyperConnectionPayload:
    """Authenticate and read both HC boundaries around layer-3 attention."""

    descriptor = load_rank0_layer3_projection(artifact)
    try:
        manifest = json.loads((artifact / "manifest.json").read_bytes())
        slab = manifest["slabs"]["rank0-target"]
        entries = {entry["name"]: entry for entry in slab["entries"]}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise HyperConnectionError("HyperConnection manifest inventory is invalid") from exc

    contracts = {
        "hc_norm.weight": ([HC_WIDTH], HC_WIDTH * 2),
        "input_mix_weight_down.weight": ([HC_LOW_RANK, HC_WIDTH], HC_LOW_RANK * HC_WIDTH * 2),
        "block_inject_weight.weight": ([HC_STREAMS, HC_WIDTH], HC_STREAMS * HC_WIDTH * 2),
        "input_mix_weight_up.weight": ([HC_WIDTH, HC_LOW_RANK], HC_WIDTH * HC_LOW_RANK * 2),
    }
    selected: dict[str, dict] = {}
    for boundary in ("attn_hyper_connection", "mlp_hyper_connection"):
        for suffix, (shape, length) in contracts.items():
            name = f"model.language_model.layers.3.{boundary}.{suffix}"
            entry = entries.get(name)
            if (
                not isinstance(entry, dict)
                or entry.get("shape") != shape
                or entry.get("dtype") != "BF16"
                or entry.get("layout") != "checkpoint"
                or entry.get("length_bytes") != length
                or not isinstance(entry.get("offset_bytes"), int)
                or entry["offset_bytes"] % 256
            ):
                raise HyperConnectionError(f"HyperConnection contract drift: {name}")
            selected[name] = entry

    chunks = {}
    for entry in selected.values():
        matches = [
            chunk for chunk in slab["chunks"]
            if entry["offset_bytes"] < chunk["offset_bytes"] + chunk["length_bytes"]
            and entry["offset_bytes"] + entry["length_bytes"] > chunk["offset_bytes"]
        ]
        if len(matches) != 1:
            raise HyperConnectionError("HyperConnection component crosses chunk boundary")
        chunk = matches[0]
        chunks[chunk["offset_bytes"]] = chunk

    fd = os.open(descriptor.slab_path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size != descriptor.slab_bytes:
            raise HyperConnectionError("rank slab changed after authentication")
        for chunk in chunks.values():
            digest = hashlib.sha256(
                os.pread(fd, chunk["length_bytes"], chunk["offset_bytes"])
            ).hexdigest()
            if digest != chunk["sha256"]:
                raise HyperConnectionError("HyperConnection chunk digest mismatch")

        def family(boundary: str) -> HyperConnectionFamily:
            blobs = []
            for suffix in contracts:
                entry = selected[
                    f"model.language_model.layers.3.{boundary}.{suffix}"
                ]
                blob = os.pread(fd, entry["length_bytes"], entry["offset_bytes"])
                if len(blob) != entry["length_bytes"]:
                    raise HyperConnectionError("short HyperConnection weight read")
                blobs.append(blob)
            return HyperConnectionFamily(*blobs)

        return HyperConnectionPayload(
            HC_SCHEMA, descriptor, family("attn_hyper_connection"),
            family("mlp_hyper_connection")
        )
    finally:
        os.close(fd)


class CudaHyperConnectionRuntime:
    """Graph-owned exact layer-3 HC mix and combine boundaries."""

    _BUCKETS = (1, 2, 4, 8, 16)

    def __init__(
        self,
        payload: HyperConnectionPayload,
        library: Path,
        stream_pointer: int,
        device: int = 0,
        cudart: Path = DEFAULT_CUDART,
    ):
        if (
            not isinstance(payload, HyperConnectionPayload)
            or payload.schema != HC_SCHEMA
            or not isinstance(library, Path)
            or isinstance(stream_pointer, bool)
            or not isinstance(stream_pointer, int)
            or stream_pointer <= 0
        ):
            raise HyperConnectionError("CUDA HyperConnection ABI is invalid")
        self._api = _Cuda13Api(cudart)
        try:
            self._native = ctypes.CDLL(str(library))
        except OSError as exc:
            raise HyperConnectionError("cannot load HyperConnection library") from exc
        self._native.qwen38_hc_last_error.restype = ctypes.c_char_p
        for name in (
            "qwen38_hc_create", "qwen38_hc_mix",
            "qwen38_hc_combine_and_mix", "qwen38_hc_destroy",
        ):
            getattr(self._native, name).restype = ctypes.c_int
        self._stream = ctypes.c_void_p(stream_pointer)
        self._rank_slab_identity = (
            payload.descriptor.artifact_key,
            payload.descriptor.rank,
            payload.descriptor.layer,
        )
        self._plan = ctypes.c_void_p()
        self._device: dict[str, ctypes.c_void_p] = {}
        self._graphs: dict[tuple[str, int], ctypes.c_void_p] = {}
        self._execs: dict[tuple[str, int], ctypes.c_void_p] = {}
        self._nodes: dict[tuple[str, int], int] = {}
        self._closed = False
        try:
            self._api.call("cudaSetDevice", device)
            weight_pointers = []
            for boundary in (payload.attention, payload.mlp):
                for blob in (
                    boundary.norm_bf16, boundary.down_bf16,
                    boundary.injection_bf16, boundary.up_bf16,
                ):
                    name = f"upload_weight_{len(weight_pointers)}"
                    self._allocate_named(name, len(blob))
                    pointer = self._device[name]
                    self._api.call(
                        "cudaMemcpy", pointer, ctypes.c_char_p(blob), len(blob), 1
                    )
                    weight_pointers.append(pointer)
            self._allocate_named("hidden", 16 * HC_WIDTH * 2)
            self._allocate_named("block_input", 16 * HC_HIDDEN * 2)
            self._allocate_named("injection", 16 * HC_STREAMS * 2)
            self._allocate_named("reduced", 16 * HC_HIDDEN * 4)
            self._allocate_named("updated_hidden", 16 * HC_WIDTH * 2)
            self._allocate_named("next_block_input", 16 * HC_HIDDEN * 2)
            self._allocate_named("next_injection", 16 * HC_STREAMS * 2)
            hidden = bytearray()
            for index in range(16 * HC_WIDTH):
                value = ((index % 29) - 14) / 32.0
                hidden.extend(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16))
            self._api.call(
                "cudaMemcpy", self._device["hidden"], ctypes.c_char_p(bytes(hidden)),
                len(hidden), 1,
            )
            self._native_call(
                "qwen38_hc_create", device, *weight_pointers,
                ctypes.byref(self._plan),
            )
            # Plan construction copies authenticated weights to graph-owned
            # storage. Release upload staging so K0 residency pays one copy.
            for index in range(len(weight_pointers)):
                pointer = self._device.pop(f"upload_weight_{index}")
                self._api.call("cudaFree", pointer)
            for m in self._BUCKETS:
                self._native_call(
                    "qwen38_hc_mix", self._plan, self._device["hidden"],
                    self._device["block_input"], self._device["injection"],
                    m, self._stream,
                )
                self._native_call(
                    "qwen38_hc_combine_and_mix", self._plan,
                    self._device["hidden"], self._device["reduced"],
                    self._device["injection"], self._device["updated_hidden"],
                    self._device["next_block_input"],
                    self._device["next_injection"], m, self._stream,
                )
                self._api.call("cudaStreamSynchronize", self._stream)
                self._capture("mix", m)
                self._capture("combine", m)
        except BaseException:
            self._close_noexcept()
            raise

    def _allocate_named(self, name: str, size: int) -> None:
        pointer = ctypes.c_void_p()
        self._api.call("cudaMalloc", ctypes.byref(pointer), size)
        self._device[name] = pointer

    def _capture(self, operation: str, m: int) -> None:
        graph, executable = ctypes.c_void_p(), ctypes.c_void_p()
        self._api.call("cudaStreamBeginCapture", self._stream, 1)
        if operation == "mix":
            self._native_call(
                "qwen38_hc_mix", self._plan, self._device["hidden"],
                self._device["block_input"], self._device["injection"],
                m, self._stream,
            )
        else:
            self._native_call(
                "qwen38_hc_combine_and_mix", self._plan,
                self._device["hidden"], self._device["reduced"],
                self._device["injection"], self._device["updated_hidden"],
                self._device["next_block_input"], self._device["next_injection"],
                m, self._stream,
            )
        self._api.call("cudaStreamEndCapture", self._stream, ctypes.byref(graph))
        self._api.call("cudaGraphInstantiate", ctypes.byref(executable), graph, 0)
        nodes = ctypes.c_size_t()
        self._api.call("cudaGraphGetNodes", graph, None, ctypes.byref(nodes))
        self._graphs[(operation, m)] = graph
        self._execs[(operation, m)] = executable
        self._nodes[(operation, m)] = nodes.value

    def launch_mix(self, m: int) -> None:
        self._launch("mix", m)

    def launch_combine(self, m: int) -> None:
        self._launch("combine", m)

    def reference_mix(self, m: int) -> None:
        if self._closed or m not in self._BUCKETS:
            raise HyperConnectionError("HyperConnection reference bucket is invalid")
        self._native_call(
            "qwen38_hc_mix", self._plan, self._device["hidden"],
            self._device["block_input"], self._device["injection"], m,
            self._stream,
        )

    def reference_combine(self, m: int) -> None:
        if self._closed or m not in self._BUCKETS:
            raise HyperConnectionError("HyperConnection reference bucket is invalid")
        self._native_call(
            "qwen38_hc_combine_and_mix", self._plan, self._device["hidden"],
            self._device["reduced"], self._device["injection"],
            self._device["updated_hidden"], self._device["next_block_input"],
            self._device["next_injection"], m, self._stream,
        )

    def _launch(self, operation: str, m: int) -> None:
        if self._closed or m not in self._BUCKETS:
            raise HyperConnectionError("HyperConnection graph bucket is invalid")
        self._api.call("cudaGraphLaunch", self._execs[(operation, m)], self._stream)

    def synchronize(self) -> None:
        if self._closed:
            raise HyperConnectionError("HyperConnection runtime is closed")
        self._api.call("cudaStreamSynchronize", self._stream)

    @property
    def pointers(self) -> Mapping[str, int]:
        if self._closed:
            raise HyperConnectionError("HyperConnection runtime is closed")
        return MappingProxyType({
            name: int(pointer.value) for name, pointer in self._device.items()
            if not name.startswith("allocation_")
        })

    @property
    def rank_slab_identity(self) -> tuple[str, int, int]:
        if self._closed:
            raise HyperConnectionError("HyperConnection runtime is closed")
        return self._rank_slab_identity

    @property
    def graph_nodes(self) -> Mapping[tuple[str, int], int]:
        return MappingProxyType(dict(self._nodes))

    def hashes(self, m: int, names: tuple[str, ...] | None = None) -> Mapping[str, str]:
        if m not in self._BUCKETS:
            raise HyperConnectionError("HyperConnection hash bucket is invalid")
        sizes = {
            "block_input": m * HC_HIDDEN * 2,
            "injection": m * HC_STREAMS * 2,
            "updated_hidden": m * HC_WIDTH * 2,
            "next_block_input": m * HC_HIDDEN * 2,
            "next_injection": m * HC_STREAMS * 2,
            "reduced": m * HC_HIDDEN * 4,
        }
        selected = tuple(sizes) if names is None else names
        if not selected or len(set(selected)) != len(selected) or any(
            name not in sizes for name in selected
        ):
            raise HyperConnectionError("HyperConnection hash selection is invalid")
        result = {}
        for name in selected:
            size = sizes[name]
            output = ctypes.create_string_buffer(size)
            self._api.call("cudaMemcpy", ctypes.addressof(output), self._device[name], size, 2)
            result[name] = hashlib.sha256(output.raw).hexdigest()
        return MappingProxyType(result)

    def benchmark(self, m: int, iterations: int = 40) -> Mapping[str, float]:
        if m not in self._BUCKETS or not 5 <= iterations <= 1000:
            raise HyperConnectionError("HyperConnection benchmark contract drift")

        def measured(operation: str) -> float:
            start, end = ctypes.c_void_p(), ctypes.c_void_p()
            self._api.call("cudaEventCreate", ctypes.byref(start))
            self._api.call("cudaEventCreate", ctypes.byref(end))
            try:
                for _ in range(3): self._launch(operation, m)
                self._api.call("cudaStreamSynchronize", self._stream)
                self._api.call("cudaEventRecord", start, self._stream)
                for _ in range(iterations): self._launch(operation, m)
                self._api.call("cudaEventRecord", end, self._stream)
                self._api.call("cudaEventSynchronize", end)
                elapsed = ctypes.c_float()
                self._api.call("cudaEventElapsedTime", ctypes.byref(elapsed), start, end)
                return elapsed.value / iterations
            finally:
                self._api.call("cudaEventDestroy", end)
                self._api.call("cudaEventDestroy", start)

        return MappingProxyType({"mix_ms": measured("mix"),
                                 "combine_mix_ms": measured("combine")})

    def _native_call(self, name: str, *args: object) -> None:
        if getattr(self._native, name)(*args) != 0:
            raw = self._native.qwen38_hc_last_error()
            detail = raw.decode("utf-8", "replace") if raw else "unknown failure"
            raise HyperConnectionError(f"{name}: {detail}")

    def close(self) -> None:
        errors = self._close_noexcept()
        if errors:
            raise HyperConnectionError("HyperConnection cleanup failed: " + "; ".join(errors))

    def _close_noexcept(self) -> list[str]:
        if self._closed:
            return []
        errors = []
        for executable in self._execs.values():
            try: self._api.call("cudaGraphExecDestroy", executable)
            except Exception as exc: errors.append(str(exc))
        for graph in self._graphs.values():
            try: self._api.call("cudaGraphDestroy", graph)
            except Exception as exc: errors.append(str(exc))
        if self._plan.value:
            try: self._native_call("qwen38_hc_destroy", self._plan)
            except Exception as exc: errors.append(str(exc))
        for pointer in reversed(tuple(self._device.values())):
            try: self._api.call("cudaFree", pointer)
            except Exception as exc: errors.append(str(exc))
        self._closed = True
        return errors

    def __enter__(self) -> "CudaHyperConnectionRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
