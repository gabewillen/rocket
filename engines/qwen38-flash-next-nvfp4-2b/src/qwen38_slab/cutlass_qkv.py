"""Captured SM121 tensor-core projection for the fixed Qwen3.8 c16 ABI."""

from __future__ import annotations

import ctypes
import math
import struct
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .device_decode import DEFAULT_CUDART, DeviceDecodeError, _Cuda13Api
from .projection import (
    FullQkvProjectionPayload,
    PROJECTION_FAMILY_ROWS,
    PROJECTION_K,
    PROJECTION_SCHEMA,
    OUTPUT_PROJECTION_K,
    OUTPUT_PROJECTION_N,
)

DEFAULT_CUTLASS_QKV = Path(__file__).resolve().parents[2] / "build/libqwen38_cutlass_qkv.so"
FULL_OUTPUTS = sum(PROJECTION_FAMILY_ROWS)
FULL_OUTPUT_BYTES = 16 * FULL_OUTPUTS * 2


class CutlassQkvRuntime:
    """Own one immutable production-width plan and one captured c16 graph."""

    def __init__(
        self,
        payload: FullQkvProjectionPayload,
        device: int = 0,
        library: Path = DEFAULT_CUTLASS_QKV,
        cudart: Path = DEFAULT_CUDART,
    ):
        if not isinstance(payload, FullQkvProjectionPayload):
            raise DeviceDecodeError("full QKV payload ABI is invalid")
        if not isinstance(library, Path) or not isinstance(cudart, Path):
            raise DeviceDecodeError("CUDA libraries must be explicit Paths")
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise DeviceDecodeError("CUDA device must be a nonnegative integer")
        expected_weights = tuple(
            rows * PROJECTION_K // 2 for rows in PROJECTION_FAMILY_ROWS
        )
        expected_scales = tuple(
            rows * PROJECTION_K // 16 for rows in PROJECTION_FAMILY_ROWS
        )
        if (
            payload.descriptor.schema != PROJECTION_SCHEMA
            or payload.descriptor.rank != 0
            or payload.descriptor.layer != 3
            or tuple(map(len, payload.packed_weights)) != expected_weights
            or tuple(map(len, payload.swizzled_scales)) != expected_scales
            or len(payload.global_scales) != 3
            or not all(
                math.isfinite(value) and value > 0.0
                for value in payload.global_scales
            )
            or len(payload.activations_bf16) != 16 * PROJECTION_K * 2
            or len(payload.output_weight) != OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 2
            or len(payload.output_scale) != OUTPUT_PROJECTION_N * OUTPUT_PROJECTION_K // 16
            or not math.isfinite(payload.output_global_scale)
            or payload.output_global_scale <= 0.0
        ):
            raise DeviceDecodeError("full QKV payload extent is invalid")
        try:
            self._native = ctypes.CDLL(str(library))
        except OSError as exc:
            raise DeviceDecodeError(f"cannot load fixed CUTLASS QKV library: {library}") from exc
        self._configure_native()
        self._api = _Cuda13Api(cudart)
        self._stream = ctypes.c_void_p()
        self._plan = ctypes.c_void_p()
        self._qsa_plan = ctypes.c_void_p()
        self._qsa_inputs = (ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p())
        self._qsa_input_bytes = (ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t())
        self._graph = ctypes.c_void_p()
        self._exec = ctypes.c_void_p()
        self._device_buffers: list[ctypes.c_void_p] = []
        self._output = ctypes.c_void_p()
        self._output_elements = ctypes.c_size_t()
        self._closed = False
        try:
            self._api.call("cudaSetDevice", device)
            self._api.call("cudaStreamCreateWithFlags", ctypes.byref(self._stream), 1)
            pointers = []
            for blob in (*payload.packed_weights, *payload.swizzled_scales):
                pointer = ctypes.c_void_p()
                self._api.call("cudaMalloc", ctypes.byref(pointer), len(blob))
                self._api.call("cudaMemcpy", pointer, ctypes.c_char_p(blob), len(blob), 1)
                self._device_buffers.append(pointer)
                pointers.append(pointer)
            output_pointers = []
            for blob in (payload.output_weight, payload.output_scale):
                pointer = ctypes.c_void_p()
                self._api.call("cudaMalloc", ctypes.byref(pointer), len(blob))
                self._api.call("cudaMemcpy", pointer, ctypes.c_char_p(blob), len(blob), 1)
                self._device_buffers.append(pointer)
                output_pointers.append(pointer)
            activation = ctypes.c_void_p()
            self._api.call("cudaMalloc", ctypes.byref(activation), len(payload.activations_bf16))
            self._api.call(
                "cudaMemcpy", activation, ctypes.c_char_p(payload.activations_bf16),
                len(payload.activations_bf16), 1,
            )
            self._device_buffers.append(activation)
            weights, scales = pointers[:3], pointers[3:]
            self._native_call(
                "qwen38_cutlass_qkv_create",
                weights[0], scales[0], ctypes.c_float(payload.global_scales[0]),
                weights[1], scales[1], ctypes.c_float(payload.global_scales[1]),
                weights[2], scales[2], ctypes.c_float(payload.global_scales[2]),
                device, ctypes.byref(self._plan),
            )
            self._native_call(
                "qwen38_cutlass_qkv_output", self._plan,
                ctypes.byref(self._output), ctypes.byref(self._output_elements),
            )
            if self._output_elements.value != 16 * FULL_OUTPUTS:
                raise DeviceDecodeError("fixed CUTLASS QKV output extent drift")
            self._native_call(
                "qwen38_qsa_indexer_create", output_pointers[0], output_pointers[1],
                ctypes.c_float(payload.output_global_scale), device,
                ctypes.byref(self._qsa_plan),
            )
            query, keys, table = self._qsa_inputs
            query_bytes, key_bytes, table_bytes = self._qsa_input_bytes
            self._native_call(
                "qwen38_qsa_indexer_inputs", self._qsa_plan,
                ctypes.byref(query), ctypes.byref(query_bytes),
                ctypes.byref(keys), ctypes.byref(key_bytes),
                ctypes.byref(table), ctypes.byref(table_bytes),
            )
            self._api.call("cudaStreamBeginCapture", self._stream, 1)
            self._native_call("qwen38_cutlass_qkv_launch", self._plan, activation, self._stream)
            self._api.call("cudaStreamEndCapture", self._stream, ctypes.byref(self._graph))
            self._api.call("cudaGraphInstantiate", ctypes.byref(self._exec), self._graph, 0)
        except BaseException:
            self._close_noexcept()
            raise

    def launch(self) -> None:
        self._require_open()
        self._api.call("cudaGraphLaunch", self._exec, self._stream)

    def capture_launch(self, stream: ctypes.c_void_p) -> None:
        """Append quantization and fixed Q/K/V nodes to an external capture."""

        self._require_open()
        if not isinstance(stream, ctypes.c_void_p) or not stream.value:
            raise DeviceDecodeError("external CUDA stream is invalid")
        self._native_call(
            "qwen38_cutlass_qkv_launch",
            self._plan,
            self._device_buffers[-1],
            stream,
        )

    def capture_qsa_topk_expansion(
        self,
        block_indices: ctypes.c_void_p,
        logical_positions: ctypes.c_void_p,
        sequence_lengths: ctypes.c_void_p,
        token_to_request: ctypes.c_void_p,
        token_indices: ctypes.c_void_p,
        rows: int,
        stream: ctypes.c_void_p,
    ) -> None:
        """Append fixed 512-block to 2,051-token expansion to a capture."""

        self._require_open()
        pointers = (
            block_indices,
            logical_positions,
            sequence_lengths,
            token_to_request,
            token_indices,
            stream,
        )
        if (
            any(
                not isinstance(pointer, ctypes.c_void_p) or not pointer.value
                for pointer in pointers
            )
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or not 1 <= rows <= 16
        ):
            raise DeviceDecodeError("fixed QSA expansion pointer ABI is invalid")
        self._native_call(
            "qwen38_qsa_expand_topk",
            block_indices,
            logical_positions,
            sequence_lengths,
            token_to_request,
            token_indices,
            rows,
            stream,
        )

    def capture_qsa_indexer(
        self,
        logical_positions: ctypes.c_void_p,
        sequence_lengths: ctypes.c_void_p,
        token_to_request: ctypes.c_void_p,
        stream: ctypes.c_void_p,
    ) -> None:
        """Append fixed paged score, stable top-k, and causal expansion."""

        self._require_open()
        if any(
            not isinstance(pointer, ctypes.c_void_p) or not pointer.value
            for pointer in (logical_positions, sequence_lengths, token_to_request, stream)
        ):
            raise DeviceDecodeError("fixed QSA indexer pointer ABI is invalid")
        self._native_call(
            "qwen38_qsa_indexer_launch", self._qsa_plan, logical_positions,
            sequence_lengths, token_to_request, stream,
        )

    def capture_qsa_attention(
        self,
        logical_positions: ctypes.c_void_p,
        token_to_request: ctypes.c_void_p,
        stream: ctypes.c_void_p,
    ) -> None:
        """Append fixed K/V update and FP32-LSE sparse attention nodes."""

        self._require_open()
        if any(
            not isinstance(pointer, ctypes.c_void_p) or not pointer.value
            for pointer in (logical_positions, token_to_request, stream)
        ):
            raise DeviceDecodeError("fixed QSA attention pointer ABI is invalid")
        self._native_call(
            "qwen38_qsa_attention_launch", self._qsa_plan, self._output,
            logical_positions, token_to_request, stream,
        )

    def qsa_output_bytes(self) -> bytes:
        """Copy the fixed 16 by 2,051 selected logical token ids."""

        self._require_open()
        pointer, elements = ctypes.c_void_p(), ctypes.c_size_t()
        self._native_call(
            "qwen38_qsa_indexer_output", self._qsa_plan,
            ctypes.byref(pointer), ctypes.byref(elements),
        )
        if elements.value != 16 * 2051:
            raise DeviceDecodeError("fixed QSA output extent drift")
        output = ctypes.create_string_buffer(elements.value * 4)
        self._api.call("cudaMemcpy", ctypes.addressof(output), pointer, len(output), 2)
        return output.raw

    def attention_output_bytes(self) -> bytes:
        """Copy fixed rank-local 16 by 12 by 256 BF16 attention output."""

        self._require_open()
        pointer, elements = ctypes.c_void_p(), ctypes.c_size_t()
        self._native_call(
            "qwen38_qsa_attention_output", self._qsa_plan,
            ctypes.byref(pointer), ctypes.byref(elements),
        )
        if elements.value != 16 * 12 * 256:
            raise DeviceDecodeError("fixed QSA attention output extent drift")
        output = ctypes.create_string_buffer(elements.value * 2)
        self._api.call("cudaMemcpy", ctypes.addressof(output), pointer, len(output), 2)
        return output.raw

    def projected_attention_bytes(self) -> bytes:
        """Copy fixed rank-local 16 by 2,560 BF16 projected output."""

        self._require_open()
        pointer, elements = ctypes.c_void_p(), ctypes.c_size_t()
        self._native_call("qwen38_qsa_projected_output", self._qsa_plan,
                          ctypes.byref(pointer), ctypes.byref(elements))
        if elements.value != 16 * 2560:
            raise DeviceDecodeError("fixed attention projection extent drift")
        output = ctypes.create_string_buffer(elements.value * 2)
        self._api.call("cudaMemcpy", ctypes.addressof(output), pointer, len(output), 2)
        return output.raw

    def update_activations(self, activations_bf16: bytes) -> None:
        """Replace the stable c16 input contents without changing graph pointers."""

        self._require_open()
        if (
            not isinstance(activations_bf16, bytes)
            or len(activations_bf16) != 16 * PROJECTION_K * 2
        ):
            raise DeviceDecodeError("live QKV activation extent is invalid")
        self._api.call(
            "cudaMemcpy",
            self._device_buffers[-1],
            ctypes.c_char_p(activations_bf16),
            len(activations_bf16),
            1,
        )

    @property
    def qsa_input_buffers(self) -> Mapping[str, tuple[int, int]]:
        """Return stable borrowed buffers for the single CUDA state owner."""

        self._require_open()
        return MappingProxyType(
            {
                name: (int(pointer.value), int(size.value))
                for name, pointer, size in zip(
                    ("query_bf16", "compressed_keys_bf16", "page_table_i32"),
                    self._qsa_inputs,
                    self._qsa_input_bytes,
                )
            }
        )

    def finish(self) -> None:
        self._require_open()
        self._api.call("cudaStreamSynchronize", self._stream)

    def output_bytes(self) -> bytes:
        self._require_open()
        output = ctypes.create_string_buffer(FULL_OUTPUT_BYTES)
        self._api.call(
            "cudaMemcpy", ctypes.addressof(output), self._output, FULL_OUTPUT_BYTES, 2
        )
        return output.raw

    def selected_outputs(self, rows: int = 4) -> tuple[float, ...]:
        if isinstance(rows, bool) or not isinstance(rows, int) or not 1 <= rows <= 4:
            raise DeviceDecodeError("selected output rows must be in 1..4")
        raw = self.output_bytes()
        values = [
            struct.unpack("<f", struct.pack("<I", value << 16))[0]
            for (value,) in struct.iter_unpack("<H", raw)
        ]
        result = []
        family_base = (0, 6144, 6144 + 256)
        widths = PROJECTION_FAMILY_ROWS
        for batch in range(16):
            for base, width in zip(family_base, widths):
                start = batch * FULL_OUTPUTS + base
                result.extend(values[start : start + rows])
        return tuple(result)

    @property
    def graph_nodes(self) -> int:
        self._require_open()
        nodes = ctypes.c_size_t()
        self._api.call("cudaGraphGetNodes", self._graph, None, ctypes.byref(nodes))
        return nodes.value

    @property
    def weight_table_pointer(self) -> int:
        """Return the borrowed first weight address for target ABI identity."""

        self._require_open()
        return int(self._device_buffers[0].value)

    def benchmark(self, iterations: int = 200) -> Mapping[str, float]:
        self._require_open()
        if isinstance(iterations, bool) or not isinstance(iterations, int) or not 10 <= iterations <= 10_000:
            raise DeviceDecodeError("benchmark iterations must be in 10..10000")
        activation = self._device_buffers[-1]

        def measured(launch) -> float:
            start, end = ctypes.c_void_p(), ctypes.c_void_p()
            self._api.call("cudaEventCreate", ctypes.byref(start))
            self._api.call("cudaEventCreate", ctypes.byref(end))
            try:
                for _ in range(10): launch()
                self._api.call("cudaStreamSynchronize", self._stream)
                self._api.call("cudaEventRecord", start, self._stream)
                for _ in range(iterations): launch()
                self._api.call("cudaEventRecord", end, self._stream)
                self._api.call("cudaEventSynchronize", end)
                elapsed = ctypes.c_float()
                self._api.call("cudaEventElapsedTime", ctypes.byref(elapsed), start, end)
                return elapsed.value / iterations
            finally:
                self._api.call("cudaEventDestroy", end)
                self._api.call("cudaEventDestroy", start)

        quant_ms = measured(
            lambda: self._native_call("qwen38_cutlass_qkv_quantize", self._plan, activation, self._stream)
        )
        projection_ms = measured(
            lambda: self._native_call("qwen38_cutlass_qkv_project", self._plan, self._stream)
        )
        graph_ms = measured(lambda: self._api.call("cudaGraphLaunch", self._exec, self._stream))
        return MappingProxyType(
            {"requant_ms": quant_ms, "projection_ms": projection_ms, "graph_ms": graph_ms}
        )

    def benchmark_qsa(
        self,
        logical_positions: ctypes.c_void_p,
        sequence_lengths: ctypes.c_void_p,
        token_to_request: ctypes.c_void_p,
        iterations: int = 40,
    ) -> Mapping[str, float]:
        """Measure fixed max-context score and exact selection independently."""

        self._require_open()
        if not 5 <= iterations <= 1000:
            raise DeviceDecodeError("QSA benchmark iterations must be in 5..1000")
        def measured(name: str) -> float:
            start, end = ctypes.c_void_p(), ctypes.c_void_p()
            self._api.call("cudaEventCreate", ctypes.byref(start))
            self._api.call("cudaEventCreate", ctypes.byref(end))
            try:
                for _ in range(3):
                    self._native_call(name, self._qsa_plan, logical_positions,
                                      sequence_lengths, token_to_request, self._stream)
                self._api.call("cudaStreamSynchronize", self._stream)
                self._api.call("cudaEventRecord", start, self._stream)
                for _ in range(iterations):
                    self._native_call(name, self._qsa_plan, logical_positions,
                                      sequence_lengths, token_to_request, self._stream)
                self._api.call("cudaEventRecord", end, self._stream)
                self._api.call("cudaEventSynchronize", end)
                elapsed = ctypes.c_float()
                self._api.call("cudaEventElapsedTime", ctypes.byref(elapsed), start, end)
                return elapsed.value / iterations
            finally:
                self._api.call("cudaEventDestroy", end)
                self._api.call("cudaEventDestroy", start)
        score = measured("qwen38_qsa_indexer_score")
        select = measured("qwen38_qsa_indexer_select_expand")
        def measured_attention(name: str, args: tuple[object, ...]) -> float:
            start, end = ctypes.c_void_p(), ctypes.c_void_p()
            self._api.call("cudaEventCreate", ctypes.byref(start))
            self._api.call("cudaEventCreate", ctypes.byref(end))
            try:
                for _ in range(3): self._native_call(name, *args, self._stream)
                self._api.call("cudaStreamSynchronize", self._stream)
                self._api.call("cudaEventRecord", start, self._stream)
                for _ in range(iterations): self._native_call(name, *args, self._stream)
                self._api.call("cudaEventRecord", end, self._stream)
                self._api.call("cudaEventSynchronize", end)
                elapsed = ctypes.c_float()
                self._api.call("cudaEventElapsedTime", ctypes.byref(elapsed), start, end)
                return elapsed.value / iterations
            finally:
                self._api.call("cudaEventDestroy", end)
                self._api.call("cudaEventDestroy", start)
        sparse_attention = measured_attention(
            "qwen38_qsa_sparse_attention",
            (self._qsa_plan, self._output, logical_positions, token_to_request),
        )
        output_projection = measured_attention(
            "qwen38_qsa_output_project", (self._qsa_plan,)
        )
        attention = sparse_attention + output_projection
        return MappingProxyType({"score_ms": score, "select_expand_ms": select,
                                 "sparse_attention_ms": sparse_attention,
                                 "output_projection_ms": output_projection,
                                 "attention_ms": attention,
                                 "total_ms": score + select + attention})

    def close(self) -> None:
        if self._closed: return
        errors = self._close_noexcept()
        if errors: raise DeviceDecodeError("CUTLASS QKV cleanup failed: " + "; ".join(errors))

    def __enter__(self) -> "CutlassQkvRuntime": return self
    def __exit__(self, exc_type, exc, traceback) -> None: self.close()

    def _native_call(self, name: str, *args) -> None:
        if getattr(self._native, name)(*args) != 0:
            detail = self._native.qwen38_cutlass_qkv_last_error()
            message = detail.decode("utf-8", "replace") if detail else "unknown failure"
            raise DeviceDecodeError(f"{name}: {message}")

    def _require_open(self) -> None:
        if self._closed: raise DeviceDecodeError("CUTLASS QKV runtime is closed")

    def _configure_native(self) -> None:
        self._native.qwen38_cutlass_qkv_last_error.restype = ctypes.c_char_p
        for name in (
            "qwen38_cutlass_qkv_create", "qwen38_cutlass_qkv_launch",
            "qwen38_cutlass_qkv_quantize", "qwen38_cutlass_qkv_project",
            "qwen38_cutlass_qkv_output", "qwen38_cutlass_qkv_destroy",
            "qwen38_qsa_expand_topk",
            "qwen38_qsa_indexer_create", "qwen38_qsa_indexer_launch",
            "qwen38_qsa_indexer_score", "qwen38_qsa_indexer_select_expand",
            "qwen38_qsa_indexer_inputs", "qwen38_qsa_indexer_output",
            "qwen38_qsa_indexer_destroy",
            "qwen38_qsa_attention_launch", "qwen38_qsa_sparse_attention",
            "qwen38_qsa_output_project", "qwen38_qsa_attention_output",
            "qwen38_qsa_projected_output",
        ):
            getattr(self._native, name).restype = ctypes.c_int

    def _close_noexcept(self) -> list[str]:
        errors = []
        if self._stream.value:
            try: self._api.call("cudaStreamSynchronize", self._stream)
            except Exception as exc: errors.append(str(exc))
        for name, handle in (("cudaGraphExecDestroy", self._exec), ("cudaGraphDestroy", self._graph)):
            if handle.value:
                try: self._api.call(name, handle)
                except Exception as exc: errors.append(str(exc))
        if self._plan.value:
            try: self._native_call("qwen38_cutlass_qkv_destroy", self._plan)
            except Exception as exc: errors.append(str(exc))
        if self._qsa_plan.value:
            try: self._native_call("qwen38_qsa_indexer_destroy", self._qsa_plan)
            except Exception as exc: errors.append(str(exc))
        for pointer in reversed(self._device_buffers):
            try: self._api.call("cudaFree", pointer)
            except Exception as exc: errors.append(str(exc))
        if self._stream.value:
            try: self._api.call("cudaStreamDestroy", self._stream)
            except Exception as exc: errors.append(str(exc))
        self._closed = True
        return errors
