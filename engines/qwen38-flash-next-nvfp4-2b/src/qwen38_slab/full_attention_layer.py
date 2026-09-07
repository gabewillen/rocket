"""Exact layer-3 full-attention state transition for the specialized K0 path."""

from __future__ import annotations

import ctypes
import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

from .decode import PreparedDecode
from .device_decode import Cuda13GraphRuntime, DeviceDecodeError, K0DeviceBinding
from .hyperconnection import CudaHyperConnectionRuntime


class FullAttentionLayerError(RuntimeError):
    """Exact layer transition or TP2 transport failure."""


class RdmaPairReduceRuntime:
    """Thin owner for the existing two-rail PairReduce implementation."""

    def __init__(
        self, rank: int, library: Path, bootstrap_host: str,
        bootstrap_port: int = 18839, timeout_ms: int = 120_000,
    ):
        if (
            rank not in (0, 1) or not isinstance(library, Path)
            or not isinstance(bootstrap_host, str) or not bootstrap_host
            or isinstance(bootstrap_port, bool) or not 1 <= bootstrap_port <= 65535
            or isinstance(timeout_ms, bool) or not 100 <= timeout_ms <= 120_000
        ):
            raise FullAttentionLayerError("layer PairReduce configuration is invalid")
        try:
            self._native = ctypes.CDLL(str(library))
        except OSError as exc:
            raise FullAttentionLayerError("cannot load layer PairReduce library") from exc
        self._native.qwen38_layer_pair_reduce_last_error.restype = ctypes.c_char_p
        for name in (
            "qwen38_layer_pair_reduce_create", "qwen38_layer_pair_reduce_launch",
            "qwen38_layer_pair_reduce_counts", "qwen38_layer_pair_reduce_destroy",
        ):
            getattr(self._native, name).restype = ctypes.c_int
        self._plan = ctypes.c_void_p()
        self._closed = False
        self._call(
            "qwen38_layer_pair_reduce_create", rank,
            bootstrap_host.encode("utf-8"), bootstrap_port, timeout_ms,
            ctypes.byref(self._plan),
        )

    def reduce(self, input_pointer: int, output_pointer: int, m: int,
               stream_pointer: int) -> None:
        if self._closed or m not in (1, 2, 4, 8, 16):
            raise FullAttentionLayerError("layer PairReduce bucket is invalid")
        self._call(
            "qwen38_layer_pair_reduce_launch", self._plan,
            ctypes.c_void_p(input_pointer), ctypes.c_void_p(output_pointer), m,
            ctypes.c_void_p(stream_pointer),
        )

    @property
    def counts(self) -> Mapping[str, int]:
        spans = ctypes.c_uint64()
        metrics = ctypes.c_uint64()
        failures = ctypes.c_uint64()
        self._call(
            "qwen38_layer_pair_reduce_counts", self._plan,
            ctypes.byref(spans), ctypes.byref(metrics), ctypes.byref(failures),
        )
        return MappingProxyType({"spans": spans.value, "metrics": metrics.value,
                                 "failures": failures.value})

    def _call(self, name: str, *args: object) -> None:
        if getattr(self._native, name)(*args) != 0:
            raw = self._native.qwen38_layer_pair_reduce_last_error()
            detail = raw.decode("utf-8", "replace") if raw else "unknown failure"
            raise FullAttentionLayerError(f"{name}: {detail}")

    def close(self) -> None:
        if self._closed: return
        self._call("qwen38_layer_pair_reduce_destroy", self._plan)
        self._closed = True

    def __enter__(self) -> "RdmaPairReduceRuntime": return self
    def __exit__(self, exc_type, exc, traceback) -> None: self.close()


@dataclass(frozen=True)
class FullAttentionPublication:
    generation: int
    graph_batch: int
    intermediate_hashes: Mapping[str, str]
    stage_ms: Mapping[str, float]


class FullAttentionLayerExecutor:
    """Compose exact HC/attention/TP2/HC order without publishing partial state."""

    _STAGES = (
        "attn_hc_mix", "activation_bind", "qsa_attention",
        "pair_reduce", "mlp_hc_combine_mix",
    )

    def __init__(
        self, runtime: Cuda13GraphRuntime, binding: K0DeviceBinding,
        hyperconnection: CudaHyperConnectionRuntime,
        reducer: RdmaPairReduceRuntime, tracer,
    ):
        if any(value is None for value in (runtime, binding, hyperconnection, reducer, tracer)):
            raise FullAttentionLayerError("full-attention dependencies are required")
        try:
            runtime_identity = runtime.rank_slab_identity
            hc_identity = hyperconnection.rank_slab_identity
        except (AttributeError, DeviceDecodeError) as exc:
            raise FullAttentionLayerError("full-attention slab identity is unavailable") from exc
        if runtime_identity != hc_identity or runtime_identity[1:] != (0, 3):
            raise FullAttentionLayerError("full-attention slab identity drift")
        self._runtime = runtime
        self._binding = binding
        self._hc = hyperconnection
        self._reducer = reducer
        self._tracer = tracer
        self._last_generation = 0
        self._faulted = False

    def execute(
        self, prepared: PreparedDecode, *, verify_reference: bool = False
    ) -> FullAttentionPublication:
        if self._faulted:
            raise FullAttentionLayerError("faulted full-attention executor cannot retry")
        if not isinstance(prepared, PreparedDecode):
            raise FullAttentionLayerError("full-attention executor requires PreparedDecode")
        m = prepared.lease.graph_batch
        if m not in (1, 2, 4, 8, 16) or prepared.lease.generation != self._last_generation + 1:
            raise FullAttentionLayerError("full-attention generation or bucket drift")
        try:
            hashes, stage_ms = self._run(prepared, m)
            if verify_reference:
                self._hc.reference_mix(m)
                self._hc.synchronize()
                reference_mix = self._hc.hashes(m, ("block_input", "injection"))
                self._hc.reference_combine(m)
                self._hc.synchronize()
                reference_final = self._hc.hashes(
                    m, ("reduced", "updated_hidden", "next_block_input", "next_injection")
                )
                reference = {
                    "attn_hc.block_input": reference_mix["block_input"],
                    "attn_hc.injection": reference_mix["injection"],
                    "pair_reduce.output": reference_final["reduced"],
                    "mlp_hc.updated_hidden": reference_final["updated_hidden"],
                    "mlp_hc.block_input": reference_final["next_block_input"],
                    "mlp_hc.injection": reference_final["next_injection"],
                }
                compared = {key: hashes[key] for key in reference}
                if compared != reference:
                    raise FullAttentionLayerError("composed/reference intermediate hash drift")
        except BaseException:
            self._faulted = True
            raise
        self._last_generation = prepared.lease.generation
        return FullAttentionPublication(
            self._last_generation, m, MappingProxyType(dict(hashes)),
            MappingProxyType(dict(stage_ms)),
        )

    def _run(
        self, prepared: PreparedDecode, m: int
    ) -> tuple[Mapping[str, str], Mapping[str, float]]:
        pointers = self._hc.pointers
        stage_ms = {}
        start = time.perf_counter_ns()
        with self._observed("attn_hc_mix", m):
            self._hc.launch_mix(m)
            self._hc.synchronize()
        stage_ms["attn_hc_mix"] = (time.perf_counter_ns() - start) / 1.0e6
        mix_hashes = self._hc.hashes(m, ("block_input", "injection"))
        start = time.perf_counter_ns()
        with self._observed("activation_bind", m):
            self._runtime.bind_qkv_activation_device(pointers["block_input"], m)
        stage_ms["activation_bind"] = (time.perf_counter_ns() - start) / 1.0e6
        start = time.perf_counter_ns()
        with self._observed("qsa_attention", m):
            self._binding.upload_and_launch(prepared)
        stage_ms["qsa_attention"] = (time.perf_counter_ns() - start) / 1.0e6
        projected = self._runtime.read_qsa_projected_output()[: m * 2560 * 2]
        start = time.perf_counter_ns()
        with self._observed("pair_reduce", m):
            self._reducer.reduce(
                self._runtime.projected_attention_pointer, pointers["reduced"], m,
                self._runtime.stream_pointer,
            )
        stage_ms["pair_reduce"] = (time.perf_counter_ns() - start) / 1.0e6
        reduced_hash = self._hc.hashes(m, ("reduced",))["reduced"]
        start = time.perf_counter_ns()
        with self._observed("mlp_hc_combine_mix", m):
            self._hc.launch_combine(m)
            self._hc.synchronize()
        stage_ms["mlp_hc_combine_mix"] = (time.perf_counter_ns() - start) / 1.0e6
        final_hashes = self._hc.hashes(
            m, ("updated_hidden", "next_block_input", "next_injection")
        )
        return MappingProxyType({
            "attn_hc.block_input": mix_hashes["block_input"],
            "attn_hc.injection": mix_hashes["injection"],
            "attention.projected": hashlib.sha256(projected).hexdigest(),
            "pair_reduce.output": reduced_hash,
            "mlp_hc.updated_hidden": final_hashes["updated_hidden"],
            "mlp_hc.block_input": final_hashes["next_block_input"],
            "mlp_hc.injection": final_hashes["next_injection"],
        }), MappingProxyType(stage_ms)

    @contextmanager
    def _observed(self, stage: str, m: int) -> Iterator[None]:
        if stage not in self._STAGES:
            raise FullAttentionLayerError("unbounded full-attention stage")
        with self._tracer.start_as_current_span(
            "rocket.qwen38.decode.full_attention_layer"
        ) as span:
            span.set_attribute("stage", stage)
            span.set_attribute("layer", 3)
            span.set_attribute("rank", 0)
            span.set_attribute("graph_batch", m)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")
