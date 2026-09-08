# SPDX-License-Identifier: Apache-2.0
"""Authenticated Qwen3.8 GDN chunk-prefill adapter.

The pinned FLA implementation is a Python-launched Triton pipeline. On SM121a,
vLLM selects FlashInfer's CuTe-DSL implementation through the supported
``flashinfer.gdn_prefill.chunk_gated_delta_rule`` Torch tensor API. This module
keeps that boundary explicit and separate from Rocket's packed-decode port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

PINNED_VLLM_COMMIT = "8e685d198"
PINNED_FLASHINFER_VERSION = "0.6.17"
FLASHINFER_GDN_CHUNK_IDENTITY = (
    "vllm:8e685d198:flashinfer:0.6.17:gdn-prefill-sm121a"
)

ORACLE_PREFILL_ROWS = frozenset((35, 87))
KEY_HEADS = 8
VALUE_HEADS = 24
HEAD_DIM = 128
ATTENTION_SCALE = HEAD_DIM**-0.5


class GdnChunkPrefillError(RuntimeError):
    """The authenticated chunk-prefill backend or tensor contract changed."""


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, *args: object) -> object: ...
    def set_attribute(self, key: str, value: object) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


@dataclass(frozen=True)
class GdnChunkPrefillTensors:
    """Caller-owned contiguous CUDA tensors for one authenticated sequence."""

    q: object
    k: object
    v: object
    log_decay: object
    beta: object
    initial_state: object
    output: object
    final_state: object
    cu_seqlens: object


class GdnChunkPrefillBackend(Protocol):
    implementation_identity: str

    def launch(
        self, tensors: GdnChunkPrefillTensors
    ) -> tuple[object, object]: ...


def _dtype_name(tensor: object) -> str:
    return str(getattr(tensor, "dtype", "")).removeprefix("torch.")


def _device_name(tensor: object) -> str:
    return str(getattr(tensor, "device", ""))


def _validate_tensor(
    tensor: object, *, shape: tuple[int, ...], dtype: str
) -> None:
    contiguous = getattr(tensor, "is_contiguous", None)
    if (
        tuple(getattr(tensor, "shape", ())) != shape
        or _dtype_name(tensor) != dtype
        or _device_name(tensor) != "cuda:0"
        or not callable(contiguous)
        or contiguous() is not True
    ):
        raise GdnChunkPrefillError("GDN chunk-prefill tensor contract changed")


class FlashInferSm121GdnChunkBackend:
    """FlashInfer 0.6.17 SM121a backend using its supported Torch ABI.

    The call runs on Torch's current CUDA stream. The adapter supplies already
    normalized Q/K, so in-kernel normalization remains disabled as in pinned
    vLLM. ``log_decay`` is exponentiated because FlashInfer consumes decay,
    while the pinned FLA boundary carries log decay.
    """

    implementation_identity = FLASHINFER_GDN_CHUNK_IDENTITY

    def __init__(self) -> None:
        import flashinfer
        import torch
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        if getattr(flashinfer, "__version__", None) != PINNED_FLASHINFER_VERSION:
            raise GdnChunkPrefillError("FlashInfer GDN prefill version changed")
        if tuple(torch.cuda.get_device_capability(0)) != (12, 1):
            raise GdnChunkPrefillError("GDN chunk prefill requires SM121a")
        self._torch = torch
        self._kernel = chunk_gated_delta_rule

    def launch(
        self, tensors: GdnChunkPrefillTensors
    ) -> tuple[object, object]:
        decay = self._torch.exp(tensors.log_decay)
        result = self._kernel(
            q=tensors.q,
            k=tensors.k,
            v=tensors.v,
            g=decay,
            beta=tensors.beta,
            scale=ATTENTION_SCALE,
            initial_state=tensors.initial_state,
            output_final_state=True,
            cu_seqlens=tensors.cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=tensors.output,
            output_state=tensors.final_state,
            use_cp="auto",
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise GdnChunkPrefillError("FlashInfer GDN prefill result changed")
        return result


class AuthenticatedGdnChunkPrefillAdapter:
    """Bounded one-sequence adapter over an authenticated chunk kernel.

    The backend and tracer are borrowed and must outlive this adapter. Calls are
    single-stream through Torch's current CUDA stream and are not reentrant.
    Only the authenticated 35-row and 87-row oracle shapes are accepted. A
    failed call does not claim state publication; the caller owns cleanup.
    """

    def __init__(
        self,
        rank: int,
        layer: int,
        backend: GdnChunkPrefillBackend,
        tracer: OtelTracer,
    ) -> None:
        if (
            rank not in (0, 1)
            or not 0 <= layer < 48
            or layer % 4 == 3
            or backend.implementation_identity != FLASHINFER_GDN_CHUNK_IDENTITY
        ):
            raise GdnChunkPrefillError("GDN chunk-prefill owner identity changed")
        self._rank = rank
        self._layer = layer
        self._backend = backend
        self._tracer = tracer
        self._active = False

    def execute(self, tensors: GdnChunkPrefillTensors) -> tuple[object, object]:
        rows = int(getattr(tensors.q, "shape", (0,))[0])
        with self._tracer.start_as_current_span(
            "rocket.qwen38.gdn.chunk_prefill"
        ) as span:
            span.set_attribute("execution.domain", "chunk_prefill")
            span.set_attribute("rank", self._rank)
            span.set_attribute("layer", self._layer)
            span.set_attribute("rows", rows if rows in ORACLE_PREFILL_ROWS else -1)
            try:
                if self._active or rows not in ORACLE_PREFILL_ROWS:
                    raise GdnChunkPrefillError(
                        "GDN chunk-prefill execution contract changed"
                    )
                self._validate(tensors, rows)
                self._active = True
                output, final_state = self._backend.launch(tensors)
                if output is not tensors.output or final_state is not tensors.final_state:
                    raise GdnChunkPrefillError(
                        "GDN chunk-prefill publication contract changed"
                    )
            except BaseException:
                span.set_attribute("outcome", "error")
                raise
            finally:
                self._active = False
            span.set_attribute("outcome", "success")
            return output, final_state

    @staticmethod
    def _validate(tensors: GdnChunkPrefillTensors, rows: int) -> None:
        _validate_tensor(tensors.q, shape=(rows, KEY_HEADS, HEAD_DIM), dtype="bfloat16")
        _validate_tensor(tensors.k, shape=(rows, KEY_HEADS, HEAD_DIM), dtype="bfloat16")
        _validate_tensor(tensors.v, shape=(rows, VALUE_HEADS, HEAD_DIM), dtype="bfloat16")
        _validate_tensor(tensors.log_decay, shape=(rows, VALUE_HEADS), dtype="float32")
        _validate_tensor(tensors.beta, shape=(rows, VALUE_HEADS), dtype="float32")
        _validate_tensor(
            tensors.initial_state,
            shape=(1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
            dtype="float32",
        )
        _validate_tensor(
            tensors.output,
            shape=(rows, VALUE_HEADS, HEAD_DIM),
            dtype="bfloat16",
        )
        _validate_tensor(
            tensors.final_state,
            shape=(1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
            dtype="float32",
        )
        _validate_tensor(tensors.cu_seqlens, shape=(2,), dtype="int64")
        if tensors.output is tensors.q or tensors.final_state is tensors.initial_state:
            raise GdnChunkPrefillError("GDN chunk-prefill output alias changed")


__all__ = [
    "ATTENTION_SCALE",
    "AuthenticatedGdnChunkPrefillAdapter",
    "FLASHINFER_GDN_CHUNK_IDENTITY",
    "FlashInferSm121GdnChunkBackend",
    "GdnChunkPrefillError",
    "GdnChunkPrefillTensors",
    "HEAD_DIM",
    "KEY_HEADS",
    "ORACLE_PREFILL_ROWS",
    "PINNED_FLASHINFER_VERSION",
    "PINNED_VLLM_COMMIT",
    "VALUE_HEADS",
]
