# SPDX-License-Identifier: Apache-2.0
"""Authenticated Qwen3.8 GDN chunk-prefill adapter.

The pinned FLA implementation is a Python-launched Triton pipeline. On SM121a,
vLLM selects FlashInfer's CuTe-DSL implementation through the supported
``flashinfer.gdn_prefill.chunk_gated_delta_rule`` Torch tensor API. This module
keeps that boundary explicit and separate from Rocket's packed-decode port.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PINNED_VLLM_COMMIT = "8e685d198"
PINNED_FLASHINFER_VERSION = "0.6.17"
FLASHINFER_GDN_CHUNK_IDENTITY = (
    "vllm:8e685d198:flashinfer:0.6.17:gdn-prefill-sm121a"
)
NATIVE_GDN_M35_IDENTITY = "rocket:qwen38:native-cuda:gdn-prefill-m35:v1"
GDN_PREFILL_ACCURACY_IDENTITY = FLASHINFER_GDN_CHUNK_IDENTITY
ORACLE_MANIFEST_SHA256 = (
    "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b"
)

ORACLE_PREFILL_ROWS = frozenset((35, 87))
KEY_HEADS = 8
VALUE_HEADS = 24
HEAD_DIM = 128
ATTENTION_SCALE = HEAD_DIM**-0.5
REAL_TENSOR_BUNDLE_SCHEMA = "rocket.qwen38.gdn-prefill-tensors.v1"


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


class NativeGdnM35PrefillBackend:
    """Native CUDA M35 recurrence over caller-owned Torch device buffers."""

    implementation_identity = NATIVE_GDN_M35_IDENTITY

    class _Tensors(ctypes.Structure):
        _fields_ = [
            ("q", ctypes.c_void_p),
            ("k", ctypes.c_void_p),
            ("v", ctypes.c_void_p),
            ("log_decay", ctypes.c_void_p),
            ("beta", ctypes.c_void_p),
            ("initial_state", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("final_state", ctypes.c_void_p),
        ]

    class _Record(ctypes.Structure):
        _fields_ = [
            ("phase", ctypes.c_uint8),
            ("status", ctypes.c_uint8),
            ("rank", ctypes.c_int8),
            ("layer", ctypes.c_int8),
            ("rows", ctypes.c_uint8),
            ("success", ctypes.c_bool),
        ]

    class _Config(ctypes.Structure):
        _fields_ = [
            ("device", ctypes.c_int),
            ("rank", ctypes.c_int),
            ("layer", ctypes.c_int),
            ("rows", ctypes.c_int),
            ("publish", ctypes.c_void_p),
            ("publish_context", ctypes.c_void_p),
        ]

    def __init__(
        self,
        library: Path,
        *,
        rank: int = 0,
        layer: int = 0,
        rows: int = 35,
        torch_module: object | None = None,
    ) -> None:
        if (rank, layer, rows) != (0, 0, 35):
            raise GdnChunkPrefillError("native GDN M35 identity changed")
        self.expected_rank = rank
        self.expected_layer = layer
        self.expected_rows = rows
        self._torch = torch_module
        if self._torch is None:
            import torch

            self._torch = torch
        if tuple(self._torch.cuda.get_device_capability(0)) != (12, 1):
            raise GdnChunkPrefillError("native GDN M35 requires SM121a")
        self._library = ctypes.CDLL(str(library))
        self._records: list[tuple[int, int, int, int, int, bool]] = []
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, self._Record)
        self._callback = callback_type(
            lambda _context, record: self._records.append(
                (
                    record.phase,
                    record.status,
                    record.rank,
                    record.layer,
                    record.rows,
                    record.success,
                )
            )
        )
        self._library.qwen38_gdn_m35_prefill_create.argtypes = [
            ctypes.POINTER(self._Config),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._library.qwen38_gdn_m35_prefill_create.restype = ctypes.c_int
        self._library.qwen38_gdn_m35_prefill_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(self._Tensors),
            ctypes.c_void_p,
        ]
        self._library.qwen38_gdn_m35_prefill_launch.restype = ctypes.c_int
        self._library.qwen38_gdn_m35_prefill_destroy.argtypes = [ctypes.c_void_p]
        self._library.qwen38_gdn_m35_prefill_destroy.restype = ctypes.c_int
        self._owner = ctypes.c_void_p()
        config = self._Config(
            0,
            rank,
            layer,
            rows,
            ctypes.cast(self._callback, ctypes.c_void_p),
            None,
        )
        if (
            self._library.qwen38_gdn_m35_prefill_create(
                ctypes.byref(config), ctypes.byref(self._owner)
            )
            != 0
            or not self._owner.value
        ):
            raise GdnChunkPrefillError("native GDN M35 construction failed")

    def close(self) -> None:
        if self._owner.value:
            self._library.qwen38_gdn_m35_prefill_destroy(self._owner)
            self._owner = ctypes.c_void_p()

    @property
    def telemetry_records(self) -> tuple[tuple[int, int, int, int, int, bool], ...]:
        return tuple(self._records)

    def launch(self, tensors: GdnChunkPrefillTensors) -> tuple[object, object]:
        native = self._Tensors(
            *(
                ctypes.c_void_p(tensor.data_ptr())
                for tensor in (
                    tensors.q,
                    tensors.k,
                    tensors.v,
                    tensors.log_decay,
                    tensors.beta,
                    tensors.initial_state,
                    tensors.output,
                    tensors.final_state,
                )
            )
        )
        stream = self._torch.cuda.current_stream(0).cuda_stream
        if self._library.qwen38_gdn_m35_prefill_launch(
            self._owner, ctypes.byref(native), ctypes.c_void_p(stream)
        ) != 0:
            raise GdnChunkPrefillError("native GDN M35 launch failed")
        return tensors.output, tensors.final_state

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


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
            or backend.implementation_identity
            not in (FLASHINFER_GDN_CHUNK_IDENTITY, NATIVE_GDN_M35_IDENTITY)
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


class _ProofSpan:
    def __init__(self, attributes: dict[str, object]) -> None:
        self._attributes = attributes

    def __enter__(self) -> "_ProofSpan":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def set_attribute(self, key: str, value: object) -> None:
        self._attributes[key] = value


class _ProofTracer:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def start_as_current_span(self, _name: str) -> _ProofSpan:
        return _ProofSpan(self.attributes)


def _tensor_sha256(tensor: object, torch_module: object) -> str:
    detached = tensor.detach().contiguous().view(torch_module.uint8)
    return hashlib.sha256(detached.cpu().numpy().tobytes()).hexdigest()


_REAL_TENSOR_LAYOUTS = {
    "q": ("bfloat16", lambda rows: (rows, KEY_HEADS, HEAD_DIM)),
    "k": ("bfloat16", lambda rows: (rows, KEY_HEADS, HEAD_DIM)),
    "v": ("bfloat16", lambda rows: (rows, VALUE_HEADS, HEAD_DIM)),
    "log_decay": ("float32", lambda rows: (rows, VALUE_HEADS)),
    "beta": ("float32", lambda rows: (rows, VALUE_HEADS)),
    "initial_state": (
        "float32",
        lambda _rows: (1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
    ),
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_authenticated_real_tensor_bundle(
    bundle: Path, torch_module: object
) -> tuple[int, int, GdnChunkPrefillTensors]:
    """Load a content-addressed real-model tensor bundle onto CUDA device 0."""
    try:
        manifest = json.loads((bundle / "manifest.json").read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GdnChunkPrefillError("GDN prefill tensor bundle unavailable") from exc
    if not isinstance(manifest, dict):
        raise GdnChunkPrefillError("GDN prefill tensor manifest changed")
    digest_input = dict(manifest)
    artifact_key = digest_input.pop("artifact_key", None)
    observed_key = hashlib.sha256(_canonical_json(digest_input)).hexdigest()
    rank, layer, rows = (
        manifest.get("rank"),
        manifest.get("layer"),
        manifest.get("rows"),
    )
    if (
        artifact_key != observed_key
        or bundle.name != observed_key
        or manifest.get("schema") != REAL_TENSOR_BUNDLE_SCHEMA
        or manifest.get("oracle_manifest_sha256") != ORACLE_MANIFEST_SHA256
        or manifest.get("implementation") != FLASHINFER_GDN_CHUNK_IDENTITY
        or isinstance(rank, bool)
        or rank not in (0, 1)
        or isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < 48
        or layer % 4 == 3
        or isinstance(rows, bool)
        or rows not in ORACLE_PREFILL_ROWS
        or not isinstance(manifest.get("tensors"), list)
    ):
        raise GdnChunkPrefillError("GDN prefill tensor identity changed")
    entries = {
        entry.get("name"): entry
        for entry in manifest["tensors"]
        if isinstance(entry, dict)
    }
    if set(entries) != set(_REAL_TENSOR_LAYOUTS) or len(entries) != len(
        manifest["tensors"]
    ):
        raise GdnChunkPrefillError("GDN prefill tensor inventory changed")

    loaded: dict[str, object] = {}
    for name, (dtype_name, shape_fn) in _REAL_TENSOR_LAYOUTS.items():
        entry = entries[name]
        shape = shape_fn(rows)
        element_bytes = 2 if dtype_name == "bfloat16" else 4
        expected_bytes = math.prod(shape) * element_bytes
        filename = entry.get("file")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or entry.get("dtype") != dtype_name
            or tuple(entry.get("shape", ())) != shape
            or entry.get("bytes") != expected_bytes
        ):
            raise GdnChunkPrefillError("GDN prefill tensor layout changed")
        tensor_path = bundle / filename
        try:
            if tensor_path.is_symlink():
                raise GdnChunkPrefillError("GDN prefill tensor path changed")
            payload = tensor_path.read_bytes()
        except OSError as exc:
            raise GdnChunkPrefillError("GDN prefill tensor unavailable") from exc
        if (
            len(payload) != expected_bytes
            or hashlib.sha256(payload).hexdigest() != entry.get("sha256")
        ):
            raise GdnChunkPrefillError("GDN prefill tensor payload changed")
        dtype = getattr(torch_module, dtype_name)
        loaded[name] = (
            torch_module.frombuffer(bytearray(payload), dtype=dtype)
            .clone()
            .reshape(shape)
            .to("cuda:0")
        )
    loaded["output"] = torch_module.empty(
        (rows, VALUE_HEADS, HEAD_DIM), dtype=torch_module.bfloat16, device="cuda:0"
    )
    loaded["final_state"] = torch_module.empty(
        (1, VALUE_HEADS, HEAD_DIM, HEAD_DIM),
        dtype=torch_module.float32,
        device="cuda:0",
    )
    loaded["cu_seqlens"] = torch_module.tensor(
        [0, rows], dtype=torch_module.int64, device="cuda:0"
    )
    return rank, layer, GdnChunkPrefillTensors(**loaded)


def execute_authenticated_real_tensor_bundle(
    bundle: Path,
    *,
    torch_module: object | None = None,
    backend: GdnChunkPrefillBackend | None = None,
) -> dict[str, object]:
    if torch_module is None:
        import torch as torch_module
    rank, layer, tensors = load_authenticated_real_tensor_bundle(bundle, torch_module)
    tracer = _ProofTracer()
    implementation = backend or FlashInferSm121GdnChunkBackend()
    expected_identity = (
        getattr(implementation, "expected_rank", rank),
        getattr(implementation, "expected_layer", layer),
        getattr(implementation, "expected_rows", tensors.q.shape[0]),
    )
    if expected_identity != (rank, layer, tensors.q.shape[0]):
        raise GdnChunkPrefillError("GDN chunk-prefill backend identity changed")
    try:
        output, final_state = AuthenticatedGdnChunkPrefillAdapter(
            rank, layer, implementation, tracer
        ).execute(tensors)
        torch_module.cuda.synchronize(0)
        result = {
            "status": "success",
            "implementation": implementation.implementation_identity,
            "rank": rank,
            "layer": layer,
            "rows": tensors.q.shape[0],
            "output_sha256": _tensor_sha256(output, torch_module),
            "final_state_sha256": _tensor_sha256(final_state, torch_module),
            "telemetry": tracer.attributes,
        }
        records = getattr(implementation, "telemetry_records", None)
        if records is not None:
            result["native_records"] = records
        return result
    finally:
        close = getattr(implementation, "close", None)
        if callable(close):
            close()


def compare_native_real_tensor_bundle(
    bundle: Path, native_library: Path
) -> dict[str, object]:
    """Run fixed FlashInfer and native owners over independent bundle loads."""
    import torch

    reference_rank, reference_layer, reference = (
        load_authenticated_real_tensor_bundle(bundle, torch)
    )
    native_rank, native_layer, native = load_authenticated_real_tensor_bundle(
        bundle, torch
    )
    if (reference_rank, reference_layer) != (native_rank, native_layer):
        raise GdnChunkPrefillError("GDN comparison bundle identity changed")
    reference_backend = FlashInferSm121GdnChunkBackend()
    native_backend = NativeGdnM35PrefillBackend(
        native_library,
        rank=native_rank,
        layer=native_layer,
        rows=int(native.q.shape[0]),
        torch_module=torch,
    )
    try:
        reference_output, reference_state = reference_backend.launch(reference)
        native_output, native_state = native_backend.launch(native)
        torch.cuda.synchronize(0)

        def drift(left: object, right: object) -> dict[str, float]:
            difference = left.float() - right.float()
            return {
                "max_abs": float(difference.abs().max().item()),
                "rms": float(difference.square().mean().sqrt().item()),
            }

        native_output_sha256 = _tensor_sha256(native_output, torch)
        reference_output_sha256 = _tensor_sha256(reference_output, torch)
        native_final_state_sha256 = _tensor_sha256(native_state, torch)
        reference_final_state_sha256 = _tensor_sha256(reference_state, torch)
        exact_parity = (
            native_output_sha256 == reference_output_sha256
            and native_final_state_sha256 == reference_final_state_sha256
        )
        return {
            "status": "success",
            "rows": int(native.q.shape[0]),
            "output": drift(native_output, reference_output),
            "final_state": drift(native_state, reference_state),
            "native_output_sha256": native_output_sha256,
            "reference_output_sha256": reference_output_sha256,
            "native_final_state_sha256": native_final_state_sha256,
            "reference_final_state_sha256": reference_final_state_sha256,
            "exact_parity": exact_parity,
            "accepted_tolerance": "exact_hash",
            "accuracy_implementation": (
                NATIVE_GDN_M35_IDENTITY
                if exact_parity
                else GDN_PREFILL_ACCURACY_IDENTITY
            ),
            "native_records": native_backend.telemetry_records,
        }
    finally:
        native_backend.close()


def executable_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-bundle", required=True, type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--compare-flashinfer", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.compare_flashinfer:
            if args.native_library is None:
                raise GdnChunkPrefillError(
                    "native library is required for GDN comparison"
                )
            result = compare_native_real_tensor_bundle(
                args.tensor_bundle, args.native_library
            )
        else:
            backend = (
                NativeGdnM35PrefillBackend(args.native_library)
                if args.native_library
                else None
            )
            result = execute_authenticated_real_tensor_bundle(
                args.tensor_bundle, backend=backend
            )
    except BaseException:
        print(json.dumps({"status": "error", "stage": "chunk_prefill"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


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
    "ORACLE_MANIFEST_SHA256",
    "REAL_TENSOR_BUNDLE_SCHEMA",
    "PINNED_FLASHINFER_VERSION",
    "PINNED_VLLM_COMMIT",
    "NATIVE_GDN_M35_IDENTITY",
    "GDN_PREFILL_ACCURACY_IDENTITY",
    "NativeGdnM35PrefillBackend",
    "VALUE_HEADS",
    "execute_authenticated_real_tensor_bundle",
    "compare_native_real_tensor_bundle",
    "executable_main",
    "load_authenticated_real_tensor_bundle",
]
