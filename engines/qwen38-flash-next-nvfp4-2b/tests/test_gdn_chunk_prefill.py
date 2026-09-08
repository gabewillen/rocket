# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import types
import unittest
import inspect
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from qwen38_slab.gdn_chunk_prefill import (
    ATTENTION_SCALE,
    FLASHINFER_GDN_CHUNK_IDENTITY,
    AuthenticatedGdnChunkPrefillAdapter,
    FlashInferSm121GdnChunkBackend,
    GdnChunkPrefillError,
    GdnChunkPrefillTensors,
    ORACLE_MANIFEST_SHA256,
    REAL_TENSOR_BUNDLE_SCHEMA,
    execute_authenticated_real_tensor_bundle,
)


class Tensor:
    def __init__(self, shape, dtype, device="cuda:0", payload=b"tensor"):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.payload = payload

    def is_contiguous(self):
        return True

    def detach(self):
        return self

    def contiguous(self):
        return self

    def view(self, _dtype):
        return self

    def clone(self):
        return Tensor(self.shape, self.dtype, self.device, self.payload)

    def reshape(self, shape):
        self.shape = tuple(shape)
        return self

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        return self

    def numpy(self):
        return types.SimpleNamespace(tobytes=lambda: self.payload)


def tensors(rows=35):
    return GdnChunkPrefillTensors(
        Tensor((rows, 8, 128), "bfloat16"),
        Tensor((rows, 8, 128), "bfloat16"),
        Tensor((rows, 24, 128), "bfloat16"),
        Tensor((rows, 24), "float32"),
        Tensor((rows, 24), "float32"),
        Tensor((1, 24, 128, 128), "float32"),
        Tensor((rows, 24, 128), "bfloat16"),
        Tensor((1, 24, 128, 128), "float32"),
        Tensor((2,), "int64"),
    )


class Span:
    def __init__(self):
        self.attributes = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def set_attribute(self, key, value):
        self.attributes[key] = value


class Tracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, _name):
        span = Span()
        self.spans.append(span)
        return span


class Backend:
    implementation_identity = FLASHINFER_GDN_CHUNK_IDENTITY

    def __init__(self):
        self.calls = []

    def launch(self, value):
        self.calls.append(value)
        return value.output, value.final_state


class GdnChunkPrefillTests(unittest.TestCase):
    def test_real_tensor_bundle_owns_tensors_and_runs_supported_backend(self):
        allocations = []
        torch = types.SimpleNamespace(
            bfloat16="bfloat16",
            float32="float32",
            int64="int64",
            uint8="uint8",
            cuda=types.SimpleNamespace(
                synchronize=lambda device: allocations.append(("sync", device))
            ),
        )

        def allocate(shape, *_args, dtype, device, **_kwargs):
            self.assertEqual(device, "cuda:0")
            value = Tensor(tuple(shape), dtype)
            allocations.append(value)
            return value

        torch.empty = allocate
        torch.tensor = lambda values, *, dtype, device: allocate(
            (len(values),), dtype=dtype, device=device
        )
        torch.frombuffer = lambda payload, *, dtype: Tensor(
            (len(payload),), dtype, device="cpu", payload=bytes(payload)
        )
        backend = Backend()
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._real_tensor_bundle(Path(temporary))
            result = execute_authenticated_real_tensor_bundle(
                bundle, torch_module=torch, backend=backend
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["implementation"], FLASHINFER_GDN_CHUNK_IDENTITY)
        self.assertEqual(result["telemetry"]["execution.domain"], "chunk_prefill")
        self.assertEqual(allocations[-1], ("sync", 0))
        self.assertEqual(len(backend.calls), 1)
        source = inspect.getsource(execute_authenticated_real_tensor_bundle)
        self.assertNotIn("data_ptr", source)
        self.assertNotIn("ctypes", source)

    def test_real_tensor_bundle_rejects_payload_and_identity_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._real_tensor_bundle(Path(temporary))
            (bundle / "q.bin").write_bytes(b"bad")
            with self.assertRaises(GdnChunkPrefillError):
                execute_authenticated_real_tensor_bundle(
                    bundle, torch_module=types.SimpleNamespace(), backend=Backend()
                )

    @staticmethod
    def _real_tensor_bundle(root: Path) -> Path:
        rows = 35
        layouts = {
            "q": ("bfloat16", (rows, 8, 128), 2),
            "k": ("bfloat16", (rows, 8, 128), 2),
            "v": ("bfloat16", (rows, 24, 128), 2),
            "log_decay": ("float32", (rows, 24), 4),
            "beta": ("float32", (rows, 24), 4),
            "initial_state": ("float32", (1, 24, 128, 128), 4),
        }
        entries = []
        payloads = {}
        for index, (name, (dtype, shape, width)) in enumerate(layouts.items(), 1):
            size = width
            for extent in shape:
                size *= extent
            payload = bytes([index]) * size
            payloads[f"{name}.bin"] = payload
            entries.append(
                {
                    "name": name,
                    "file": f"{name}.bin",
                    "dtype": dtype,
                    "shape": list(shape),
                    "bytes": size,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifest = {
            "schema": REAL_TENSOR_BUNDLE_SCHEMA,
            "oracle_manifest_sha256": ORACLE_MANIFEST_SHA256,
            "implementation": FLASHINFER_GDN_CHUNK_IDENTITY,
            "rank": 0,
            "layer": 0,
            "rows": rows,
            "tensors": entries,
        }
        key = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["artifact_key"] = key
        bundle = root / key
        bundle.mkdir()
        for filename, payload in payloads.items():
            (bundle / filename).write_bytes(payload)
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        return bundle

    def test_authenticated_oracle35_call_publishes_exact_owned_outputs(self):
        backend, tracer = Backend(), Tracer()
        adapter = AuthenticatedGdnChunkPrefillAdapter(0, 0, backend, tracer)
        value = tensors()
        self.assertEqual(adapter.execute(value), (value.output, value.final_state))
        self.assertEqual(backend.calls, [value])
        self.assertEqual(
            tracer.spans[-1].attributes,
            {
                "execution.domain": "chunk_prefill",
                "rank": 0,
                "layer": 0,
                "rows": 35,
                "outcome": "success",
            },
        )

    def test_decode_shape_wrong_kind_and_bad_state_fail_closed(self):
        tracer = Tracer()
        with self.assertRaises(GdnChunkPrefillError):
            AuthenticatedGdnChunkPrefillAdapter(0, 3, Backend(), tracer)
        adapter = AuthenticatedGdnChunkPrefillAdapter(0, 4, Backend(), tracer)
        with self.assertRaises(GdnChunkPrefillError):
            adapter.execute(tensors(1))
        wrong = tensors()
        wrong.initial_state.dtype = "bfloat16"
        with self.assertRaises(GdnChunkPrefillError):
            adapter.execute(wrong)
        self.assertEqual(tracer.spans[-1].attributes["outcome"], "error")

    def test_publication_alias_is_rejected_before_backend(self):
        backend = Backend()
        adapter = AuthenticatedGdnChunkPrefillAdapter(1, 47 - 1, backend, Tracer())
        value = tensors(87)
        object.__setattr__(value, "final_state", value.initial_state)
        with self.assertRaises(GdnChunkPrefillError):
            adapter.execute(value)
        self.assertEqual(backend.calls, [])

    def test_flashinfer_backend_uses_pinned_supported_tensor_abi(self):
        calls = []

        def kernel(**kwargs):
            calls.append(kwargs)
            return kwargs["output"], kwargs["output_state"]

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__path__ = []
        flashinfer.__version__ = "0.6.17"
        gdn_prefill = types.ModuleType("flashinfer.gdn_prefill")
        gdn_prefill.chunk_gated_delta_rule = kernel
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(get_device_capability=lambda _device: (12, 1))
        decay = object()
        torch.exp = lambda _value: decay
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.gdn_prefill": gdn_prefill, "torch": torch},
        ):
            backend = FlashInferSm121GdnChunkBackend()
        value = tensors()
        self.assertEqual(backend.launch(value), (value.output, value.final_state))
        self.assertIs(calls[0]["g"], decay)
        self.assertEqual(calls[0]["scale"], ATTENTION_SCALE)
        self.assertFalse(calls[0]["use_qk_l2norm_in_kernel"])
        self.assertTrue(calls[0]["output_final_state"])
        self.assertEqual(calls[0]["use_cp"], "auto")
        self.assertIs(calls[0]["output"], value.output)
        self.assertIs(calls[0]["output_state"], value.final_state)

    def test_flashinfer_version_and_sm121a_are_exact(self):
        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__path__ = []
        flashinfer.__version__ = "0.6.18"
        gdn_prefill = types.ModuleType("flashinfer.gdn_prefill")
        gdn_prefill.chunk_gated_delta_rule = lambda **_kwargs: None
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(get_device_capability=lambda _device: (12, 1))
        with patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.gdn_prefill": gdn_prefill, "torch": torch},
        ):
            with self.assertRaises(GdnChunkPrefillError):
                FlashInferSm121GdnChunkBackend()


if __name__ == "__main__":
    unittest.main()
