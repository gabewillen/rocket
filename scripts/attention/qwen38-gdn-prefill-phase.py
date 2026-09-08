#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time the pinned FlashInfer recurrence at Qwen3.8 TP2 prefill shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
from types import SimpleNamespace
from typing import Callable

import torch
import flashinfer
import vllm
from flashinfer.autotuner import AutoTuner
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
)
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    layer_norm_fwd,
)
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    contiguous = tensor.detach().contiguous()
    payload = contiguous.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def fixture_tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def write_projection_fixture(root: pathlib.Path, tokens: int, hidden, qkvz_layer,
                             ba_layer) -> None:
    directory = root / f"tokens-{tokens}"
    directory.mkdir(parents=True, exist_ok=False)
    qkvz_a, qkvz_sfa = scaled_fp4_quant(
        hidden, qkvz_layer.input_global_scale_inv,
        is_sf_swizzled_layout=True, backend="flashinfer-cutlass",
        padded_n=hidden.shape[-1],
    )
    ba_a, ba_sfa = scaled_fp4_quant(
        hidden, ba_layer.input_global_scale_inv,
        is_sf_swizzled_layout=True, backend="flashinfer-cutlass",
        padded_n=hidden.shape[-1],
    )
    tensors = {
        "hidden.bin": hidden,
        "qkvz_a.bin": qkvz_a,
        "qkvz_sfa.bin": qkvz_sfa.view(torch.uint8),
        "ba_a.bin": ba_a,
        "ba_sfa.bin": ba_sfa.view(torch.uint8),
        "qkvz_b.bin": qkvz_layer.weight,
        "qkvz_sfb.bin": qkvz_layer.weight_scale.view(torch.uint8),
        "ba_b.bin": ba_layer.weight,
        "ba_sfb.bin": ba_layer.weight_scale.view(torch.uint8),
        "alpha.bin": qkvz_layer.alpha,
    }
    files = {}
    for name, tensor in tensors.items():
        payload = fixture_tensor_bytes(tensor)
        (directory / name).write_bytes(payload)
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "format": "rocket-gdn-fp4-fixture-v1",
        "provenance": "python-synthetic-seed-7",
        "tokens": tokens,
        "qkvz_mnk": [tokens, 8192, 2560],
        "ba_mnk": [tokens, 64, 2560],
        "layouts": {
            "hidden": {"shape": [tokens, 2560], "stride": [2560, 1], "dtype": "bfloat16"},
            "packed_a": {"shape": [tokens, 1280], "stride": [1280, 1], "dtype": "uint8"},
            "sfa": {"shape": [((tokens + 127) // 128) * 128, 160],
                    "stride": [160, 1], "dtype": "uint8"},
            "qkvz_b": {"shape": [8192, 1280], "stride": [1280, 1], "dtype": "uint8"},
            "qkvz_sfb": {"shape": [8192, 160], "stride": [160, 1], "dtype": "uint8"},
            "ba_b": {"shape": [64, 1280], "stride": [1280, 1], "dtype": "uint8"},
            "ba_sfb": {"shape": [128, 160], "stride": [160, 1], "dtype": "uint8"},
        },
        "files": files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_fixture_tensor(directory: pathlib.Path, manifest: dict[str, object],
                        name: str, dtype: torch.dtype, shape: tuple[int, ...],
                        device: torch.device) -> torch.Tensor:
    payload = (directory / name).read_bytes()
    record = manifest["files"][name]
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise RuntimeError(f"projection fixture identity mismatch: {name}")
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    return raw.view(dtype).reshape(shape).to(device)


def import_projection_fixture(root: pathlib.Path, tokens: int,
                              device: torch.device):
    directory = root / f"tokens-{tokens}"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format") != "rocket-gdn-fp4-fixture-v1" or
            manifest.get("tokens") != tokens):
        raise RuntimeError("projection fixture contract changed")
    padded_m = ((tokens + 127) // 128) * 128
    hidden = load_fixture_tensor(directory, manifest, "hidden.bin", torch.bfloat16,
                                 (tokens, 2560), device)
    qkvz_a = load_fixture_tensor(directory, manifest, "qkvz_a.bin", torch.uint8,
                                 (tokens, 1280), device)
    qkvz_sfa = load_fixture_tensor(directory, manifest, "qkvz_sfa.bin", torch.uint8,
                                   (padded_m, 160), device)
    ba_a = load_fixture_tensor(directory, manifest, "ba_a.bin", torch.uint8,
                               (tokens, 1280), device)
    ba_sfa = load_fixture_tensor(directory, manifest, "ba_sfa.bin", torch.uint8,
                                 (padded_m, 160), device)
    qkvz_layer = torch.nn.Module()
    qkvz_layer.output_size_per_partition = 8192
    qkvz_layer.weight = torch.nn.Parameter(load_fixture_tensor(
        directory, manifest, "qkvz_b.bin", torch.uint8, (8192, 1280), device),
        requires_grad=False)
    qkvz_layer.weight_scale = torch.nn.Parameter(load_fixture_tensor(
        directory, manifest, "qkvz_sfb.bin", torch.uint8, (8192, 160), device),
        requires_grad=False)
    qkvz_layer.alpha = torch.nn.Parameter(load_fixture_tensor(
        directory, manifest, "alpha.bin", torch.float32, (), device),
        requires_grad=False)
    ba_layer = torch.nn.Module()
    ba_layer.output_size_per_partition = 48
    ba_layer.weight = torch.nn.Parameter(load_fixture_tensor(
        directory, manifest, "ba_b.bin", torch.uint8, (64, 1280), device),
        requires_grad=False)
    ba_layer.weight_scale = torch.nn.Parameter(load_fixture_tensor(
        directory, manifest, "ba_sfb.bin", torch.uint8, (128, 160), device),
        requires_grad=False)
    ba_layer.alpha = qkvz_layer.alpha
    qkvz_qa = QuantizedActivation(qkvz_a, qkvz_sfa, torch.bfloat16,
                                  torch.Size((tokens, 2560)), kNvfp4Dynamic)
    ba_qa = QuantizedActivation(ba_a, ba_sfa, torch.bfloat16,
                                torch.Size((tokens, 2560)), kNvfp4Dynamic)
    return qkvz_layer, ba_layer, qkvz_qa, ba_qa, hidden, manifest


def loaded_shared_object_identity(basename: str) -> dict[str, object]:
    """Return one loaded artifact identity, failing on absent/ambiguous matches."""
    matches: set[pathlib.Path] = set()
    for line in pathlib.Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and pathlib.Path(fields[5]).name == basename:
            matches.add(pathlib.Path(fields[5]))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one loaded {basename}, found {sorted(map(str, matches))}"
        )
    path = matches.pop()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def sm120_cutlass_tactic(tactic: object) -> dict[str, object]:
    """Decode the pinned FlashInfer 0.6.17 SM120 tactic table."""
    tiles = (
        (128, 32, 128),
        (128, 32, 256),
        (128, 64, 128),
        (128, 64, 256),
        (128, 128, 128),
        (128, 128, 256),
        (256, 128, 128),
        (128, 256, 128),
    )
    if tactic == -1:
        return {
            "id": -1,
            "tile_mnk": [128, 128, 256],
            "scheduler": "dp_static_persistent",
            "swap_ab": False,
            "cluster": [1, 1, 1],
            "source": "FlashInfer-0.6.17-fallback",
        }
    if not isinstance(tactic, int) or not 0 <= tactic < len(tiles) * 4:
        raise RuntimeError(f"unrecognized FlashInfer SM120 FP4 tactic {tactic!r}")
    tile = tiles[tactic // 4]
    variant = tactic % 4
    return {
        "id": tactic,
        "tile_mnk": list(tile),
        "scheduler": "stream_k" if variant >= 2 else "dp_static_persistent",
        "swap_ab": variant in (0, 2),
        "cluster": [1, 1, 1],
        "source": "FlashInfer-0.6.17-getConfigs",
    }


class Fp4DispatchTrace:
    """Observe FlashInfer's actual runner/tactic choice without changing it."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self._tuner = AutoTuner.get()
        self._original = self._tuner.choose_one

    def __enter__(self) -> "Fp4DispatchTrace":
        def traced_choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
            runner, tactic = self._original(
                custom_op, runners, tuning_config, inputs, **kwargs
            )
            if custom_op == "fp4_gemm":
                shapes = tuple(self._tuner._get_input_sizes(inputs))
                runner_keys = [
                    AutoTuner._get_cache_key(
                        custom_op,
                        candidate,
                        shapes,
                        tuning_config,
                        candidate.get_cache_key_extras(inputs),
                    )
                    for candidate in runners
                ]
                memory_hit = any(
                    key in self._tuner.profiling_cache for key in runner_keys
                )
                loaded_hit = any(
                    key.file_key in self._tuner._file_configs for key in runner_keys
                )
                cache_hit, cached_runner, cached_tactic, _ = self._tuner.search_cache(
                    custom_op, runners, shapes, tuning_config, inputs=inputs
                )
                runner_index = runners.index(runner)
                if cached_runner != runner_index or cached_tactic != tactic:
                    raise RuntimeError("FlashInfer dispatch changed during attribution")
                bundled_enabled = (
                    os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0") == "1"
                )
                if memory_hit:
                    cache_source = "process_memory"
                elif loaded_hit:
                    cache_source = "explicit_loaded_config"
                elif cache_hit and bundled_enabled:
                    cache_source = "bundled_package_config"
                elif not cache_hit and tactic == -1:
                    cache_source = "miss_fallback"
                else:
                    raise RuntimeError("unattributable FlashInfer cache-chain result")
                tensors = []
                for value in inputs:
                    if isinstance(value, torch.Tensor):
                        tensors.append({
                            "shape": list(value.shape),
                            "stride": list(value.stride()),
                            "dtype": str(value.dtype),
                            "data_ptr_mod_128": value.data_ptr() % 128,
                        })
                    else:
                        tensors.append(None)
                self.records.append({
                    "runner": (
                        f"{type(runner).__module__}.{type(runner).__qualname__}"
                    ),
                    "runner_index": runner_index,
                    "tactic": sm120_cutlass_tactic(tactic),
                    "cache_chain_result": cache_source,
                    "tuning_mode": bool(self._tuner.is_tuning_mode),
                    "bundled_cache_enabled": bundled_enabled,
                    "inputs": tensors,
                })
            return runner, tactic

        self._tuner.choose_one = traced_choose_one
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del self._tuner.choose_one

    def summary(self) -> list[dict[str, object]]:
        unique: dict[str, dict[str, object]] = {}
        counts: dict[str, int] = {}
        for record in self.records:
            key = json.dumps(record, sort_keys=True)
            unique[key] = record
            counts[key] = counts.get(key, 0) + 1
        result = []
        for key, record in unique.items():
            result.append({**record, "python_dispatch_calls": counts[key]})
        return result


def gpu_clock_power_snapshot() -> dict[str, object]:
    fields = "clocks.current.graphics,clocks.current.memory,power.draw,power.limit"
    command = [
        "nvidia-smi", "--id=0", f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(values) != 4:
        raise RuntimeError(f"unexpected nvidia-smi clock/power output: {values}")
    return {
        "graphics_clock_mhz": values[0],
        "memory_clock_mhz": values[1],
        "power_draw_w": values[2],
        "power_limit_w": values[3],
        "controlled": False,
        "scope": "post_measurement_snapshot",
    }


def projection_contract(tokens: int, warmup: int, iterations: int) -> dict[str, object]:
    return {
        "version": 1,
        "engine": "vllm",
        "executable": "python:qwen38-gdn-prefill-phase.py",
        "scope": "one_cuda_graph_replay_of_two_sequential_modelopt_apply_calls",
        "logical_mnk": [[tokens, 8192, 2560], [tokens, 48, 2560]],
        "physical_mnk": [[tokens, 8192, 2560], [tokens, 64, 2560]],
        "activation_quantizations": 2,
        "gemms": 2,
        "quant_backend": "vllm.scaled_fp4_quant:flashinfer-cutlass",
        "includes": ["qkvz_dynamic_quant", "qkvz_gemm", "ba_dynamic_quant", "ba_gemm"],
        "excludes": ["weight_preparation", "autotuning", "host_dispatch", "output_projection"],
        "eager_warmups": warmup,
        "capture_calls": 1,
        "post_capture_replays_before_samples": 1,
        "samples": iterations,
        "timer": "cuda_events_around_graph_replay",
        "p50": "statistics.median",
        "p95": "sorted[ceil(samples*0.95)-1]",
        "gpu_clocks": "unlocked",
        "weights": "synthetic_random_packed_nvfp4_with_unit_scales",
    }


def projection_data_identity(hidden, qkvz_layer, ba_layer) -> dict[str, object]:
    """Name synthetic comparator data so it cannot be equated with slab data."""
    return {
        "hidden_bf16": tensor_identity(hidden),
        "qkvz_weight_packed": tensor_identity(qkvz_layer.weight),
        "qkvz_weight_scale_swizzled": tensor_identity(qkvz_layer.weight_scale),
        "qkvz_alpha": tensor_identity(qkvz_layer.alpha),
        "ba_weight_packed_padded_n64": tensor_identity(ba_layer.weight),
        "ba_weight_scale_swizzled_padded_n128": tensor_identity(ba_layer.weight_scale),
        "ba_alpha": tensor_identity(ba_layer.alpha),
    }


def attribute_projection_backend(
    result: dict[str, object], method: object, trace: Fp4DispatchTrace,
    contract: dict[str, object],
) -> None:
    """Attach observed dispatch and measurement scope to a projection result."""
    kernel = method.kernel
    kernel_type = type(kernel)
    result["vllm_kernel_class"] = (
        f"{kernel_type.__module__}.{kernel_type.__qualname__}"
    )
    dispatches = trace.summary()
    if not dispatches:
        raise RuntimeError("no FlashInfer FP4 dispatch observed")
    result["flashinfer_dispatches"] = dispatches
    result["flashinfer_artifact"] = loaded_shared_object_identity(
        "fp4_gemm_cutlass_sm120.so"
    )
    result["comparator_contract"] = contract


def measure(
    phase: str,
    operation: Callable[[], None],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    return {
        "phase": phase,
        "p50_us": statistics.median(samples),
        "p95_us": percentile(samples, 0.95),
        "graph_capture": "pass",
        "samples_us": samples,
        "gpu_clock_power": gpu_clock_power_snapshot(),
    }


def nvfp4_layer(method: ModelOptNvFp4LinearMethod, inputs: int, outputs: int):
    layer = torch.nn.Module()
    layer.output_size_per_partition = outputs
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (outputs, inputs // 2), dtype=torch.uint8,
                      device="cuda"),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(outputs, inputs // 16, dtype=torch.float8_e4m3fn,
                   device="cuda"),
        requires_grad=False,
    )
    layer.input_global_scale_inv = torch.nn.Parameter(
        torch.ones((), dtype=torch.float32, device="cuda"), requires_grad=False
    )
    layer.alpha = torch.nn.Parameter(
        torch.ones((), dtype=torch.float32, device="cuda"), requires_grad=False
    )
    method.kernel.process_weights_after_loading(layer)
    return layer


def run(tokens: int, warmup: int, iterations: int,
        fixture_dump: pathlib.Path | None = None,
        fixture_import: pathlib.Path | None = None) -> list[dict[str, object]]:
    torch.manual_seed(7)
    device = torch.device("cuda")
    q = torch.randn(tokens, 8, 128, dtype=torch.bfloat16, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(tokens, 8, 128, dtype=torch.float32, device=device),
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn(tokens, 24, 128, dtype=torch.bfloat16, device=device)
    g = torch.rand(tokens, 24, dtype=torch.float32, device=device)
    beta = torch.rand(tokens, 24, dtype=torch.float32, device=device)
    initial = torch.randn(1, 24, 128, 128, dtype=torch.float32, device=device)
    output = torch.empty(tokens, 24, 128, dtype=torch.bfloat16, device=device)
    final_state = torch.empty_like(initial)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64, device=device)

    conv_input = torch.randn(
        tokens, 5120, dtype=torch.bfloat16, device=device
    ).transpose(0, 1)
    conv_weight = torch.randn(5120, 4, dtype=torch.bfloat16, device=device)
    conv_state = torch.randn(1, 5120, 3, dtype=torch.bfloat16, device=device)
    query_start = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    cache_indices = torch.zeros(1, dtype=torch.int32, device=device)
    has_initial = torch.ones(1, dtype=torch.bool, device=device)
    chunks = math.ceil(tokens / 8)
    batch_ptr = torch.zeros(chunks, dtype=torch.int32, device=device)
    chunk_offsets = torch.arange(chunks, dtype=torch.int32, device=device)
    conv_metadata = SimpleNamespace(
        nums_dict={
            8: {
                "tot": chunks,
                "mlist": batch_ptr,
                "mlist_len": chunks,
                "offsetlist": chunk_offsets,
                "batch_ptr": batch_ptr,
                "token_chunk_offset_ptr": chunk_offsets,
            }
        },
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=chunk_offsets,
    )
    conv_result = SimpleNamespace(output=None)

    def causal_convolution() -> None:
        conv_result.output = causal_conv1d_fn(
            conv_input,
            conv_weight,
            None,
            conv_state,
            query_start,
            cache_indices=cache_indices,
            has_initial_state=has_initial,
            activation="silu",
            metadata=conv_metadata,
            validate_data=False,
        )

    convolution_result = measure(
        "vllm_causal_convolution_and_conv_state",
        causal_convolution,
        warmup,
        iterations,
    )
    causal_convolution()
    torch.cuda.synchronize()
    post_result = SimpleNamespace(outputs=None)
    a = torch.randn(tokens, 24, dtype=torch.bfloat16, device=device)
    b = torch.randn(tokens, 24, dtype=torch.bfloat16, device=device)
    a_log = torch.randn(24, dtype=torch.float32, device=device)
    dt_bias = torch.randn(24, dtype=torch.float32, device=device)

    def post_conv_prep() -> None:
        post_result.outputs = fused_post_conv_prep(
            conv_result.output.transpose(0, 1),
            a,
            b,
            a_log,
            dt_bias,
            num_k_heads=8,
            head_k_dim=128,
            head_v_dim=128,
            apply_l2norm=True,
            output_g_exp=False,
        )

    post_prep_result = measure(
        "vllm_post_conv_prep", post_conv_prep, warmup, iterations
    )

    def recurrence() -> None:
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output,
            output_state=final_state,
        )

    recurrence_result = measure(
        "flashinfer_recurrent_scan", recurrence, warmup, iterations
    )
    recurrence()
    torch.cuda.synchronize()
    reference_output = output.clone()
    reference_state = final_state.clone()
    recurrence()
    torch.cuda.synchronize()
    recurrence_result.update({
        "output_max_error": float((output - reference_output).abs().max()),
        "final_state_max_error": float(
            (final_state - reference_state).abs().max()
        ),
        "output_bytes": output.numel() * output.element_size(),
        "final_state_bytes": final_state.numel() * final_state.element_size(),
    })

    config = ModelOptNvFp4Config("NVFP4", True, None, [], 16)
    method = ModelOptNvFp4LinearMethod(config)
    fixture_manifest = None
    if fixture_import is None:
        qkvz_layer = nvfp4_layer(method, 2560, 8192)
        ba_layer = nvfp4_layer(method, 2560, 48)
    else:
        qkvz_layer, ba_layer, qkvz_input, ba_input, hidden, fixture_manifest = (
            import_projection_fixture(fixture_import, tokens, device)
        )
    output_layer = nvfp4_layer(method, 3072, 2560)
    if fixture_import is None:
        hidden = torch.randn(tokens, 2560, dtype=torch.bfloat16, device=device)
    qkvz_input = hidden if fixture_import is None else qkvz_input
    ba_input = hidden if fixture_import is None else ba_input
    projection_outputs = SimpleNamespace(qkvz=None, ba=None, output=None)

    def input_projection() -> None:
        projection_outputs.qkvz = method.apply(qkvz_layer, qkvz_input)
        projection_outputs.ba = method.apply(ba_layer, ba_input)

    with Fp4DispatchTrace() as input_trace:
        input_result = measure(
            "nvfp4_input_projection", input_projection, warmup, iterations
        )
    attribute_projection_backend(
        input_result, method, input_trace,
        projection_contract(tokens, warmup, iterations),
    )
    if fixture_import is None:
        input_result["data_identity"] = projection_data_identity(
            hidden, qkvz_layer, ba_layer
        )
        if fixture_dump is not None:
            write_projection_fixture(fixture_dump, tokens, hidden, qkvz_layer, ba_layer)
    else:
        input_result["fixture_manifest"] = fixture_manifest
        input_result["comparator_contract"].update({
            "scope": "one_cuda_graph_replay_of_two_prequantized_modelopt_apply_calls",
            "activation_quantizations": 0,
            "includes": ["qkvz_gemm", "ba_gemm"],
            "weights": fixture_manifest["provenance"],
        })

    norm_input = torch.randn(tokens * 24, 128, dtype=torch.bfloat16,
                             device=device)
    norm_gate = torch.randn_like(norm_input)
    norm_weight = torch.ones(128, dtype=torch.bfloat16, device=device)
    norm_output = torch.empty_like(norm_input)

    def gated_norm() -> None:
        layer_norm_fwd(
            norm_input,
            norm_weight,
            None,
            1e-6,
            z=norm_gate,
            out=norm_output,
            group_size=128,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="silu",
        )

    norm_result = measure("gated_rmsnorm", gated_norm, warmup, iterations)

    flattened_norm = norm_output.view(tokens, 3072)

    def output_projection() -> None:
        projection_outputs.output = method.apply(output_layer, flattened_norm)

    with Fp4DispatchTrace() as output_trace:
        output_result = measure(
            "nvfp4_output_projection", output_projection, warmup, iterations
        )
    output_contract = projection_contract(tokens, warmup, iterations)
    output_contract.update({
        "scope": "one_cuda_graph_replay_of_one_modelopt_apply_call",
        "logical_mnk": [[tokens, 2560, 3072]],
        "physical_mnk": [[tokens, 2560, 3072]],
        "activation_quantizations": 1,
        "gemms": 1,
        "includes": ["output_dynamic_quant", "output_gemm"],
    })
    attribute_projection_backend(
        output_result, method, output_trace, output_contract
    )

    inactive_state = torch.empty_like(final_state)

    def state_publication() -> None:
        inactive_state.copy_(final_state)

    publication_result = measure(
        "recurrent_state_publication", state_publication, warmup, iterations
    )
    for result in (
        input_result,
        convolution_result,
        post_prep_result,
        recurrence_result,
        norm_result,
        output_result,
        publication_result,
    ):
        result["tokens"] = tokens
    return [
        input_result,
        convolution_result,
        post_prep_result,
        recurrence_result,
        norm_result,
        output_result,
        publication_result,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[300, 8192])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    fixtures = parser.add_mutually_exclusive_group()
    fixtures.add_argument("--dump-projection-fixtures", type=pathlib.Path)
    fixtures.add_argument("--import-projection-fixtures", type=pathlib.Path)
    args = parser.parse_args()
    script_path = pathlib.Path(__file__).resolve()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "flashinfer": flashinfer.__version__,
                "vllm": vllm.__version__,
                "profiler_path": str(script_path),
                "profiler_sha256": sha256_file(script_path),
                "container_hostname": os.uname().nodename,
            },
            sort_keys=True,
        )
    )
    for tokens in args.tokens:
        for result in run(
            tokens, args.warmup, args.iterations,
            args.dump_projection_fixtures, args.import_projection_fixtures,
        ):
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
