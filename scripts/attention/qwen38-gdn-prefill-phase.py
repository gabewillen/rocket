#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time the pinned FlashInfer recurrence at Qwen3.8 TP2 prefill shapes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from types import SimpleNamespace
from typing import Callable

import torch
import flashinfer
import vllm
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4LinearMethod,
)
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


def run(tokens: int, warmup: int, iterations: int) -> list[dict[str, object]]:
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
    qkvz_layer = nvfp4_layer(method, 2560, 8192)
    ba_layer = nvfp4_layer(method, 2560, 48)
    output_layer = nvfp4_layer(method, 3072, 2560)
    hidden = torch.randn(tokens, 2560, dtype=torch.bfloat16, device=device)
    projection_outputs = SimpleNamespace(qkvz=None, ba=None, output=None)

    def input_projection() -> None:
        projection_outputs.qkvz = method.apply(qkvz_layer, hidden)
        projection_outputs.ba = method.apply(ba_layer, hidden)

    input_result = measure(
        "nvfp4_input_projection", input_projection, warmup, iterations
    )

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

    output_result = measure(
        "nvfp4_output_projection", output_projection, warmup, iterations
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
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "flashinfer": flashinfer.__version__,
                "vllm": vllm.__version__,
            },
            sort_keys=True,
        )
    )
    for tokens in args.tokens:
        for result in run(tokens, args.warmup, args.iterations):
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
