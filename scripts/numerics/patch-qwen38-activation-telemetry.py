#!/usr/bin/env python3
"""Add bounded, opt-in Qwen3.8 activation telemetry to NVIDIA vLLM."""

import argparse
from pathlib import Path

IMPORT_ANCHOR = "from itertools import islice\n\nimport torch\n"
IMPORT_PATCH = "from itertools import islice\nimport json\nimport os\n\nimport torch\n"
CLASS_ANCHOR = "class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):\n"
HELPER = r'''_ROCKET_CALIBRATION_MAXIMA = {}
_ROCKET_TELEMETRY_CALLS = {}
_ROCKET_ROUTER_PREFILL_BARRIERS = set()
_ROCKET_ROUTER_DIAGNOSTIC_SAMPLES = []
_ROCKET_TELEMETRY_SCHEMA = "rocket.qwen38.activation-telemetry.v4"


def _rocket_install_linear_load_trace():
    if os.getenv("ROCKET_QWEN38_LOAD_TRACE") != "1":
        return
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    if getattr(MergedColumnParallelLinear, "_rocket_load_trace", False):
        return
    original = MergedColumnParallelLinear.load_weights

    def traced(self, weights):
        def rows():
            for name, value in weights:
                print(
                    "ROCKET_QWEN38_LINEAR_LOAD\t"
                    + json.dumps(
                        {
                            "prefix": self.prefix,
                            "name": name,
                            "quant_method": type(self.quant_method).__name__,
                            "parameters": sorted(self._parameters),
                            "shard_id": getattr(value, "shard_id", None),
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                yield name, value

        return original(self, rows())

    MergedColumnParallelLinear.load_weights = traced
    MergedColumnParallelLinear._rocket_load_trace = True


def _rocket_tensor(value, index=None):
    if index is not None:
        if not isinstance(value, (tuple, list)) or len(value) <= index:
            return None
        value = value[index]
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    return None


def _rocket_summary(value):
    tensor = _rocket_tensor(value)
    if tensor is None or tensor.numel() == 0:
        return None
    sample_limit = min(
        16384,
        max(64, int(os.getenv("ROCKET_NVFP4_SAMPLE_ELEMENTS", "2048"))),
    )
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() > sample_limit:
        stride = max(1, flat.numel() // sample_limit)
        flat = flat[::stride][:sample_limit]
    absolute = flat.abs()
    quantiles = torch.quantile(
        absolute, torch.tensor([0.5, 0.9, 0.99], device=absolute.device)
    )
    # Fixed log2 magnitude bins keep records comparable and cardinality bounded.
    edges = torch.tensor(
        [0.0, 2.0**-12, 2.0**-8, 2.0**-4, 1.0, 4.0, 16.0, 64.0, 256.0],
        device=absolute.device,
    )
    bins = torch.bucketize(absolute, edges, right=True)
    histogram = torch.bincount(bins, minlength=edges.numel() + 1).cpu().tolist()
    return {
        "source_numel": tensor.numel(),
        "sample_numel": flat.numel(),
        "absmax": float(absolute.max().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "abs_p50": float(quantiles[0].item()),
        "abs_p90": float(quantiles[1].item()),
        "abs_p99": float(quantiles[2].item()),
        "histogram_log2": histogram,
    }


def _rocket_query_widths(rows):
    from vllm.forward_context import get_forward_context

    context = get_forward_context()
    metadata = context.attn_metadata
    pending = list(metadata) if isinstance(metadata, (list, tuple)) else [metadata]
    candidates = set()
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
            continue
        if isinstance(value, (list, tuple)):
            pending.extend(value)
            continue
        offsets = getattr(value, "query_start_loc", None)
        actual = getattr(value, "num_actual_tokens", None)
        if not isinstance(offsets, torch.Tensor) or actual != rows:
            continue
        boundaries = tuple(int(item) for item in offsets.detach().cpu().tolist())
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != rows
            or any(right <= left for left, right in zip(boundaries, boundaries[1:]))
        ):
            continue
        candidates.add(
            tuple(right - left for left, right in zip(boundaries, boundaries[1:]))
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"router cohort query widths are missing or ambiguous for {rows} rows"
        )
    widths = next(iter(candidates))
    return widths


def _rocket_router_cohort(tensor, top_k, channel):
    expected_top_k = 10
    if top_k != expected_top_k:
        raise RuntimeError(
            f"Qwen3.8 router top-k contract changed: got {top_k}, "
            f"expected {expected_top_k}"
        )
    rows = int(tensor.shape[0])
    sequences = int(os.getenv("ROCKET_ROUTER_SEQUENCES", "0"))
    verify_width = int(os.getenv("ROCKET_ROUTER_VERIFY_WIDTH", "0"))
    cohort = os.getenv("ROCKET_ROUTER_COHORT", "")
    rank_text = os.getenv("ROCKET_ROUTER_RANK", "")
    cache_barrier = os.getenv("ROCKET_ROUTER_CACHE_BARRIER", "")
    cache_block_size = int(os.getenv("ROCKET_ROUTER_CACHE_BLOCK_SIZE", "0"))
    if (
        not cohort
        and not rank_text
        and sequences == 0
        and verify_width == 0
        and not cache_barrier
        and cache_block_size == 0
    ):
        return None
    if sequences < 1 or verify_width < 1:
        raise RuntimeError(
            "router cohort sequences and verify_width must both be positive"
        )
    rank = int(rank_text or "-1")
    if rank not in (0, 1):
        raise RuntimeError(f"router cohort rank must be 0 or 1, got {rank}")
    if not cohort:
        raise RuntimeError("ROCKET_ROUTER_COHORT is required for router telemetry")
    if cache_barrier and sequences != 16:
        raise RuntimeError("router cache barrier is only valid for c16 telemetry")
    if sequences == 16 and not (
        (cache_barrier == "two-cache-pages-v2" and cache_block_size == 3216)
        or (cache_barrier == "fresh-prefill-t300-v1" and cache_block_size == 0)
    ):
        raise RuntimeError(
            "c16 router cohort requires an authenticated cache or fresh-prefill barrier"
        )

    widths = _rocket_query_widths(rows)
    if any(width > verify_width for width in widths):
        if cache_barrier == "fresh-prefill-t300-v1":
            barrier_key = (channel, cohort)
            if len(widths) != sequences or any(width != 300 for width in widths):
                raise RuntimeError(
                    "c16 T300 prefill barrier must contain all 16 fresh requests"
                )
            if barrier_key in _ROCKET_ROUTER_PREFILL_BARRIERS:
                raise RuntimeError(
                    "c16 T300 prefill barrier may occur exactly once per layer"
                )
            _ROCKET_ROUTER_PREFILL_BARRIERS.add(barrier_key)
            return None
        if cache_barrier == "two-cache-pages-v2":
            raise RuntimeError(
                "c16 cache barrier violated by post-gate prefill or oversized width"
            )
        # Cohort prefill and mixed prefill/decode batches are outside the
        # verifier workload. Their scheduler-owned widths remain larger than
        # target plus the configured speculative width.
        return None
    if len(widths) > sequences:
        raise RuntimeError(
            f"router cohort scheduled {len(widths)} requests above c{sequences}"
        )
    if sequences == 16:
        barrier_key = (channel, cohort)
        speculative_rows = sum(width - 1 for width in widths)
        if speculative_rows == 0:
            if cache_barrier == "fresh-prefill-t300-v1":
                raise RuntimeError(
                    "c16 T300 emitted an unexpected target-only call after prefill"
                )
            if len(widths) != sequences or any(width != 1 for width in widths):
                raise RuntimeError(
                    "c16 target-only cache prefill must contain all 16 requests"
                )
            if barrier_key in _ROCKET_ROUTER_PREFILL_BARRIERS:
                raise RuntimeError(
                    "c16 target-only cache prefill may occur exactly once per layer"
                )
            _ROCKET_ROUTER_PREFILL_BARRIERS.add(barrier_key)
            return None
        if barrier_key not in _ROCKET_ROUTER_PREFILL_BARRIERS:
            raise RuntimeError(
                "c16 verifier telemetry started before the all-request cache prefill"
            )
    row_offsets = [0]
    for width in widths:
        row_offsets.append(row_offsets[-1] + width)
    logits = tensor.detach().float()
    probabilities = torch.softmax(logits, dim=-1)
    weights, selected = torch.topk(probabilities, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    selected_rows = selected.cpu().tolist()
    weight_rows = weights.cpu().tolist()
    route_rows = []
    for sequence, (start, width) in enumerate(zip(row_offsets, widths)):
        for position in range(width):
            row = start + position
            route_rows.append(
                {
                    "row": row,
                    "sequence": sequence,
                    "position": position,
                    "position_kind": (
                        "target" if position == 0 else "speculative"
                    ),
                    "expert_ids": [int(item) for item in selected_rows[row]],
                    "weights": [float(item) for item in weight_rows[row]],
                }
            )
    return {
        "cohort": cohort,
        "rank": rank,
        "cohort_sequences": sequences,
        "sequences": len(widths),
        "verify_width": verify_width,
        "request_widths": list(widths),
        "row_offsets": row_offsets,
        "route_top_k": top_k,
        "route_layout": "sequence_major",
        "route_rows": route_rows,
    }


def _rocket_emit(name, kind, value, *, output_index=None, top_k=None):
    count = _ROCKET_TELEMETRY_CALLS.get(name, 0) + 1
    _ROCKET_TELEMETRY_CALLS[name] = count
    max_emissions = min(
        12,
        max(1, int(os.getenv("ROCKET_NVFP4_MAX_EMISSIONS", "4"))),
    )
    # Non-router channels emit at calls 1, 2, 4, ... and then stop. Router
    # cohorts retain the first bounded set of scheduler-width verifier calls.
    if top_k is None and (count & (count - 1) or count.bit_length() > max_emissions):
        return
    tensor = _rocket_tensor(value, output_index)
    summary = _rocket_summary(tensor)
    if summary is None:
        return
    record = {
        "schema": _ROCKET_TELEMETRY_SCHEMA,
        "channel": name,
        "kind": kind,
        "call": count,
        **summary,
    }
    if top_k is not None and tensor.ndim > 0 and tensor.shape[-1] > 0:
        router_cohort = _rocket_router_cohort(tensor, top_k, name)
        if router_cohort is None:
            return
        cohort_key = (name, router_cohort["cohort"])
        cohort_count = _ROCKET_TELEMETRY_CALLS.get(cohort_key, 0) + 1
        _ROCKET_TELEMETRY_CALLS[cohort_key] = cohort_count
        router_limit = min(
            16,
            max(1, int(os.getenv("ROCKET_ROUTER_MAX_COHORTS", "2"))),
        )
        if cohort_count > router_limit:
            return
        selected = torch.tensor(
            [
                expert_id
                for row in router_cohort["route_rows"]
                for expert_id in row["expert_ids"]
            ],
            dtype=torch.int64,
            device=tensor.device,
        )
        expert_counts = torch.bincount(selected, minlength=tensor.shape[-1])
        ranked_counts, ranked_ids = torch.topk(
            expert_counts, min(16, expert_counts.numel())
        )
        ranked = [
            (int(selections), int(expert_id))
            for selections, expert_id in zip(
                ranked_counts.cpu().tolist(), ranked_ids.cpu().tolist()
            )
            if selections > 0
        ]
        record["top_experts"] = [
            {"expert_id": expert_id, "selections": selections}
            for selections, expert_id in ranked
        ]
        record["selected_expert_count"] = int(selected.numel())
        record["cohort_call"] = cohort_count
        record.update(router_cohort)
        if len(_ROCKET_ROUTER_DIAGNOSTIC_SAMPLES) < 4:
            _ROCKET_ROUTER_DIAGNOSTIC_SAMPLES.append(
                {
                    "channel": name,
                    "cohort_call": cohort_count,
                    "rank": router_cohort["rank"],
                    "route_rows": len(router_cohort["route_rows"]),
                    "request_widths": router_cohort["request_widths"][:16],
                    "row_offsets": router_cohort["row_offsets"][:17],
                }
            )
    print("ROCKET_NVFP4_TELEMETRY\t" + json.dumps(record, sort_keys=True), flush=True)


def _rocket_pre_hook(name, kind, *, legacy=False):
    def capture(module, inputs):
        del module
        value = _rocket_tensor(inputs)
        if value is None:
            return
        _rocket_emit(name + ".input", kind + "_input", value)
        if legacy:
            maximum = float(value.detach().float().abs().amax().item())
            if maximum > _ROCKET_CALIBRATION_MAXIMA.get(name, 0.0):
                _ROCKET_CALIBRATION_MAXIMA[name] = maximum
                print(
                    f"ROCKET_NVFP4_CALIBRATION\t{name}\t{maximum:.9g}",
                    flush=True,
                )
    return capture


def _rocket_post_hook(name, kind, *, output_index=None, top_k=None):
    def capture(module, inputs, output):
        del module, inputs
        _rocket_emit(
            name + ".output", kind + "_output", output,
            output_index=output_index, top_k=top_k,
        )
    return capture


def _rocket_hook_projection(module, name, kind, *, legacy=False):
    if module is None:
        return
    module.register_forward_pre_hook(_rocket_pre_hook(name, kind, legacy=legacy))
    module.register_forward_hook(_rocket_post_hook(name, kind))


def _rocket_install_activation_telemetry(model):
    if os.getenv("ROCKET_NVFP4_CALIBRATE") != "1":
        return
    top_k = int(getattr(model.config, "num_experts_per_tok", 0))
    if top_k != 10:
        raise RuntimeError(
            f"Qwen3.8 config num_experts_per_tok changed: got {top_k}, expected 10"
        )
    for layer_index, layer in enumerate(model.layers):
        prefix = f"layer.{layer_index}"
        attention = getattr(layer, "linear_attn", None)
        if attention is not None:
            for projection_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
                _rocket_hook_projection(
                    getattr(attention, projection_name, None),
                    f"{prefix}.linear_attn.{projection_name}",
                    "linear_attention_projection",
                    legacy=True,
                )
            attention.register_forward_hook(
                _rocket_post_hook(f"{prefix}.linear_attn", "linear_attention")
            )
            recurrent = getattr(attention, "chunk_gated_delta_rule", None)
            if recurrent is not None and hasattr(recurrent, "register_forward_hook"):
                recurrent.register_forward_hook(
                    _rocket_post_hook(
                        f"{prefix}.linear_attn.recurrent_state",
                        "recurrent_state",
                        output_index=1,
                    )
                )

        full_attention = getattr(layer, "self_attn", None)
        if full_attention is not None:
            _rocket_hook_projection(
                getattr(full_attention, "qkv_proj", None),
                f"{prefix}.full_attn.qkv_proj",
                "full_attention_qkv_projection",
            )
            _rocket_hook_projection(
                getattr(full_attention, "o_proj", None),
                f"{prefix}.full_attn.o_proj",
                "full_attention_output_projection",
            )
            full_attention.register_forward_hook(
                _rocket_post_hook(f"{prefix}.full_attn", "full_attention")
            )

        ple = getattr(layer, "ple", None)
        if ple is not None:
            embedding = getattr(ple, "ple_embedding", None)
            if embedding is not None:
                embedding.register_forward_hook(
                    _rocket_post_hook(f"{prefix}.ple.embedding", "ple_embedding")
                )
            ple.register_forward_hook(
                _rocket_post_hook(f"{prefix}.ple", "ple")
            )

        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            _rocket_hook_projection(
                gate, f"{prefix}.router.gate", "router_logits"
            )
            # A second post-hook adds bounded top-k expert identities.
            gate.register_forward_hook(
                _rocket_post_hook(
                    f"{prefix}.router.topk", "router_topk", top_k=top_k
                )
            )


'''
INIT_ANCHOR = "        enable_qwen38next_low_latency_gemm(self, self.model_config.dtype)\n"
INIT_PATCH = (
    INIT_ANCHOR
    + "        _rocket_install_linear_load_trace()\n"
    + "        _rocket_install_activation_telemetry(self.model)\n"
)
def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise SystemExit(f"vLLM source drift: expected one {label} anchor")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_py", type=Path)
    args = parser.parse_args()
    source = args.model_py.read_text(encoding="utf-8")
    if "ROCKET_NVFP4_TELEMETRY" in source:
        raise SystemExit("already patched")
    source = replace_once(source, IMPORT_ANCHOR, IMPORT_PATCH, "import")
    source = replace_once(source, CLASS_ANCHOR, HELPER + CLASS_ANCHOR, "class")
    source = replace_once(source, INIT_ANCHOR, INIT_PATCH, "initialization")
    args.model_py.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
