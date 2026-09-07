#!/usr/bin/env python3
"""Checkpoint-derived memory-bandwidth roofline for Qwen3.8-Flash-Next."""

from __future__ import annotations

import argparse
import json
import math
import struct
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping


DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
MAX_HEADER_BYTES = 64 * 2**20
MAX_INDEX_BYTES = 128 * 2**20
NVFP4_BYTES_PER_BF16_BYTE = 0.28125
NVFP4_BYTES_PER_FP8_BYTE = 0.5625

BASE_DENSE_FAMILIES = (
    "base_linear_attention",
    "base_full_attention",
    "base_hyperconnection",
    "base_shared_expert",
    "base_routers",
    "base_ple_dense",
    "base_other",
)
MTP_DENSE_FAMILIES = (
    "mtp_attention",
    "mtp_hyperconnection",
    "mtp_shared_expert",
    "mtp_routers",
    "mtp_input_projection",
    "mtp_other",
)
BF16_QUANTIZATION_FAMILIES = frozenset(
    BASE_DENSE_FAMILIES + MTP_DENSE_FAMILIES + ("embedding_table", "lm_head")
)
REQUIRED_FAMILIES = {
    "base_routed_experts",
    "base_linear_attention",
    "base_full_attention",
    "base_hyperconnection",
    "base_shared_expert",
    "base_routers",
    "base_ple_dense",
    "embedding_table",
    "lm_head",
    "mtp_routed_experts",
    "mtp_attention",
    "mtp_hyperconnection",
    "mtp_shared_expert",
    "mtp_routers",
    "mtp_input_projection",
    "ple_table",
}


def tensor_bytes(meta: Mapping[str, object]) -> int:
    """Return stored payload bytes, rejecting malformed or reversed offsets."""
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("every tensor needs two data_offsets")
    start, end = offsets
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError("tensor data_offsets must be ordered non-negative integers")
    return end - start


def tensor_family(name: str) -> str:
    """Map a checkpoint tensor name to one traffic/precision family."""
    if name.startswith("model.visual"):
        return "vision_excluded"
    if name.startswith("mtp."):
        if ".mlp.experts." in name:
            return "mtp_routed_experts"
        if ".self_attn." in name:
            return "mtp_attention"
        if ".mlp.shared_expert." in name:
            return "mtp_shared_expert"
        if ".mlp.gate." in name or ".mlp.shared_expert_gate." in name:
            return "mtp_routers"
        if "hyper_connection" in name:
            return "mtp_hyperconnection"
        if name.startswith("mtp.fc_") or name.startswith("mtp.pre_fc_norm_"):
            return "mtp_input_projection"
        return "mtp_other"
    if ".ple.ple_embedding.ngram_embedding." in name:
        return "ple_table"
    if ".mlp.experts." in name:
        return "base_routed_experts"
    if name == "model.language_model.embed_tokens.weight":
        return "embedding_table"
    if name == "lm_head.weight":
        return "lm_head"
    if ".linear_attn." in name:
        return "base_linear_attention"
    if ".self_attn." in name:
        return "base_full_attention"
    if ".mlp.shared_expert." in name:
        return "base_shared_expert"
    if ".mlp.gate." in name or ".mlp.shared_expert_gate." in name:
        return "base_routers"
    if "hyper_connection" in name:
        return "base_hyperconnection"
    if ".ple." in name:
        return "base_ple_dense"
    if name.startswith("model.language_model."):
        return "base_other"
    return "other"


def inventory_details(
    headers: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Return copied inventory summaries derived from borrowed headers.

    The function performs no I/O or mutation. Each tensor is assigned exactly
    once. Missing Qwen3.8 execution families raise ValueError.
    """
    family_bytes: Counter[str] = Counter()
    family_tensors: Counter[str] = Counter()
    dtype_bytes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for name, meta in headers.items():
        if not isinstance(name, str) or not isinstance(meta, Mapping):
            raise ValueError("headers must map tensor names to metadata objects")
        dtype = meta.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"tensor {name!r} needs a string dtype")
        size = tensor_bytes(meta)
        family = tensor_family(name)
        if family == "other":
            raise ValueError(f"checkpoint family drift, unknown tensor: {name}")
        family_bytes[family] += size
        family_tensors[family] += 1
        dtype_bytes[family][dtype] += size

    missing = sorted(family for family in REQUIRED_FAMILIES if not family_tensors[family])
    if missing:
        raise ValueError("checkpoint family drift, missing: " + ", ".join(missing))
    return {
        family: {
            "bytes": family_bytes[family],
            "tensor_count": family_tensors[family],
            "dtype_bytes": dict(sorted(dtype_bytes[family].items())),
        }
        for family in sorted(family_bytes)
    }


def inventory(headers: Mapping[str, Mapping[str, object]]) -> dict[str, int]:
    """Return stored bytes by family for compatibility with earlier callers."""
    return {
        family: int(detail["bytes"])
        for family, detail in inventory_details(headers).items()
    }


def expected_union(experts: int, top_k: int, token_positions: int) -> float:
    """Expected expert union for independent, uniform top-k token routes."""
    if not (0 < top_k <= experts and token_positions > 0):
        raise ValueError("require 0 < top_k <= experts and token_positions > 0")
    return experts * (1.0 - (1.0 - top_k / experts) ** token_positions)


def _read_safetensors_header(path: Path) -> dict[str, dict[str, object]]:
    """Read and validate one bounded local safetensors header, never its payload."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= MAX_HEADER_BYTES:
            raise ValueError(f"safetensors header outside size bound: {path.name}")
        raw_header = stream.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError(f"truncated safetensors header: {path.name}")
    decoded = json.loads(raw_header)
    if not isinstance(decoded, dict):
        raise ValueError(f"safetensors header must be an object: {path.name}")
    decoded.pop("__metadata__", None)
    for name, meta in decoded.items():
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise ValueError(f"invalid tensor metadata in {path.name}")
        tensor_bytes(meta)
    payload_end = max(
        (int(meta["data_offsets"][1]) for meta in decoded.values()), default=0
    )
    if path.stat().st_size < 8 + header_size + payload_end:
        raise ValueError(f"truncated safetensors payload: {path.name}")
    return decoded


def local_headers(model_dir: Path) -> dict[str, dict[str, object]]:
    """Read headers for a complete local sharded checkpoint without payload I/O."""
    root = model_dir.resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.stat().st_size > MAX_INDEX_BYTES:
        raise ValueError("safetensors index outside size bound")
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index needs a non-empty weight_map")
    if any(not isinstance(name, str) or not isinstance(shard, str)
           for name, shard in weight_map.items()):
        raise ValueError("safetensors weight_map entries must be strings")

    headers: dict[str, dict[str, object]] = {}
    for filename in sorted(set(weight_map.values())):
        relative_shard = Path(filename)
        if relative_shard.is_absolute() or ".." in relative_shard.parts:
            raise ValueError(f"checkpoint shard escapes model directory: {filename}")
        shard_path = root / relative_shard
        shard = _read_safetensors_header(shard_path)
        overlap = headers.keys() & shard.keys()
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)[:3]}")
        headers.update(shard)
    if set(headers) != set(weight_map):
        missing = len(set(weight_map) - set(headers))
        extra = len(set(headers) - set(weight_map))
        raise ValueError(f"index/header tensor mismatch: {missing} missing, {extra} extra")
    return headers


def remote_headers(repo: str, revision: str) -> dict[str, dict[str, object]]:
    """Fetch bounded safetensors headers from an explicitly pinned Hub revision."""
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    with urllib.request.urlopen(base + "model.safetensors.index.json") as response:
        index = json.load(response)
    headers: dict[str, dict[str, object]] = {}
    for filename in sorted(set(index["weight_map"].values())):
        request = urllib.request.Request(
            base + filename, headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request) as response:
            prefix = response.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated remote safetensors prefix: {filename}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= MAX_HEADER_BYTES:
            raise ValueError(f"remote safetensors header outside size bound: {filename}")
        request = urllib.request.Request(
            base + filename,
            headers={"Range": f"bytes=8-{7 + header_size}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request) as response:
            raw_header = response.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"truncated remote safetensors header: {filename}")
        shard = json.loads(raw_header)
        shard.pop("__metadata__", None)
        overlap = headers.keys() & shard.keys()
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)[:3]}")
        headers.update(shard)
    return headers


def _validate_model_inputs(
    concurrency: int,
    experts: int,
    top_k: int,
    nodes: int,
    gb_s_per_node: float,
    ple_rows: int,
    hidden_size: int,
    ple_bytes: int,
    embedding_bytes: int,
    route: str,
    mtp_proposals: int,
    accepted_tokens_per_step: float,
) -> None:
    """Validate the pure roofline contract at its public boundary."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if not 0 < top_k <= experts:
        raise ValueError("require 0 < top_k <= experts")
    if nodes <= 0 or not math.isfinite(gb_s_per_node) or gb_s_per_node <= 0:
        raise ValueError("nodes and GB/s per node must be positive")
    if ple_rows < 0 or hidden_size <= 0 or ple_bytes <= 0 or embedding_bytes <= 0:
        raise ValueError("row counts and element sizes are outside their valid ranges")
    if route not in {"uniform", "same"}:
        raise ValueError("route must be 'uniform' or 'same'")
    if mtp_proposals < 0:
        raise ValueError("MTP proposal count must be non-negative")
    if (not math.isfinite(accepted_tokens_per_step)
            or not 1.0 <= accepted_tokens_per_step <= mtp_proposals + 1.0):
        raise ValueError("accepted tokens/step must be within [1, MTP proposals + 1]")


def _expert_union(experts: int, top_k: int, positions: int, route: str) -> float:
    return float(top_k) if route == "same" else expected_union(experts, top_k, positions)


def _source_family(runtime_family: str) -> str | None:
    """Return the stored family supplying one runtime traffic component."""
    aliases = {
        "verifier_lm_head": "lm_head",
        "mtp_lm_head": "lm_head",
        "verifier_embedding_rows": "embedding_table",
        "mtp_embedding_rows": "embedding_table",
        "verifier_ple_rows": "ple_table",
    }
    if runtime_family in aliases:
        return aliases[runtime_family]
    if runtime_family.startswith(("base_", "mtp_")):
        return runtime_family
    return None


def _rank_levers(
    by_family: Mapping[str, int | float],
    details: Mapping[str, Mapping[str, object]],
    total: int | float,
    bandwidth: float,
    accepted_per_step: float,
    non_spec_accepted_per_step: float,
    non_spec_total: int | float,
    ple_removal_bytes: int | float,
) -> list[dict[str, object]]:
    """Rank isolated traffic reductions without asserting their quality cost."""
    active_by_source: Counter[str] = Counter()
    for runtime_family, active_bytes in by_family.items():
        source_family = _source_family(runtime_family)
        if source_family is not None:
            active_by_source[source_family] += active_bytes

    levers: list[dict[str, object]] = []
    for family, active_bytes in active_by_source.items():
        detail = details.get(family)
        if detail is None:
            continue
        stored_bytes = int(detail["bytes"])
        dtype_bytes = detail["dtype_bytes"]
        bf16_fraction = (
            int(dtype_bytes.get("BF16", 0)) / stored_bytes
            if family in BF16_QUANTIZATION_FAMILIES
            else 0.0
        )
        savings = active_bytes * bf16_fraction * (1.0 - NVFP4_BYTES_PER_BF16_BYTE)
        if savings > 0:
            levers.append({
                "action": "quantize_bf16_to_checkpoint_nvfp4_layout",
                "family": family,
                "estimated_bytes_removed_per_step": savings,
                "estimated_resident_bytes_removed": (
                    int(dtype_bytes.get("BF16", 0))
                    * (1.0 - NVFP4_BYTES_PER_BF16_BYTE)
                ),
                "estimate_assumption": (
                    "BF16 payload becomes 4-bit packed weights plus one FP8 scale per "
                    "16 values: 0.5625 bytes/value versus 2 bytes/value"
                ),
            })
        if family in {"mtp_routed_experts", "ple_table"}:
            fp8_fraction = int(dtype_bytes.get("F8_E4M3", 0)) / stored_bytes
            savings = active_bytes * fp8_fraction * (1.0 - NVFP4_BYTES_PER_FP8_BYTE)
            if savings > 0:
                levers.append({
                    "action": "quantize_fp8_to_checkpoint_nvfp4_layout",
                    "family": family,
                    "estimated_bytes_removed_per_step": savings,
                    "estimated_resident_bytes_removed": (
                        int(dtype_bytes.get("F8_E4M3", 0))
                        * (1.0 - NVFP4_BYTES_PER_FP8_BYTE)
                    ),
                    "estimate_assumption": (
                        "FP8 payload becomes 4-bit packed weights plus one FP8 scale "
                        "per 16 values: 0.5625 bytes/value versus 1 byte/value"
                    ),
                })
    if non_spec_total < total:
        levers.append({
            "action": "remove_mtp_speculation",
            "family": "mtp_and_wider_verifier",
            "estimated_bytes_removed_per_step": total - non_spec_total,
            "estimated_resident_bytes_removed": sum(
                int(detail["bytes"])
                for family, detail in details.items()
                if family.startswith("mtp_")
            ),
            "estimate_assumption": (
                "proposal count becomes zero and accepted tokens/step becomes one"
            ),
            "accepted_positions_per_step_after": non_spec_accepted_per_step,
        })
    if ple_removal_bytes > 0:
        levers.append({
            "action": "remove_ple",
            "family": "base_ple_dense_and_rows",
            "estimated_bytes_removed_per_step": ple_removal_bytes,
            "estimated_resident_bytes_removed": (
                int(details["base_ple_dense"]["bytes"])
                + int(details["ple_table"]["bytes"])
            ),
            "estimate_assumption": "PLE projections and verifier PLE row gathers are bypassed",
        })

    levers.sort(
        key=lambda item: (
            -float(item["estimated_bytes_removed_per_step"]),
            str(item["action"]),
            str(item["family"]),
        )
    )
    for rank, lever in enumerate(levers, start=1):
        savings = float(lever["estimated_bytes_removed_per_step"])
        reduced_total = total - savings
        accepted_after = float(
            lever.get("accepted_positions_per_step_after", accepted_per_step)
        )
        lever.update({
            "rank": rank,
            "estimated_step_traffic_share_removed": savings / total,
            "estimated_step_bytes_after": reduced_total,
            "estimated_aggregate_accepted_tokens_per_s_after": (
                bandwidth * accepted_after / reduced_total
            ),
            "quality_effect": "unmeasured",
        })
    return levers


def model(
    headers: Mapping[str, Mapping[str, object]],
    batch: int,
    experts: int,
    top_k: int,
    nodes: int,
    gb_s_per_node: float,
    ple_rows: int,
    hidden_size: int,
    ple_bytes: int,
    embedding_bytes: int,
    route: str,
    mtp_proposals: int,
    accepted_tokens_per_step: float,
) -> dict[str, object]:
    """Compute one concurrency point without I/O or input mutation.

    mtp_proposals is vLLM's num_speculative_tokens. The verifier evaluates
    mtp_proposals + 1 positions in one target-model pass. Each proposal costs
    one sequential MTP-model pass and one shared LM-head pass.
    accepted_tokens_per_step includes the verifier's guaranteed token and is
    therefore in [1, mtp_proposals + 1].
    """
    _validate_model_inputs(
        batch, experts, top_k, nodes, gb_s_per_node, ple_rows, hidden_size,
        ple_bytes, embedding_bytes, route, mtp_proposals,
        accepted_tokens_per_step,
    )
    details = inventory_details(headers)
    sizes = {family: int(detail["bytes"]) for family, detail in details.items()}
    verifier_positions_per_stream = mtp_proposals + 1
    verifier_positions = batch * verifier_positions_per_stream
    verifier_union = _expert_union(experts, top_k, verifier_positions, route)
    mtp_continuation_union = (
        _expert_union(experts, top_k, batch, route) if mtp_proposals else 0.0
    )
    mtp_first_union = verifier_union if mtp_proposals else 0.0

    by_family: dict[str, int | float] = {
        family: sizes.get(family, 0) for family in BASE_DENSE_FAMILIES
    }
    by_family["base_routed_experts"] = (
        sizes["base_routed_experts"] * verifier_union / experts
    )
    by_family["verifier_lm_head"] = sizes["lm_head"]
    by_family["verifier_embedding_rows"] = (
        verifier_positions * hidden_size * embedding_bytes
    )
    by_family["verifier_ple_rows"] = (
        verifier_positions * hidden_size * ple_rows * ple_bytes
    )
    by_family.update({
        family: mtp_proposals * sizes.get(family, 0)
        for family in MTP_DENSE_FAMILIES
    })
    by_family["mtp_routed_experts"] = (
        sizes["mtp_routed_experts"]
        * (
            mtp_first_union
            + max(0, mtp_proposals - 1) * mtp_continuation_union
        )
        / experts
    )
    by_family["mtp_lm_head"] = mtp_proposals * sizes["lm_head"]
    by_family["mtp_embedding_rows"] = (
        (
            verifier_positions
            + max(0, mtp_proposals - 1) * batch
            if mtp_proposals
            else 0
        )
        * hidden_size
        * embedding_bytes
    )

    total = sum(by_family.values())
    if total <= 0:
        raise ValueError("checkpoint-derived step traffic must be positive")
    bandwidth = nodes * gb_s_per_node * 1e9
    steps_s = bandwidth / total
    aggregate_tokens_s = steps_s * batch * accepted_tokens_per_step
    traffic_ranking = [
        {
            "rank": rank,
            "family": family,
            "bytes_per_step": value,
            "traffic_share": value / total,
        }
        for rank, (family, value) in enumerate(
            sorted(
                ((family, value) for family, value in by_family.items() if value > 0),
                key=lambda item: (-item[1], item[0]),
            ),
            start=1,
        )
    ]
    mtp_continuation_pass = (
        sum(sizes.get(family, 0) for family in MTP_DENSE_FAMILIES)
        + sizes["mtp_routed_experts"] * mtp_continuation_union / experts
        + sizes["lm_head"]
        + batch * hidden_size * embedding_bytes
    )
    mtp_first_pass = (
        sum(sizes.get(family, 0) for family in MTP_DENSE_FAMILIES)
        + sizes["mtp_routed_experts"] * mtp_first_union / experts
        + sizes["lm_head"]
        + verifier_positions * hidden_size * embedding_bytes
        if mtp_proposals
        else 0
    )
    non_spec_union = _expert_union(experts, top_k, batch, route)
    non_spec_total = (
        sum(sizes.get(family, 0) for family in BASE_DENSE_FAMILIES)
        + sizes["base_routed_experts"] * non_spec_union / experts
        + sizes["lm_head"]
        + batch * hidden_size * (embedding_bytes + ple_rows * ple_bytes)
    )
    accepted_per_step = batch * accepted_tokens_per_step
    lever_ranking = _rank_levers(
        by_family,
        details,
        total,
        bandwidth,
        accepted_per_step,
        batch,
        non_spec_total,
        by_family["base_ple_dense"] + by_family["verifier_ple_rows"],
    )
    return {
        "assumptions": {
            "batch": batch,
            "concurrency": batch,
            "experts": experts,
            "top_k": top_k,
            "route_model": route,
            "verifier_positions_per_stream": verifier_positions_per_stream,
            "verifier_expected_unique_experts": verifier_union,
            "expected_unique_experts": verifier_union,
            "mtp_first_pass_expected_unique_experts": mtp_first_union,
            "mtp_continuation_expected_unique_experts": mtp_continuation_union,
            "nodes": nodes,
            "gb_s_per_node": gb_s_per_node,
            "ple_rows_per_verifier_position": ple_rows,
            "accepted_tokens_per_step": accepted_tokens_per_step,
            "mtp_proposals": mtp_proposals,
            "nvfp4_bytes_per_bf16_byte": NVFP4_BYTES_PER_BF16_BYTE,
            "nvfp4_bytes_per_fp8_byte": NVFP4_BYTES_PER_FP8_BYTE,
            "cudagraph_padding": "excluded",
            "scope": (
                "checkpoint weight and embedding-row read roofline; assumes one verifier "
                "weight stream across all verification positions, a first MTP pass over "
                "those positions, and one single-position MTP continuation per remaining "
                "proposal; excludes activation, recurrent-state, KV, compute, launch, "
                "and fabric stalls"
            ),
        },
        "checkpoint_bytes": sizes,
        "checkpoint_inventory": details,
        "residency_ranking": [
            {
                "rank": rank,
                "family": family,
                "bytes": int(detail["bytes"]),
                "dtype_bytes": detail["dtype_bytes"],
            }
            for rank, (family, detail) in enumerate(
                sorted(
                    details.items(),
                    key=lambda item: (-int(item[1]["bytes"]), item[0]),
                ),
                start=1,
            )
        ],
        "step_bytes": {
            "by_family": by_family,
            "shared_dense_and_lm_head": (
                sum(by_family[family] for family in BASE_DENSE_FAMILIES)
                + by_family["verifier_lm_head"]
            ),
            "routed_expert_union": by_family["base_routed_experts"],
            "embedding_and_ple_rows": (
                by_family["verifier_embedding_rows"] + by_family["verifier_ple_rows"]
            ),
            "mtp_first_pass_weights": mtp_first_pass,
            "mtp_continuation_pass_weights": mtp_continuation_pass,
            "mtp_proposal_traffic": sum(
                value for family, value in by_family.items() if family.startswith("mtp_")
            ),
            "total": total,
        },
        "traffic_ranking": traffic_ranking,
        "lever_ranking": lever_ranking,
        "ceiling": {
            "decode_steps_per_s": steps_s,
            "batch_steps_per_s": steps_s,
            "aggregate_accepted_tokens_per_s": aggregate_tokens_s,
            "accepted_tokens_per_s_per_stream": (
                steps_s * accepted_tokens_per_step
            ),
            "checkpoint_bytes_per_accepted_token": total / accepted_per_step,
        },
    }


def model_sweep(
    headers: Mapping[str, Mapping[str, object]],
    concurrencies: tuple[int, ...],
    **model_args: object,
) -> list[dict[str, object]]:
    """Compute a bounded ordered sweep, rejecting duplicates and empty input."""
    if not concurrencies or len(concurrencies) > 64:
        raise ValueError("concurrency sweep must contain between 1 and 64 values")
    if len(set(concurrencies)) != len(concurrencies):
        raise ValueError("concurrency sweep values must be unique")
    return [model(headers, batch=concurrency, **model_args) for concurrency in concurrencies]


def parse_concurrencies(value: str) -> tuple[int, ...]:
    """Parse a comma-separated concurrency list for the CLI boundary."""
    try:
        values = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "concurrencies must be comma-separated integers"
        ) from error
    if not values or len(values) > 64 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("concurrencies need 1..64 positive values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("concurrencies must be unique")
    return values


def _print_tsv(results: list[dict[str, object]], legacy: bool = False) -> None:
    if legacy:
        print(
            "batch\troute\tunique_experts\tstep_GiB\tbatch_steps_s\t"
            "aggregate_tok_s\tper_stream_tok_s"
        )
        result = results[0]
        assumptions = result["assumptions"]
        step_bytes = result["step_bytes"]
        ceiling = result["ceiling"]
        print(
            f'{assumptions["batch"]}\t{assumptions["route_model"]}\t'
            f'{assumptions["expected_unique_experts"]:.3f}\t'
            f'{step_bytes["total"]/2**30:.6f}\t{ceiling["batch_steps_per_s"]:.3f}\t'
            f'{ceiling["aggregate_accepted_tokens_per_s"]:.3f}\t'
            f'{ceiling["accepted_tokens_per_s_per_stream"]:.3f}'
        )
        return
    print(
        "concurrency\troute\tverify_positions\tverify_unique_experts\t"
        "mtp_first_unique_experts\tmtp_continuation_unique_experts\tstep_GiB\t"
        "GiB_per_accepted_token\tdecode_steps_s\taggregate_tok_s\tper_stream_tok_s"
    )
    for result in results:
        assumptions = result["assumptions"]
        step_bytes = result["step_bytes"]
        ceiling = result["ceiling"]
        print(
            f'{assumptions["concurrency"]}\t{assumptions["route_model"]}\t'
            f'{assumptions["verifier_positions_per_stream"]}\t'
            f'{assumptions["verifier_expected_unique_experts"]:.3f}\t'
            f'{assumptions["mtp_first_pass_expected_unique_experts"]:.3f}\t'
            f'{assumptions["mtp_continuation_expected_unique_experts"]:.3f}\t'
            f'{step_bytes["total"]/2**30:.6f}\t'
            f'{ceiling["checkpoint_bytes_per_accepted_token"]/2**30:.6f}\t'
            f'{ceiling["decode_steps_per_s"]:.3f}\t'
            f'{ceiling["aggregate_accepted_tokens_per_s"]:.3f}\t'
            f'{ceiling["accepted_tokens_per_s_per_stream"]:.3f}'
        )


def main() -> None:
    """Parse CLI configuration, read one checkpoint boundary, and print results."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--headers", type=Path)
    source.add_argument("--model-dir", type=Path)
    source.add_argument("--repo")
    parser.add_argument("--revision")
    point = parser.add_mutually_exclusive_group()
    point.add_argument(
        "--batch", type=int,
        help="single-concurrency compatibility mode for published commands",
    )
    point.add_argument("--concurrency", type=int)
    point.add_argument("--concurrencies", type=parse_concurrencies)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gb-s-per-node", type=float, default=238.0)
    parser.add_argument("--ple-rows", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--ple-bytes", type=int, default=1)
    parser.add_argument("--embedding-bytes", type=int, default=2)
    parser.add_argument("--route", choices=("uniform", "same"), default="uniform")
    parser.add_argument("--mtp-proposals", type=int, default=0)
    parser.add_argument("--accepted-tokens-per-step", type=float, default=1.0)
    parser.add_argument("--tsv", action="store_true")
    args = parser.parse_args()
    if args.repo:
        if not args.revision:
            parser.error("--repo requires --revision")
        headers = remote_headers(args.repo, args.revision)
    elif args.model_dir:
        headers = local_headers(args.model_dir)
    else:
        with args.headers.open(encoding="utf-8") as stream:
            headers = json.load(stream)

    if args.batch is not None:
        concurrencies = (args.batch,)
    elif args.concurrency is not None:
        concurrencies = (args.concurrency,)
    elif args.concurrencies is not None:
        concurrencies = args.concurrencies
    else:
        concurrencies = DEFAULT_CONCURRENCIES
    model_args = {
        "experts": args.experts,
        "top_k": args.top_k,
        "nodes": args.nodes,
        "gb_s_per_node": args.gb_s_per_node,
        "ple_rows": args.ple_rows,
        "hidden_size": args.hidden_size,
        "ple_bytes": args.ple_bytes,
        "embedding_bytes": args.embedding_bytes,
        "route": args.route,
        "mtp_proposals": args.mtp_proposals,
        "accepted_tokens_per_step": args.accepted_tokens_per_step,
    }
    try:
        results = model_sweep(headers, concurrencies, **model_args)
    except ValueError as error:
        parser.error(str(error))
    if args.tsv:
        _print_tsv(results, legacy=args.batch is not None)
    elif len(results) == 1:
        print(json.dumps(results[0], indent=2, sort_keys=True))
    else:
        print(json.dumps({"concurrencies": list(concurrencies), "results": results},
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
