#!/usr/bin/env python3
"""Plan a Rocket-owned NVFP4 overlay for NVIDIA's GLM-5.3-Flash snapshot.

Requires Python 3.9 or newer. The command reads safetensors headers and the
bytes of selected BF16 tensors.
It writes a deterministic JSON manifest to stdout. It never changes the base
checkpoint and does not quantize weights.

Contract:
  * input is a NIM safetensors snapshot with config.json, checksums.blake3,
    model.safetensors.index.json, and all referenced shards;
  * routed experts, vision tensors, the MTP layer, non-matrix tensors, and
    non-BF16 tensors are inventory-only and remain in the immutable base;
  * candidate families are text-decode BF16 matrices whose aggregate size is
    at least --min-family-mib (64 MiB by default);
  * source tensor hashes cover the exact safetensors data ranges;
  * stdout is the only normal side effect. Validation failures write a bounded
    diagnostic to stderr and return 2. I/O failures return 1.

Usage:
  scripts/numerics/nvfp4-overlay-inventory.py SNAPSHOT > overlay-plan.json
  scripts/numerics/nvfp4-overlay-inventory.py SNAPSHOT --summary
  scripts/numerics/nvfp4-overlay-inventory.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import struct
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "rocket.nvfp4-overlay-plan.v1"
POLICY_VERSION = "glm53-nim-nonexpert-bf16-matrices.v1"
MODEL_ID = "nim/zai-org/glm-5.3-flash"
DEFAULT_SNAPSHOT = Path(
    "~/.cache/nim-glm53-2.1.2/ngc/hub/"
    "models--nim--zai-org--glm-5.3-flash/snapshots/nim-aa28e1f-nvfp4"
).expanduser()
HEADER_LIMIT_BYTES = 256 << 20
HASH_CHUNK_BYTES = 8 << 20
GIB = 1 << 30
MIB = 1 << 20
NVFP4_BLOCK_SIZE = 16
SOURCE_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")

# These checks turn the measured GLM-5.3-Flash shape profile into a drift gate.
# Values are exact bytes. All output values are still derived from the headers.
KNOWN_ROLLUP_BYTES = {
    "kda_qkv": 6_845_104_128,
    "embed+lm_head": 2_537_553_920,
    "kda_o": 2_281_701_376,
    "shared_expert": 2_113_929_216,
    "mla_o": 1_476_395_008,
}


class InventoryError(ValueError):
    """The snapshot violates the overlay inventory contract."""


@dataclass(frozen=True)
class TensorHeader:
    """Validated immutable metadata for one safetensors data range."""

    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    file_data_start: int

    @property
    def source_bytes(self) -> int:
        return self.data_end - self.data_start

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def absolute_start(self) -> int:
        return self.file_data_start + self.data_start

    @property
    def absolute_end(self) -> int:
        return self.file_data_start + self.data_end


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a borrowed file without mutating it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_range(path: Path, start: int, size: int) -> str:
    """Hash exactly one validated borrowed file range."""
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(HASH_CHUNK_BYTES, remaining))
            if not chunk:
                raise InventoryError(f"short tensor data range in {path.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash the canonical UTF-8 JSON representation of an owned value."""
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    """Read one bounded JSON object from the filesystem boundary."""
    size = path.stat().st_size
    if size > HEADER_LIMIT_BYTES:
        raise InventoryError(f"JSON file exceeds {HEADER_LIMIT_BYTES} bytes: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise InventoryError(f"expected JSON object: {path.name}")
    return value


def read_safetensors_header(path: Path) -> tuple[dict[str, object], int]:
    """Read and validate a bounded safetensors JSON header."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise InventoryError(f"missing safetensors length prefix: {path.name}")
        (header_size,) = struct.unpack("<Q", prefix)
        if header_size == 0 or header_size > HEADER_LIMIT_BYTES:
            raise InventoryError(f"invalid safetensors header size in {path.name}")
        encoded = handle.read(header_size)
        if len(encoded) != header_size:
            raise InventoryError(f"short safetensors header in {path.name}")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise InventoryError(f"expected safetensors header object: {path.name}")
    return value, 8 + header_size


def validate_tensor_header(
    name: str, shard: str, raw: object, file_data_start: int, shard_size: int
) -> TensorHeader:
    """Convert untrusted header JSON to a typed tensor record."""
    if not isinstance(raw, dict):
        raise InventoryError(f"invalid metadata for tensor {name}")
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in SOURCE_DTYPE_BYTES:
        raise InventoryError(f"unsupported dtype for tensor {name}: {dtype!r}")
    if (
        not isinstance(shape, list)
        or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
    ):
        raise InventoryError(f"invalid shape for tensor {name}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
    ):
        raise InventoryError(f"invalid data offsets for tensor {name}")
    start, end = offsets
    if start < 0 or end < start or file_data_start + end > shard_size:
        raise InventoryError(f"out-of-range tensor data for {name}")
    expected = math.prod(shape) * SOURCE_DTYPE_BYTES[dtype]
    if end - start != expected:
        raise InventoryError(
            f"size mismatch for {name}: header={end - start}, shape={expected}"
        )
    return TensorHeader(
        name=name,
        shard=shard,
        dtype=dtype,
        shape=tuple(shape),
        data_start=start,
        data_end=end,
        file_data_start=file_data_start,
    )


def parse_checksums(path: Path) -> dict[str, str]:
    """Read the NIM BLAKE3 checksum inventory without claiming verification."""
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split("  ", 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                raise InventoryError(f"invalid checksums.blake3 line {line_number}")
            if parts[1] in entries:
                raise InventoryError(f"duplicate checksum path: {parts[1]}")
            entries[parts[1]] = parts[0]
    return entries


def family_for(name: str, shape: Sequence[int], layer_types: Sequence[str]) -> str | None:
    """Classify one main-text matrix into a bounded GLM-5.3 family."""
    if name == "lm_head.weight":
        return "lm_head"
    if name == "model.language_model.embed_tokens.weight":
        return "embed"
    match = LAYER_RE.match(name)
    if match is None or len(shape) != 2:
        return None
    layer = int(match.group(1))
    if layer >= len(layer_types):
        return None
    rest = match.group(2)
    if rest.startswith("mlp.experts."):
        return None
    if rest in {"hc_attn_fn", "hc_ffn_fn"}:
        return "hyper_connection"
    if rest.startswith("mlp.shared_experts."):
        return "shared_expert"
    if rest == "mlp.gate.weight":
        return "router"
    if rest.startswith("mlp."):
        return "dense_mlp"
    if not rest.startswith("self_attn."):
        return None
    projection = rest.removeprefix("self_attn.")
    if layer_types[layer] == "linear_attention":
        direct = {
            "q_proj.weight": "kda_q",
            "k_proj.weight": "kda_k",
            "v_proj.weight": "kda_v",
            "o_proj.weight": "kda_o",
        }
        if projection in direct:
            return direct[projection]
        if projection in {
            "b_proj.weight",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "g_a_proj.weight",
            "g_b_proj.weight",
        }:
            return "kda_gates"
        return None
    if projection in {"q_a_proj.weight", "q_b_proj.weight"}:
        return "mla_q"
    if projection in {"kv_a_proj_with_mqa.weight", "kv_b_proj.weight"}:
        return "mla_kv"
    if projection == "o_proj.weight":
        return "mla_o"
    if projection.startswith("indexer."):
        return "mla_indexer"
    return None


def nvfp4_estimate(shape: Sequence[int]) -> dict[str, int]:
    """Return estimated overlay bytes for one row-major matrix."""
    if len(shape) != 2 or shape[-1] % NVFP4_BLOCK_SIZE:
        raise InventoryError(f"NVFP4 requires a rank-2, K%16=0 shape: {list(shape)}")
    elements = math.prod(shape)
    packed_weight = (elements + 1) // 2
    block_scales = shape[0] * (shape[1] // NVFP4_BLOCK_SIZE)
    global_scale = 4
    return {
        "packed_weight_bytes": packed_weight,
        "block_scale_bytes": block_scales,
        "global_scale_bytes": global_scale,
        "total_bytes": packed_weight + block_scales + global_scale,
    }


def quality_gate_template() -> dict[str, object]:
    """Return unmeasured fields required before an experiment may be ranked."""
    return {
        "status": "not_run",
        "activation_ranges": {
            "baseline_min": None,
            "baseline_max": None,
            "baseline_abs_p99_9": None,
            "candidate_min": None,
            "candidate_max": None,
            "candidate_abs_p99_9": None,
        },
        "calibration_error": {
            "relative_l2": None,
            "max_abs": None,
            "max_rel": None,
        },
        "projection_output_error": {
            "relative_l2": None,
            "max_abs": None,
            "max_rel": None,
        },
        "logit_drift": {
            "mean_abs": None,
            "max_abs": None,
            "kl_divergence": None,
        },
        "greedy_token_divergence": {
            "diverged_tokens": None,
            "evaluated_tokens": None,
            "first_divergence_index": None,
        },
        "evaluation": {
            "harness": None,
            "dataset": None,
            "metric": "perplexity",
            "baseline": None,
            "candidate": None,
            "delta": None,
            "status": "harness_not_recorded",
        },
        "quality_loss_score": None,
        "quality_loss_per_gib_per_token_removed": None,
    }


def build_experiment_matrix(families: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Build every isolated-family run followed by every unordered pair."""
    by_name = {str(row["family"]): row for row in families}
    names = sorted(by_name)
    experiments: list[dict[str, object]] = []
    for width, combinations in (
        (1, ((name,) for name in names)),
        (2, itertools.combinations(names, 2)),
    ):
        for selected in combinations:
            removable = sum(int(by_name[name]["estimated_removable_bytes_per_token"]) for name in selected)
            experiment_id = ("isolated:" if width == 1 else "pair:") + "+".join(selected)
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "kind": "isolated_family" if width == 1 else "interaction_pair",
                    "families": list(selected),
                    "estimated_removable_bytes_per_token": removable,
                    "estimated_removable_gib_per_token": removable / GIB,
                    "ranking_eligible": False,
                    "quality_gates": quality_gate_template(),
                }
            )
    return experiments


def load_inventory(
    snapshot: Path,
    min_family_bytes: int,
    *,
    verify_known_profile: bool = True,
) -> dict[str, object]:
    """Read a NIM snapshot and return an owned deterministic overlay plan."""
    required = ["config.json", "checksums.blake3", "model.safetensors.index.json"]
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise InventoryError("missing snapshot files: " + ", ".join(missing))

    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    checksums_path = snapshot / "checksums.blake3"
    config = load_json_object(config_path)
    index = load_json_object(index_path)
    text_config = config.get("text_config")
    weight_map = index.get("weight_map")
    if not isinstance(text_config, dict) or not isinstance(weight_map, dict):
        raise InventoryError("config.text_config and index.weight_map must be objects")
    layer_types = text_config.get("layer_types")
    num_hidden_layers = text_config.get("num_hidden_layers")
    if (
        not isinstance(layer_types, list)
        or not all(isinstance(value, str) for value in layer_types)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
        or len(layer_types) != num_hidden_layers
    ):
        raise InventoryError("invalid text layer_types/num_hidden_layers contract")
    if not all(isinstance(name, str) and isinstance(shard, str) for name, shard in weight_map.items()):
        raise InventoryError("weight_map must map tensor names to shard names")

    declared_checksums = parse_checksums(checksums_path)
    shard_headers: dict[str, tuple[dict[str, object], int]] = {}
    tensors: list[TensorHeader] = []
    for name, shard in sorted(weight_map.items()):
        shard_path = snapshot / shard
        if not shard_path.is_file():
            raise InventoryError(f"missing shard: {shard}")
        if shard not in declared_checksums:
            raise InventoryError(f"shard absent from checksums.blake3: {shard}")
        if shard not in shard_headers:
            shard_headers[shard] = read_safetensors_header(shard_path)
        header, file_data_start = shard_headers[shard]
        if name not in header:
            raise InventoryError(f"index tensor absent from shard header: {name}")
        tensors.append(
            validate_tensor_header(
                name, shard, header[name], file_data_start, shard_path.stat().st_size
            )
        )

    family_members: dict[str, list[TensorHeader]] = defaultdict(list)
    preserved: dict[tuple[str, str], dict[str, object]] = {}
    expert_bytes = 0
    expert_tensors = 0
    for tensor in tensors:
        match = LAYER_RE.match(tensor.name)
        layer = int(match.group(1)) if match else None
        is_main_text = (
            tensor.name in {"lm_head.weight", "model.language_model.embed_tokens.weight"}
            or (layer is not None and layer < num_hidden_layers)
        )
        is_routed_expert = ".mlp.experts." in tensor.name
        if is_routed_expert:
            expert_bytes += tensor.source_bytes
            expert_tensors += 1
            continue
        family = family_for(tensor.name, tensor.shape, layer_types)
        if is_main_text and tensor.dtype == "BF16" and family is not None:
            family_members[family].append(tensor)
            continue
        if not is_main_text or tensor.name.startswith("model.visual."):
            reason = "outside_main_text_decode"
        elif tensor.dtype != "BF16":
            reason = "source_dtype_preserved"
        elif len(tensor.shape) != 2:
            reason = "non_matrix_preserved"
        else:
            reason = "unselected_matrix_preserved"
        key = (reason, tensor.dtype)
        row = preserved.setdefault(
            key,
            {"reason": reason, "dtype": tensor.dtype, "tensor_count": 0, "source_bytes": 0},
        )
        row["tensor_count"] = int(row["tensor_count"]) + 1
        row["source_bytes"] = int(row["source_bytes"]) + tensor.source_bytes

    selected_names = {
        name
        for name, members in family_members.items()
        if sum(tensor.source_bytes for tensor in members) >= min_family_bytes
    }
    candidate_tensors = [
        (family, tensor)
        for family in sorted(selected_names)
        for tensor in sorted(family_members[family], key=lambda item: item.name)
    ]
    if not candidate_tensors:
        raise InventoryError("no candidate families met the minimum size")

    ranges: dict[tuple[int, int, int, int], list[str]] = defaultdict(list)
    source_hashes: dict[str, str] = {}
    for _, tensor in candidate_tensors:
        shard_stat = (snapshot / tensor.shard).stat()
        storage_key = (
            shard_stat.st_dev,
            shard_stat.st_ino,
            tensor.absolute_start,
            tensor.absolute_end,
        )
        ranges[storage_key].append(tensor.name)
        source_hashes[tensor.name] = sha256_range(
            snapshot / tensor.shard, tensor.absolute_start, tensor.source_bytes
        )
    by_content: dict[tuple[int, str], list[str]] = defaultdict(list)
    for _, tensor in candidate_tensors:
        by_content[(tensor.source_bytes, source_hashes[tensor.name])].append(tensor.name)

    source_identity = {
        "model_id": MODEL_ID,
        "snapshot_id": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "checksums_blake3_sha256": sha256_file(checksums_path),
        "declared_shard_blake3": {
            shard: declared_checksums[shard] for shard in sorted(shard_headers)
        },
        "checksum_verification": "manifest_parsed_shard_content_not_rehashed",
    }
    snapshot_key = canonical_sha256(source_identity)
    packing_contract = {
        "serving_dtype": "NVFP4",
        "value_format": "e2m1",
        "values_per_packed_u8": 2,
        "block_size_elements": NVFP4_BLOCK_SIZE,
        "block_axis": "last_input_dimension",
        "block_scale_format": "fp8_e4m3",
        "block_scale_logical_layout": "row_major_out_by_k_block",
        "block_scale_runtime_layout": "cutlass_sm1xx_sfa_swizzled",
        "global_scale_format": "fp32_scalar_per_source_tensor",
        "estimate_includes_global_scale": True,
        "actual_quantization": False,
    }
    policy = {
        "version": POLICY_VERSION,
        "base_mutation": "forbidden",
        "routed_experts": "preserve_exactly_as_shipped_in_base_snapshot",
        "eligible_source_dtype": "BF16",
        "eligible_rank": 2,
        "main_text_layers": [0, num_hidden_layers - 1],
        "excluded_mtp_layers_start": num_hidden_layers,
        "minimum_family_bytes": min_family_bytes,
        "family_ids": sorted(selected_names),
        "packing_contract": packing_contract,
    }
    policy_key = canonical_sha256(policy)
    overlay_key = canonical_sha256(
        {"schema": SCHEMA, "snapshot_key": snapshot_key, "policy_key": policy_key}
    )

    tensor_rows: list[dict[str, object]] = []
    for family, tensor in candidate_tensors:
        estimate = nvfp4_estimate(tensor.shape)
        shard_stat = (snapshot / tensor.shard).stat()
        storage_key = (
            shard_stat.st_dev,
            shard_stat.st_ino,
            tensor.absolute_start,
            tensor.absolute_end,
        )
        aliases = sorted(name for name in ranges[storage_key] if name != tensor.name)
        same_content = sorted(
            name
            for name in by_content[(tensor.source_bytes, source_hashes[tensor.name])]
            if name != tensor.name
        )
        cache_key = canonical_sha256(
            {
                "snapshot_key": snapshot_key,
                "policy_key": policy_key,
                "source_tensor": tensor.name,
                "source_sha256": source_hashes[tensor.name],
                "shape": tensor.shape,
                "source_dtype": tensor.dtype,
            }
        )
        tensor_rows.append(
            {
                "family": family,
                "source_tensor": tensor.name,
                "source_shard": tensor.shard,
                "source_dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "source_data_offsets": [tensor.data_start, tensor.data_end],
                "source_bytes": tensor.source_bytes,
                "source_sha256": source_hashes[tensor.name],
                "storage_identity": {
                    "shard_blake3": declared_checksums[tensor.shard],
                    "absolute_byte_range": [tensor.absolute_start, tensor.absolute_end],
                },
                "tied_storage_with": aliases,
                "same_content_separate_storage": same_content,
                "packing_estimate": estimate,
                "overlay": {
                    "status": "planned_not_materialized",
                    "cache_key": cache_key,
                    "relative_directory": f"objects/{cache_key[:2]}/{cache_key}",
                    "packed_weight_file": "weight.u8",
                    "block_scale_file": "weight_scale.f8_e4m3",
                    "global_scale_file": "weight_scale_2.f32",
                    "metadata_file": "metadata.json",
                    "output_sha256": None,
                    "actual_bytes": None,
                },
            }
        )

    family_rows: list[dict[str, object]] = []
    for family in sorted(selected_names):
        members = [row for row in tensor_rows if row["family"] == family]
        unique_ranges = {
            (
                str(row["source_shard"]),
                tuple(row["source_data_offsets"]),
            )
            for row in members
        }
        source_bytes = sum(int(row["source_bytes"]) for row in members)
        packed_bytes = 0
        for row in members:
            estimate = row["packing_estimate"]
            assert isinstance(estimate, dict)
            packed_bytes += int(estimate["total_bytes"])
        family_rows.append(
            {
                "family": family,
                "tensor_count": len(members),
                "unique_storage_ranges": len(unique_ranges),
                "source_bf16_bytes_per_token": source_bytes,
                "source_bf16_gib_per_token": source_bytes / GIB,
                "estimated_nvfp4_bytes_per_token": packed_bytes,
                "estimated_nvfp4_gib_per_token": packed_bytes / GIB,
                "estimated_removable_bytes_per_token": source_bytes - packed_bytes,
                "estimated_removable_gib_per_token": (source_bytes - packed_bytes) / GIB,
                "measurement_status": "header_derived_estimate",
                "quality_loss_per_gib_per_token_removed": None,
                "ranking_eligible": False,
            }
        )
    family_rows.sort(
        key=lambda row: (-int(row["estimated_removable_bytes_per_token"]), str(row["family"]))
    )
    for planning_rank, row in enumerate(family_rows, 1):
        row["planning_rank_before_quality_measurement"] = planning_rank

    by_family = {str(row["family"]): row for row in family_rows}
    rollup_members = {
        "kda_qkv": ["kda_q", "kda_k", "kda_v"],
        "embed+lm_head": ["embed", "lm_head"],
        "kda_o": ["kda_o"],
        "shared_expert": ["shared_expert"],
        "mla_o": ["mla_o"],
    }
    rollups: list[dict[str, object]] = []
    for rollup, members in rollup_members.items():
        if not all(member in by_family for member in members):
            if verify_known_profile:
                raise InventoryError(f"known rollup has missing family: {rollup}")
            continue
        source_bytes = sum(int(by_family[member]["source_bf16_bytes_per_token"]) for member in members)
        packed_bytes = sum(int(by_family[member]["estimated_nvfp4_bytes_per_token"]) for member in members)
        expected = KNOWN_ROLLUP_BYTES[rollup]
        rollups.append(
            {
                "rollup": rollup,
                "families": members,
                "source_bf16_bytes_per_token": source_bytes,
                "source_bf16_gib_per_token": source_bytes / GIB,
                "estimated_nvfp4_bytes_per_token": packed_bytes,
                "estimated_nvfp4_gib_per_token": packed_bytes / GIB,
                "estimated_removable_bytes_per_token": source_bytes - packed_bytes,
                "estimated_removable_gib_per_token": (source_bytes - packed_bytes) / GIB,
                "known_profile_expected_source_bytes": expected,
                "known_profile_matches": source_bytes == expected,
            }
        )
    mismatches = [str(row["rollup"]) for row in rollups if not row["known_profile_matches"]]
    if verify_known_profile and mismatches:
        raise InventoryError("known GLM-5.3 profile drift: " + ", ".join(mismatches))

    experiments = build_experiment_matrix(family_rows)
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "source": source_identity | {"snapshot_key": snapshot_key},
        "overlay": {
            "overlay_key": overlay_key,
            "policy_key": policy_key,
            "relative_root": f"overlays/{snapshot_key}/{policy_key}",
            "base_checkpoint_is_immutable": True,
            "materialization_status": "not_started",
        },
        "policy": policy,
        "ranking_contract": {
            "primary_metric": "quality_loss_per_gib_per_token_removed",
            "direction": "ascending",
            "eligibility": "all quality gates measured",
            "unmeasured_planning_order": "estimated_removable_bytes_per_token descending, family id ascending",
            "quality_loss_score_definition": None,
            "quality_loss_score_status": "must_be_defined_by_evaluation_harness_before_ranking",
        },
        "family_ranking": family_rows,
        "requested_rollups": rollups,
        "candidate_tensors": tensor_rows,
        "experiment_matrix": experiments,
        "preserved_base_inventory": {
            "routed_experts": {
                "policy": "overlay_has_no_entries_for_routed_experts",
                "tensor_count": expert_tensors,
                "source_bytes": expert_bytes,
            },
            "other": sorted(
                preserved.values(), key=lambda row: (str(row["reason"]), str(row["dtype"]))
            ),
        },
        "measurements": {
            "source": "safetensors_headers_and_selected_tensor_byte_ranges",
            "actual_quantization_run": False,
            "quality_evaluation_run": False,
            "candidate_family_count": len(family_rows),
            "candidate_tensor_count": len(tensor_rows),
            "isolated_experiment_count": len(family_rows),
            "pair_experiment_count": len(experiments) - len(family_rows),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def print_summary(manifest: Mapping[str, object]) -> None:
    """Print a stable compact table derived from a completed manifest."""
    source = manifest["source"]
    measurements = manifest["measurements"]
    assert isinstance(source, dict) and isinstance(measurements, dict)
    print(f"snapshot_key\t{source['snapshot_key']}")
    print(f"manifest_sha256\t{manifest['manifest_sha256']}")
    print("family\tbf16_GiB/token\tnvfp4_est_GiB/token\tremovable_GiB/token\ttensors")
    rows = manifest["family_ranking"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        print(
            f"{row['family']}\t{row['source_bf16_gib_per_token']:.6f}"
            f"\t{row['estimated_nvfp4_gib_per_token']:.6f}"
            f"\t{row['estimated_removable_gib_per_token']:.6f}"
            f"\t{row['tensor_count']}"
        )
    print("rollup\tbf16_GiB/token\tnvfp4_est_GiB/token\tremovable_GiB/token")
    rollups = manifest["requested_rollups"]
    assert isinstance(rollups, list)
    for row in rollups:
        assert isinstance(row, dict)
        print(
            f"{row['rollup']}\t{row['source_bf16_gib_per_token']:.6f}"
            f"\t{row['estimated_nvfp4_gib_per_token']:.6f}"
            f"\t{row['estimated_removable_gib_per_token']:.6f}"
        )
    print(
        "experiments\t"
        f"{measurements['isolated_experiment_count']} isolated + "
        f"{measurements['pair_experiment_count']} pairs"
    )
    print("quantization_run\tfalse")
    print("quality_evaluation_run\tfalse")


def write_fixture(snapshot: Path) -> None:
    """Create a tiny owned snapshot used only by --self-test."""
    snapshot.mkdir()
    tensors = {
        "lm_head.weight": ("BF16", [1, 16], bytes(range(32))),
        "model.language_model.embed_tokens.weight": ("BF16", [1, 16], bytes(range(32))),
        "model.language_model.layers.0.self_attn.q_proj.weight": (
            "BF16",
            [1, 16],
            bytes(reversed(range(32))),
        ),
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight": (
            "U8",
            [1, 8],
            bytes(range(8)),
        ),
    }
    offsets: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        offsets[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(offsets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    shard = "model-00001-of-00001.safetensors"
    with (snapshot / shard).open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)
    index = {"weight_map": {name: shard for name in tensors}}
    config = {"text_config": {"num_hidden_layers": 1, "layer_types": ["linear_attention"]}}
    for name, value in (("model.safetensors.index.json", index), ("config.json", config)):
        with (snapshot / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    shard_digest = "0" * 64
    with (snapshot / "checksums.blake3").open("w", encoding="utf-8") as handle:
        handle.write(f"{shard_digest}  {shard}\n")


def self_test() -> int:
    """Exercise valid input, storage identity, determinism, and rejection."""
    with tempfile.TemporaryDirectory(prefix="rocket-nvfp4-inventory-") as raw:
        snapshot = Path(raw) / "fixture"
        write_fixture(snapshot)
        header, data_start = read_safetensors_header(snapshot / "model-00001-of-00001.safetensors")
        q = validate_tensor_header(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model-00001-of-00001.safetensors",
            header["model.language_model.layers.0.self_attn.q_proj.weight"],
            data_start,
            (snapshot / "model-00001-of-00001.safetensors").stat().st_size,
        )
        assert q.source_bytes == 32
        assert nvfp4_estimate(q.shape) == {
            "packed_weight_bytes": 8,
            "block_scale_bytes": 1,
            "global_scale_bytes": 4,
            "total_bytes": 13,
        }
        assert family_for(q.name, q.shape, ["linear_attention"]) == "kda_q"
        first = sha256_range(snapshot / q.shard, q.absolute_start, q.source_bytes)
        second = sha256_range(snapshot / q.shard, q.absolute_start, q.source_bytes)
        assert first == second
        manifest_a = load_inventory(snapshot, 1, verify_known_profile=False)
        manifest_b = load_inventory(snapshot, 1, verify_known_profile=False)
        assert manifest_a == manifest_b
        assert manifest_a["manifest_sha256"] == canonical_sha256(
            {key: value for key, value in manifest_a.items() if key != "manifest_sha256"}
        )
        measurements = manifest_a["measurements"]
        assert isinstance(measurements, dict)
        assert measurements["actual_quantization_run"] is False
        bad = {"dtype": "BF16", "shape": [1, 16], "data_offsets": [0, 31]}
        try:
            validate_tensor_header("bad", q.shard, bad, data_start, 10_000)
        except InventoryError:
            pass
        else:
            raise AssertionError("invalid tensor size was accepted")
    print("self-test: PASS")
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the explicit CLI configuration boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", nargs="?", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--min-family-mib", type=int, default=64)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.min_family_mib <= 0:
        parser.error("--min-family-mib must be positive")
    return args


def main(argv: Sequence[str]) -> int:
    """Run the read-only inventory boundary and emit its requested output."""
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        manifest = load_inventory(args.snapshot.expanduser().resolve(), args.min_family_mib * MIB)
    except InventoryError as error:
        print(f"inventory validation failed: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"inventory I/O failed: {error}", file=sys.stderr)
        return 1
    if args.summary:
        print_summary(manifest)
    else:
        json.dump(manifest, sys.stdout, ensure_ascii=True, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
