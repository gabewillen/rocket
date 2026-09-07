"""Authenticated MTP weight-source contracts for the fixed Qwen3.8 engine.

The native source is one already-authenticated rank-local MTP slab.  This
module validates its tensor directory without opening the slab or duplicating
CUDA residency.  The production loader remains the payload-authentication and
ownership boundary.

An optional external source may replace only the 256 owner-local routed
experts.  Its descriptor is source-health evidence, not an assertion that it
is a separately trained draft head.  All 29 nonexpert tensors must hash equal
to hashes independently observed from the pinned NVIDIA source.  The caller
owns those observed hashes and all mappings are borrowed for the call.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .contract import (
    MODEL_NVFP4_ABI,
    MTP_FP8_ABI,
    PINNED_CONTRACT,
    SCHEMA,
    canonical_bytes,
)

EXTERNAL_MTP_SOURCE_SCHEMA = "rocket.qwen38-mtp-expert-source.v1"
LOCAL_EXPERTS = 256
EXPERT_EXTENTS_PER_RANK = LOCAL_EXPERTS * 3 * 2
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


class MtpSourceError(ValueError):
    """An MTP source failed its immutable tensor or provenance contract."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    length_bytes: int


_NONEXPERT = {
    "mtp.fc_embedding.weight": ((1280, 2560), 6_553_600),
    "mtp.fc_hidden.weight": ((1280, 2560), 6_553_600),
    "mtp.hyper_connection_mixer.hc_norm.weight": ((10240,), 20_480),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": ((320, 10240), 6_553_600),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": ((10240, 320), 6_553_600),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": ((4, 10240), 81_920),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": ((10240,), 20_480),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": ((320, 10240), 6_553_600),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": ((10240, 320), 6_553_600),
    "mtp.layers.0.mlp.gate.weight": ((512, 2560), 2_621_440),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": ((2560, 320), 1_638_400),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": ((320, 2560), 1_638_400),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": ((320, 2560), 1_638_400),
    "mtp.layers.0.mlp.shared_expert_gate.weight": ((1, 2560), 5_120),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": ((4, 10240), 81_920),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": ((10240,), 20_480),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": ((320, 10240), 6_553_600),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": ((10240, 320), 6_553_600),
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": ((320, 2560), 1_638_400),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": ((128,), 256),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": ((128,), 256),
    "mtp.layers.0.self_attn.k_norm.weight": ((256,), 512),
    "mtp.layers.0.self_attn.k_proj.weight": ((256, 2560), 1_310_720),
    "mtp.layers.0.self_attn.o_proj.weight": ((2560, 3072), 15_728_640),
    "mtp.layers.0.self_attn.q_norm.weight": ((256,), 512),
    "mtp.layers.0.self_attn.q_proj.weight": ((6144, 2560), 31_457_280),
    "mtp.layers.0.self_attn.v_proj.weight": ((256, 2560), 1_310_720),
    "mtp.pre_fc_norm_embedding.weight": ((2560,), 5_120),
    "mtp.pre_fc_norm_hidden.weight": ((10240,), 20_480),
}
MTP_NONEXPERT_TENSORS: Mapping[str, TensorSpec] = MappingProxyType(
    {name: TensorSpec(shape, size) for name, (shape, size) in _NONEXPERT.items()}
)


@dataclass(frozen=True)
class MtpExtent:
    name: str
    offset_bytes: int
    length_bytes: int
    shape: tuple[int, ...]
    dtype: str
    layout: str
    abi: str


@dataclass(frozen=True)
class NativeMtpSource:
    """Owned immutable directory for one loader-authenticated native slab."""

    artifact_key: str
    revision: str
    rank: int
    slab_key: str
    slab_bytes: int
    expert_abi: str
    local_experts: tuple[int, int]
    expert_extents: tuple[MtpExtent, ...]
    nonexpert_extents: tuple[MtpExtent, ...]
    target_references: tuple[str, str]
    nonexpert_contract_sha256: str


@dataclass(frozen=True)
class ExternalMtpSource:
    """Validated optional expert replacement with pinned nonexpert identity."""

    source_id: str
    base_revision: str
    rank: int
    expert_abi: str
    local_experts: tuple[int, int]
    expert_inventory_sha256: str
    nonexpert_sha256: Mapping[str, str]
    nonexpert_contract_sha256: str
    trained_head_distinct: bool = False
    nonexpert_differences: tuple[str, ...] = ()


def inspect_native_mtp_source(
    manifest: Mapping[str, object], rank: int
) -> NativeMtpSource:
    """Validate a borrowed manifest; perform no I/O and retain no input aliases."""

    if rank not in (0, 1) or not isinstance(manifest, Mapping):
        raise MtpSourceError("native MTP manifest and TP2 rank are required")
    record = dict(manifest)
    claimed = record.pop("artifact_key", None)
    if not _digest(claimed) or hashlib.sha256(canonical_bytes(record)).hexdigest() != claimed:
        raise MtpSourceError("rank-slab artifact identity changed")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("revision") != PINNED_CONTRACT.revision
        or manifest.get("overlay_artifact_key") != PINNED_CONTRACT.artifact_key
        or manifest.get("overlay_sha256") != PINNED_CONTRACT.overlay_sha256
        or manifest.get("tp_size") != 2
        or manifest.get("tensor_alignment_bytes") != 256
        or manifest.get("shared_payload_policy")
        != "target-owned-mtp-reference-read-once"
        or manifest.get("abis")
        != {"mtp_experts": MTP_FP8_ABI, "target_nvfp4": MODEL_NVFP4_ABI}
    ):
        raise MtpSourceError("native MTP provenance contract changed")
    slab_key = f"rank{rank}-mtp"
    slabs = manifest.get("slabs")
    slab = slabs.get(slab_key) if isinstance(slabs, Mapping) else None
    if not isinstance(slab, Mapping):
        raise MtpSourceError("rank-local MTP slab is absent")
    raw_entries = slab.get("entries")
    slab_bytes = slab.get("bytes")
    if (
        not isinstance(raw_entries, list)
        or len(raw_entries) != EXPERT_EXTENTS_PER_RANK + len(MTP_NONEXPERT_TENSORS)
        or isinstance(slab_bytes, bool)
        or not isinstance(slab_bytes, int)
        or slab_bytes <= 0
    ):
        raise MtpSourceError("rank-local MTP inventory size changed")
    entries = tuple(_extent(item, slab_bytes) for item in raw_entries)
    if len({item.name for item in entries}) != len(entries):
        raise MtpSourceError("rank-local MTP extent names are not unique")
    by_name = {item.name: item for item in entries}
    nonexpert = []
    for name, spec in MTP_NONEXPERT_TENSORS.items():
        extent = by_name.get(name)
        if (
            extent is None
            or extent.shape != spec.shape
            or extent.length_bytes != spec.length_bytes
            or extent.dtype != "BF16"
            or extent.layout != "checkpoint"
            or extent.abi != "native"
        ):
            raise MtpSourceError(f"nonexpert extent contract changed: {name}")
        nonexpert.append(extent)
    first = rank * LOCAL_EXPERTS
    expert_names = set(by_name).difference(MTP_NONEXPERT_TENSORS)
    expected_names = set()
    for expert in range(first, first + LOCAL_EXPERTS):
        for projection in ("down_proj", "gate_proj", "up_proj"):
            rows, columns = (
                (2560, 640) if projection == "down_proj" else (640, 2560)
            )
            for leaf, dtype, shape, size in (
                ("weight", "F8_E4M3", (rows, columns), rows * columns),
                (
                    "weight_scale_inv",
                    "BF16",
                    (rows // 128, columns // 128),
                    (rows // 128) * (columns // 128) * 2,
                ),
            ):
                name = f"mtp.layers.0.mlp.experts.{expert}.{projection}.{leaf}"
                expected_names.add(name)
                extent = by_name.get(name)
                if (
                    extent is None
                    or extent.dtype != dtype
                    or extent.shape != shape
                    or extent.length_bytes != size
                    or extent.layout != "checkpoint"
                    or extent.abi != MTP_FP8_ABI
                ):
                    raise MtpSourceError(f"native expert extent contract changed: {name}")
    if expert_names != expected_names:
        raise MtpSourceError("native expert inventory changed")
    expected_refs = (
        {"name": "lm_head.weight", "physical_owner": f"rank{rank}-target"},
        {
            "name": "model.language_model.embed_tokens.weight",
            "physical_owner": f"rank{rank}-target",
        },
    )
    references = slab.get("references")
    if not isinstance(references, list) or tuple(references) != expected_refs:
        raise MtpSourceError("target-owned MTP references changed")
    contract_digest = hashlib.sha256(
        canonical_bytes(
            [
                {
                    "name": item.name,
                    "shape": item.shape,
                    "length_bytes": item.length_bytes,
                    "dtype": item.dtype,
                    "layout": item.layout,
                    "abi": item.abi,
                }
                for item in nonexpert
            ]
        )
    ).hexdigest()
    return NativeMtpSource(
        claimed,
        PINNED_CONTRACT.revision,
        rank,
        slab_key,
        slab_bytes,
        MTP_FP8_ABI,
        (first, first + LOCAL_EXPERTS - 1),
        tuple(by_name[name] for name in sorted(expert_names)),
        tuple(nonexpert),
        tuple(item["name"] for item in expected_refs),
        contract_digest,
    )


def inspect_external_mtp_source(
    record: Mapping[str, object],
    native: NativeMtpSource,
    trusted_native_nonexpert_sha256: Mapping[str, str],
) -> ExternalMtpSource:
    """Validate an external expert descriptor against independently hashed native tensors."""

    if not isinstance(record, Mapping) or not isinstance(native, NativeMtpSource):
        raise MtpSourceError("external descriptor and native source are required")
    if record.get("trained_head_distinct") is not None:
        raise MtpSourceError("external expert ABI cannot make a distinct trained-head claim")
    expected_keys = {
        "schema",
        "source_id",
        "base_revision",
        "rank",
        "expert_abi",
        "local_experts",
        "expert_inventory_sha256",
        "nonexpert_sha256",
        "nonexpert_contract_sha256",
    }
    if set(record) != expected_keys:
        raise MtpSourceError("external MTP source descriptor fields changed")
    source_id = record.get("source_id")
    inventory = record.get("expert_inventory_sha256")
    local = record.get("local_experts")
    expert_abi = record.get("expert_abi")
    if (
        record.get("schema") != EXTERNAL_MTP_SOURCE_SCHEMA
        or not _digest(source_id)
        or record.get("base_revision") != native.revision
        or record.get("rank") != native.rank
        or local != list(native.local_experts)
        or expert_abi != MODEL_NVFP4_ABI
        or not _digest(inventory)
        or record.get("nonexpert_contract_sha256")
        != native.nonexpert_contract_sha256
    ):
        raise MtpSourceError("external expert source identity changed")
    claimed_hashes = record.get("nonexpert_sha256")
    if (
        not isinstance(claimed_hashes, Mapping)
        or set(claimed_hashes) != set(MTP_NONEXPERT_TENSORS)
        or set(trusted_native_nonexpert_sha256) != set(MTP_NONEXPERT_TENSORS)
        or any(not _digest(value) for value in claimed_hashes.values())
        or any(not _digest(value) for value in trusted_native_nonexpert_sha256.values())
    ):
        raise MtpSourceError("external source must identify exactly 29 nonexpert tensors")
    differences = tuple(
        name
        for name in MTP_NONEXPERT_TENSORS
        if claimed_hashes[name] != trusted_native_nonexpert_sha256[name]
    )
    if differences:
        raise MtpSourceError("external source must keep all 29 nonexpert tensors byte-identical")
    owned_hashes = MappingProxyType(dict(claimed_hashes))
    return ExternalMtpSource(
        source_id,
        native.revision,
        native.rank,
        expert_abi,
        native.local_experts,
        inventory,
        owned_hashes,
        native.nonexpert_contract_sha256,
    )


def _extent(raw: object, slab_bytes: int) -> MtpExtent:
    if not isinstance(raw, Mapping):
        raise MtpSourceError("MTP extent record changed")
    name = raw.get("name")
    offset = raw.get("offset_bytes")
    length = raw.get("length_bytes")
    shape = raw.get("shape")
    dtype = raw.get("dtype")
    layout = raw.get("layout")
    abi = raw.get("abi")
    if (
        not isinstance(name, str)
        or not name
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset % 256
        or isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or offset + length > slab_bytes
        or not isinstance(shape, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        or not isinstance(dtype, str)
        or not isinstance(layout, str)
        or not isinstance(abi, str)
    ):
        raise MtpSourceError("MTP extent record changed")
    return MtpExtent(name, offset, length, tuple(shape), dtype, layout, abi)


def _digest(value: object) -> bool:
    return isinstance(value, str) and _HEX_256.fullmatch(value) is not None


__all__ = [
    "EXTERNAL_MTP_SOURCE_SCHEMA",
    "ExternalMtpSource",
    "MTP_NONEXPERT_TENSORS",
    "MtpExtent",
    "MtpSourceError",
    "NativeMtpSource",
    "TensorSpec",
    "inspect_external_mtp_source",
    "inspect_native_mtp_source",
]
