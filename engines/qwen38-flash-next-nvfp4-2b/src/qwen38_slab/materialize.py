"""Offline Qwen3.8 slab materializer. Safetensors are absent from runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .contract import (
    MODEL_NVFP4_ABI,
    MTP_FP8_ABI,
    PINNED_CONTRACT,
    SCHEMA,
    SLAB_KEYS,
    SlabContract,
    SlabError,
    align_up,
    canonical_bytes,
    load_plan,
    read_safetensors_index,
    sf_swizzle,
    sha256_file,
    validate_overlay,
    validate_plan,
)

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4,
               "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def _read_exact_at(path: Path, offset: int, size: int) -> bytes:
    try:
        with path.open("rb", buffering=0) as stream:
            stream.seek(offset)
            value = stream.read(size)
    except OSError as exc:
        raise SlabError(f"source read failed for {path}: {exc}") from exc
    if len(value) != size:
        raise SlabError(f"short source read for {path}: {len(value)}/{size}")
    return value


def _shape_bytes(shape: list[int], dtype: str) -> int:
    if dtype not in DTYPE_BYTES or not isinstance(shape, list):
        raise SlabError("unsupported dtype or shape")
    count = DTYPE_BYTES[dtype]
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent <= 0:
            raise SlabError("invalid tensor shape")
        count *= extent
    return count


def _slice(raw: bytes, shape: list[int], element_bytes: int,
           fragments: list[dict[str, Any]]) -> tuple[bytes, list[int]]:
    if not fragments:
        raise SlabError("payload entry has no source slices")
    pieces: list[bytes] = []
    local_shape = list(fragments[0]["shape"])
    dimension = fragments[0].get("dimension")
    for fragment in fragments:
        if fragment.get("operation", "copy") != "copy" or fragment.get("dimension") != dimension:
            raise SlabError("unsupported or inconsistent source transform")
        if dimension is None:
            pieces.append(raw)
            continue
        start, length = fragment.get("start"), fragment.get("length")
        if not isinstance(start, int) or not isinstance(length, int) or not 0 <= start <= shape[dimension] - length:
            raise SlabError("slice extent is outside tensor")
        if dimension == 0:
            row_bytes = _shape_bytes(shape[1:] or [1], "U8") * element_bytes
            pieces.append(raw[start * row_bytes:(start + length) * row_bytes])
        elif dimension == 1 and len(shape) == 2:
            row_bytes = shape[1] * element_bytes
            for row in range(shape[0]):
                base = row * row_bytes + start * element_bytes
                pieces.append(raw[base:base + length * element_bytes])
        else:
            raise SlabError("only replicated, row, and matrix-column slices are supported")
    if len(fragments) > 1:
        if dimension is None:
            raise SlabError("replicated tensor cannot have multiple slices")
        local_shape[dimension] = sum(int(fragment["shape"][dimension]) for fragment in fragments)
    return b"".join(pieces), local_shape


class _Sources:
    def __init__(self, checkpoint: Path, overlay: Path):
        self.checkpoint = checkpoint
        self.overlay = overlay
        self.overlay_start, self.overlay_index = read_safetensors_index(overlay)
        self._validated_files: set[Path] = set()

    def base(self, entry: dict[str, Any]) -> bytes:
        source = entry.get("source")
        if not isinstance(source, dict):
            raise SlabError(f"missing source provenance for {entry.get('name')}")
        path = self.checkpoint / str(source.get("file", ""))
        try:
            stat = path.stat()
        except OSError as exc:
            raise SlabError(f"missing checkpoint source: {path}") from exc
        if stat.st_size != source.get("file_size_bytes"):
            raise SlabError(f"checkpoint source size drift: {path.name}")
        if path not in self._validated_files:
            try:
                with path.open("rb") as stream:
                    raw_length = stream.read(8)
                    if len(raw_length) != 8:
                        raise SlabError(f"short checkpoint header: {path.name}")
                    length = int.from_bytes(raw_length, "little")
                    raw_header = stream.read(length)
            except OSError as exc:
                raise SlabError(f"checkpoint header read failed: {path.name}") from exc
            if hashlib.sha256(raw_header).hexdigest() != source.get("header_sha256"):
                raise SlabError(f"checkpoint header provenance drift: {path.name}")
            expected_blob = source.get("blob")
            resolved = path.resolve()
            observed_blob = resolved.name if resolved != path.absolute() else hashlib.sha256(raw_header).hexdigest()
            if observed_blob != expected_blob:
                raise SlabError(f"checkpoint blob provenance drift: {path.name}")
            self._validated_files.add(path)
        return _read_exact_at(path, int(source["absolute_offset_bytes"]), int(source["byte_count"]))

    def overlay_tensor(self, name: str) -> tuple[bytes, str, list[int]]:
        metadata = self.overlay_index.get(name)
        if metadata is None:
            raise SlabError(f"overlay component missing: {name}")
        dtype, shape, offsets = metadata.get("dtype"), metadata.get("shape"), metadata.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
            raise SlabError(f"overlay component metadata invalid: {name}")
        size = _shape_bytes(shape, dtype)
        if offsets[1] - offsets[0] != size:
            raise SlabError(f"overlay component extent invalid: {name}")
        return _read_exact_at(self.overlay, self.overlay_start + offsets[0], size), dtype, shape


def _adapt_fragments(fragments: list[dict[str, Any]], divisor: int) -> list[dict[str, Any]]:
    result = []
    for fragment in fragments:
        item = dict(fragment)
        if item.get("dimension") == 1:
            if item["start"] % divisor or item["length"] % divisor:
                raise SlabError("NVFP4 column slice is not component aligned")
            item["start"] //= divisor
            item["length"] //= divisor
            item["shape"] = list(item["shape"])
            item["shape"][1] //= divisor
        elif item.get("dimension") == 0:
            item["shape"] = list(item["shape"])
            item["shape"][1] //= divisor
        elif item.get("dimension") is None:
            item["shape"] = list(item["shape"])
            item["shape"][1] //= divisor
        return_shape = item.get("shape")
        if not isinstance(return_shape, list) or return_shape[1] <= 0:
            raise SlabError("invalid adapted NVFP4 slice")
        result.append(item)
    return result


def _overlay_components(entry: dict[str, Any], sources: _Sources,
                        source_sha256: str) -> list[dict[str, Any]]:
    base = entry["name"].removesuffix(".weight")
    full_shape = entry.get("full_shape")
    fragments = entry.get("source_slices")
    if not isinstance(full_shape, list) or len(full_shape) != 2 or not isinstance(fragments, list):
        raise SlabError(f"overlay source matrix contract drift: {entry['name']}")
    local_matrix = entry.get("local_shape")
    if not isinstance(local_matrix, list) or len(local_matrix) != 2:
        raise SlabError("overlay local matrix shape missing")
    if hashlib.sha256(sources.base(entry)).hexdigest() != source_sha256:
        raise SlabError(f"overlay source tensor provenance drift: {entry['name']}")
    components = []
    for suffix, divisor, expected_dtype in (
        ("weight", 2, "U8"), ("weight_scale", 16, "F8_E4M3"),
        ("weight_scale_2", None, "F32"), ("input_scale", None, "F32"),
    ):
        name = f"{base}.{suffix}"
        raw, dtype, shape = sources.overlay_tensor(name)
        if dtype != expected_dtype:
            raise SlabError(f"overlay component dtype drift: {name}")
        if divisor is None:
            if shape != [1] or len(raw) != 4:
                raise SlabError(f"overlay scalar ABI drift: {name}")
            local, local_shape = raw, [1]
            layout = "scalar"
        else:
            expected_shape = [full_shape[0], full_shape[1] // divisor]
            if shape != expected_shape:
                raise SlabError(f"overlay component shape drift: {name}")
            local, local_shape = _slice(raw, shape, 1, _adapt_fragments(fragments, divisor))
            layout = "packed_e2m1_row_major"
            if suffix == "weight_scale":
                local = sf_swizzle(local, local_matrix[0], local_matrix[1])
                local_shape = [len(local)]
                layout = "cutlass_sm121_sfb"
        components.append({"name": name, "dtype": dtype, "shape": local_shape,
                           "bytes": local, "layout": layout, "abi": MODEL_NVFP4_ABI})
    return components


def _base_component(entry: dict[str, Any], sources: _Sources) -> dict[str, Any]:
    raw = sources.base(entry)
    dtype, full_shape, fragments = entry.get("dtype"), entry.get("full_shape"), entry.get("source_slices")
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES or not isinstance(full_shape, list) or not isinstance(fragments, list):
        raise SlabError(f"base tensor metadata invalid: {entry.get('name')}")
    local, local_shape = _slice(raw, full_shape, DTYPE_BYTES[dtype], fragments)
    transforms = entry.get("transforms", fragments)
    if not isinstance(transforms, list):
        raise SlabError(f"base tensor transforms invalid: {entry.get('name')}")
    for transform in transforms[len(fragments):]:
        if transform.get("operation") != "zero_fill" or not isinstance(transform.get("shape"), list):
            raise SlabError(f"unsupported base tensor transform: {entry.get('name')}")
        local += b"\0" * _shape_bytes(transform["shape"], dtype)
    expected_shape = entry.get("local_shape")
    if not isinstance(expected_shape, list) or len(local) != _shape_bytes(expected_shape, dtype):
        raise SlabError(f"base tensor local extent drift: {entry.get('name')}")
    local_shape = expected_shape
    abi = MTP_FP8_ABI if entry.get("family") == "mtp_expert_fp8_block" else "native"
    layout = "checkpoint"
    if entry.get("family") == "target_expert_nvfp4":
        abi = MODEL_NVFP4_ABI
        if entry["name"].endswith(".weight_scale"):
            projection = entry["name"].rsplit(".", 2)[-2]
            if projection == "down_proj":
                rows, k = 2560, 640
            else:
                rows, k = 640, 2560
            local = sf_swizzle(local, rows, k)
            local_shape = [len(local)]
            layout = "cutlass_sm121_sfb"
    return {"name": entry["name"], "dtype": dtype, "shape": local_shape,
            "bytes": local, "layout": layout, "abi": abi}


def _write_slab(path: Path, components: list[dict[str, Any]], contract: SlabContract) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    offset = 0
    with path.open("xb") as output:
        for component in components:
            aligned = align_up(offset, contract.tensor_alignment_bytes)
            if aligned > offset:
                output.write(b"\0" * (aligned - offset))
            payload = component.pop("bytes")
            entries.append({**component, "offset_bytes": aligned, "length_bytes": len(payload)})
            output.write(payload)
            offset = aligned + len(payload)
        final_size = align_up(offset, contract.page_bytes)
        output.write(b"\0" * (final_size - offset))
    with path.open("rb", buffering=0) as stream:
        chunk_offset = 0
        while chunk_offset < final_size:
            length = min(contract.chunk_bytes, final_size - chunk_offset)
            data = stream.read(length)
            if len(data) != length or length % contract.page_bytes:
                raise SlabError(f"invalid materialized chunk at {chunk_offset}")
            chunks.append({"offset_bytes": chunk_offset, "length_bytes": length,
                           "sha256": hashlib.sha256(data).hexdigest()})
            chunk_offset += length
    return entries, chunks, final_size


def materialize(plan_path: Path, checkpoint: Path, overlay_root: Path, output_root: Path,
                contract: SlabContract = PINNED_CONTRACT) -> Path:
    """Create one immutable artifact. Inputs are read-only; failure leaves no final path."""
    plan = load_plan(plan_path, contract)
    validate_plan(plan, contract)
    overlay_manifest, overlay_path = validate_overlay(overlay_root, contract)
    overlay_names = {item["name"] for item in overlay_manifest["source"]["tensors"]}
    overlay_source_hashes = {
        item["name"]: item.get("sha256") for item in overlay_manifest["source"]["tensors"]
    }
    if any(not isinstance(value, str) or len(value) != 64 for value in overlay_source_hashes.values()):
        raise SlabError("overlay source tensor digest inventory is invalid")
    base_payload_names = {
        entry["name"] for slab in plan["slabs"].values() for entry in slab["entries"]
        if entry["entry_type"] == "payload"
    }
    if not overlay_names <= base_payload_names:
        raise SlabError("overlay inventory is not a subset of planned source payloads")
    sources = _Sources(checkpoint, overlay_path)
    artifact_seed = {"schema": SCHEMA, "revision": contract.revision,
                     "overlay_artifact_key": contract.artifact_key,
                     "overlay_sha256": contract.overlay_sha256,
                     "plan_sha256": sha256_file(plan_path), "tp_size": contract.tp_size,
                     "page_bytes": contract.page_bytes, "tensor_alignment_bytes": contract.tensor_alignment_bytes,
                     "chunk_bytes": contract.chunk_bytes}
    staging = output_root / f".qwen38-slab.{os.getpid()}.building"
    output_root.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        raise SlabError("refusing interrupted materialization owned by this process")
    staging.mkdir()
    try:
        slabs = {}
        substituted: set[str] = set()
        physical_names: set[tuple[int, str]] = set()
        for slab_key in SLAB_KEYS:
            rank = int(slab_key[4])
            components: list[dict[str, Any]] = []
            references = []
            for entry in plan["slabs"][slab_key]["entries"]:
                if entry["entry_type"] == "reference":
                    references.append({"name": entry["name"], "physical_owner": entry["physical_owner"]})
                    continue
                identity = (rank, entry["name"])
                if identity in physical_names:
                    raise SlabError(f"payload has multiple physical owners: {identity}")
                physical_names.add(identity)
                if entry["name"] in overlay_names:
                    components.extend(_overlay_components(entry, sources, overlay_source_hashes[entry["name"]]))
                    substituted.add(entry["name"])
                else:
                    components.append(_base_component(entry, sources))
            entries, chunks, byte_count = _write_slab(staging / f"{slab_key}.slab", components, contract)
            slabs[slab_key] = {"file": f"{slab_key}.slab", "bytes": byte_count,
                               "entries": entries, "references": references, "chunks": chunks}
        payload_by_owner = {
            key: {entry["name"] for entry in slab["entries"]}
            for key, slab in slabs.items()
        }
        for slab in slabs.values():
            for reference in slab["references"]:
                if reference["name"] not in payload_by_owner.get(reference["physical_owner"], set()):
                    raise SlabError(f"shared reference has no physical payload owner: {reference['name']}")
        inventory = plan.get("inventory")
        if isinstance(inventory, dict):
            observed_source_names = {name for _, name in physical_names}
            if len(observed_source_names) != inventory.get("classified_entries"):
                raise SlabError("planned source inventory is not completely physically assigned")
        if substituted != overlay_names:
            missing = sorted(overlay_names - substituted)[:3]
            raise SlabError(f"overlay substitution incomplete: {len(substituted)}/{len(overlay_names)} missing={missing}")
        manifest = {**artifact_seed,
                    "source": {"checkpoint": str(checkpoint.absolute()),
                               "plan": str(plan_path.absolute()),
                               "overlay": str(overlay_root.absolute())},
                    "overlay_substitutions": len(substituted),
                    "shared_payload_policy": "target-owned-mtp-reference-read-once",
                    "production_source_format": "rank-slab-only-no-safetensors-traversal",
                    "direct_io": {"required": True, "flag": "O_DIRECT", "regular_file_mmap": False,
                                  "gds": False, "cufile": False, "nvidia_fs": False},
                    "abis": {"target_nvfp4": MODEL_NVFP4_ABI, "mtp_experts": MTP_FP8_ABI},
                    "slabs": slabs}
        artifact_key = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
        manifest["artifact_key"] = artifact_key
        destination = output_root / artifact_key
        if destination.exists():
            raise SlabError("refusing existing immutable artifact")
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
        (staging / "manifest.json").write_bytes(manifest_bytes)
        os.chmod(staging / "manifest.json", 0o444)
        for key in SLAB_KEYS:
            os.chmod(staging / f"{key}.slab", 0o444)
        staging.rename(destination)
        os.chmod(destination, 0o555)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(materialize(args.plan, args.checkpoint, args.overlay, args.output_root))
    except (OSError, SlabError) as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
