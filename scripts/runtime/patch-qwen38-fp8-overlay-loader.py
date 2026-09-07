#!/usr/bin/env python3
"""Patch pinned vLLM weight_utils.py for a verified Qwen3.8 FP8 overlay."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


MARKER = "ROCKET_QWEN38_FP8_OVERLAY_V1"
FUNCTION_ANCHOR = "def safetensors_weights_iterator(\n"
SORT_ANCHOR = "    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)\n"
SORT_PATCH = SORT_ANCHOR + "    rocket_overlay = _rocket_qwen38_fp8_overlay_preflight(sorted_files, safetensors_load_strategy)\n"
LAZY_ANCHOR = '''                    # Bound staging to one tensor and avoid CUDA copying directly
                    # from a 64 KiB-page safetensors mmap.
                    param = f.get_tensor(name).clone()
                    yield name, param
'''
LAZY_PATCH = '''                    # Preserve bounded staging on 64 KiB hosts for base and overlay tensors.
                    if rocket_overlay is not None and name in rocket_overlay["selected"]:
                        prefix = name.removesuffix(".weight")
                        with safe_open(rocket_overlay["overlay_file"], framework="pt") as overlay:
                            for overlay_name in (
                                name, prefix + ".weight_scale", prefix + ".input_scale"
                            ):
                                yield overlay_name, overlay.get_tensor(overlay_name).clone()
                        rocket_overlay["seen"].add(name)
                        continue
                    param = f.get_tensor(name).clone()
                    yield name, param
    if rocket_overlay is not None and rocket_overlay["seen"] != rocket_overlay["selected"]:
        missing = sorted(rocket_overlay["selected"] - rocket_overlay["seen"])
        raise ValueError(f"Qwen3.8 FP8 overlay partial replacement: {missing[:3]}")
'''

HELPER = r'''
# ROCKET_QWEN38_FP8_OVERLAY_V1
_ROCKET_QWEN38_FP8_MANIFEST_ENV = "ROCKET_QWEN38_FP8_OVERLAY_MANIFEST"
_ROCKET_QWEN38_FP8_CONFIG_ENV = "ROCKET_QWEN38_FP8_QUANT_CONFIG"
_ROCKET_QWEN38_FP8_SCHEMA = "rocket.qwen38.linear-fp8-overlay.v1"
_ROCKET_QWEN38_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
_ROCKET_QWEN38_SELECTED = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.linear_attn\."
    r"(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)\.weight$"
)


def _rocket_qwen38_canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rocket_qwen38_file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rocket_qwen38_shard_identity(path):
    lexical = Path(path).absolute()
    if not lexical.is_symlink():
        raise ValueError(f"Qwen3.8 FP8 overlay shard is not a snapshot symlink: {lexical.name}")
    target = lexical.resolve(strict=True)
    with open(target, "rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Qwen3.8 FP8 overlay truncated header: {lexical.name}")
        size = int.from_bytes(prefix, "little")
        if not 0 < size <= 64 * 1024 * 1024:
            raise ValueError(f"Qwen3.8 FP8 overlay header bound: {lexical.name}")
        raw = stream.read(size)
    if len(raw) != size:
        raise ValueError(f"Qwen3.8 FP8 overlay truncated header: {lexical.name}")
    return {
        "target": target.name,
        "size": target.stat().st_size,
        "header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
    }


def _rocket_qwen38_headers(paths):
    tensors = {}
    for path in paths:
        with open(path, "rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Qwen3.8 FP8 overlay truncated header: {path}")
            size = int.from_bytes(prefix, "little")
            if not 0 < size <= 64 * 1024 * 1024:
                raise ValueError(f"Qwen3.8 FP8 overlay header bound: {path}")
            raw = stream.read(size)
        header = json.loads(raw)
        data_start = 8 + size
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Qwen3.8 FP8 overlay duplicate base tensor: {name}")
            tensors[name] = (Path(path), data_start, meta)
    return tensors


def _rocket_qwen38_range_sha256(path, offset, size):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        stream.seek(offset)
        remaining = size
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Qwen3.8 FP8 overlay short source range: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _rocket_qwen38_fp8_overlay_preflight(sorted_files, strategy):
    manifest_name = os.getenv(_ROCKET_QWEN38_FP8_MANIFEST_ENV)
    if not manifest_name:
        return None
    if strategy in ("eager", "torchao"):
        raise ValueError("Qwen3.8 FP8 overlay requires lazy safetensors loading")
    manifest_path = Path(manifest_name).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != _ROCKET_QWEN38_FP8_SCHEMA:
        raise ValueError("Qwen3.8 FP8 overlay schema mismatch")
    source = manifest.get("source")
    overlay = manifest.get("overlay")
    if not isinstance(source, dict) or not isinstance(overlay, dict):
        raise ValueError("Qwen3.8 FP8 overlay source/overlay contract missing")
    if source.get("revision") != _ROCKET_QWEN38_REVISION:
        raise ValueError("Qwen3.8 FP8 overlay revision mismatch")
    if not sorted_files:
        raise ValueError("Qwen3.8 FP8 overlay base path is not the pinned revision")
    snapshot_parents = {Path(path).absolute().parent for path in sorted_files}
    if len(snapshot_parents) != 1 or next(iter(snapshot_parents)).name != _ROCKET_QWEN38_REVISION:
        raise ValueError("Qwen3.8 FP8 overlay lexical base path is not the pinned revision")
    shard_contract = source.get("shards")
    if not isinstance(shard_contract, dict) or set(shard_contract) != {Path(path).name for path in sorted_files}:
        raise ValueError("Qwen3.8 FP8 overlay shard identity set mismatch")
    for path in sorted_files:
        lexical = Path(path).absolute()
        expected = shard_contract[lexical.name]
        if not isinstance(expected, dict) or _rocket_qwen38_shard_identity(lexical) != expected:
            raise ValueError(f"Qwen3.8 FP8 overlay shard target identity mismatch: {lexical.name}")
    quant_config_name = os.getenv(_ROCKET_QWEN38_FP8_CONFIG_ENV)
    quant_contract = manifest.get("quant_config")
    if not quant_config_name or not isinstance(quant_contract, dict):
        raise ValueError("Qwen3.8 FP8 overlay quant-config contract missing")
    quant_config_path = Path(quant_config_name).resolve(strict=True)
    if _rocket_qwen38_file_sha256(quant_config_path) != quant_contract.get("sha256"):
        raise ValueError("Qwen3.8 FP8 overlay quant-config hash mismatch")
    quant_config = json.loads(quant_config_path.read_text())
    quantization = quant_config.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("Qwen3.8 FP8 overlay requires ModelOpt MIXED_PRECISION")
    key_payload = {"source": source, "overlay": overlay, "quant_config": quant_contract}
    if manifest.get("artifact_key") != _rocket_qwen38_canonical_sha256(key_payload):
        raise ValueError("Qwen3.8 FP8 overlay artifact key mismatch")
    relative = overlay.get("file")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Qwen3.8 FP8 overlay file path is unsafe")
    overlay_file = (manifest_path.parent / relative).resolve()
    if _rocket_qwen38_file_sha256(overlay_file) != overlay.get("sha256"):
        raise ValueError("Qwen3.8 FP8 overlay file hash mismatch")
    entries = source.get("tensors")
    if not isinstance(entries, list) or len(entries) != 180:
        raise ValueError("Qwen3.8 FP8 overlay requires exactly 180 source tensors")
    selected = set()
    layers = {}
    base = _rocket_qwen38_headers(sorted_files)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Qwen3.8 FP8 overlay source entry is not an object")
        name = entry.get("name")
        match = _ROCKET_QWEN38_SELECTED.fullmatch(name or "")
        if match is None or name in selected:
            raise ValueError(f"Qwen3.8 FP8 overlay extra/duplicate selection: {name!r}")
        selected.add(name)
        layers.setdefault(int(match.group(1)), set()).add(match.group(2))
        if name not in base:
            raise ValueError(f"Qwen3.8 FP8 overlay source tensor missing: {name}")
        path, data_start, meta = base[name]
        if meta.get("dtype") != "BF16" or meta.get("shape") != entry.get("shape"):
            raise ValueError(f"Qwen3.8 FP8 overlay source dtype/shape drift: {name}")
        offsets = meta.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"Qwen3.8 FP8 overlay source offsets invalid: {name}")
        digest = _rocket_qwen38_range_sha256(path, data_start + offsets[0], offsets[1] - offsets[0])
        if digest != entry.get("sha256"):
            raise ValueError(f"Qwen3.8 FP8 overlay source hash mismatch: {name}")
    projections = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    if len(layers) != 36 or any(value != projections for value in layers.values()):
        raise ValueError("Qwen3.8 FP8 overlay is a partial 36-layer family")
    quantized_layers = quantization.get("quantized_layers")
    excludes = quantization.get("exclude_modules")
    if not isinstance(quantized_layers, dict) or not isinstance(excludes, list):
        raise ValueError("Qwen3.8 FP8 overlay quantized_layers/excludes missing")
    selected_prefixes = {name.removesuffix(".weight") for name in selected}
    if any(quantized_layers.get(prefix, {}).get("quant_algo") != "FP8" for prefix in selected_prefixes):
        raise ValueError("Qwen3.8 FP8 overlay quant method is not FP8 for every selected tensor")
    if any(any(fnmatch.fnmatch(prefix, pattern) for pattern in excludes) for prefix in selected_prefixes):
        raise ValueError("Qwen3.8 FP8 overlay selected tensor remains excluded")
    overlay_headers = _rocket_qwen38_headers([overlay_file])
    expected = set()
    for entry in entries:
        name = entry["name"]
        prefix = name.removesuffix(".weight")
        expected.update((name, prefix + ".weight_scale", prefix + ".input_scale"))
        contracts = {
            name: ("F8_E4M3", entry["shape"]),
            prefix + ".weight_scale": ("F32", [1]),
            prefix + ".input_scale": ("F32", [1]),
        }
        for tensor_name, (dtype, shape) in contracts.items():
            found = overlay_headers.get(tensor_name)
            if found is None or found[2].get("dtype") != dtype or found[2].get("shape") != shape:
                raise ValueError(f"Qwen3.8 FP8 overlay tensor ABI mismatch: {tensor_name}")
    actual = set(overlay_headers)
    if actual != expected:
        raise ValueError(f"Qwen3.8 FP8 overlay extra/missing tensors: expected={len(expected)} actual={len(actual)}")
    return {"overlay_file": str(overlay_file), "selected": selected, "seen": set()}

'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"pinned weight_utils source drift: {label} anchor count={count}")
    return source.replace(old, new)


def snapshot_paths_match_revision(paths: list[Path], revision: str) -> bool:
    """Validate lexical snapshot ownership while allowing HF blob symlinks."""
    if not paths:
        return False
    parents = {path.absolute().parent for path in paths}
    return (
        len(parents) == 1
        and next(iter(parents)).name == revision
        and all(path.is_symlink() and path.resolve(strict=True).is_file() for path in paths)
    )


def patched(source: str) -> str:
    if MARKER in source:
        raise ValueError("weight_utils source is already patched")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("pinned weight_utils source drift: iterator definition")
    result = source.replace(FUNCTION_ANCHOR, HELPER + FUNCTION_ANCHOR)
    result = replace_once(result, SORT_ANCHOR, SORT_PATCH, "sorted files")
    result = replace_once(result, LAZY_ANCHOR, LAZY_PATCH, "64 KiB clone")
    ast.parse(result)
    return result


def patch(path: Path) -> None:
    source = path.read_text()
    result = patched(source)
    path.write_text(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_utils_py", type=Path)
    args = parser.parse_args()
    try:
        patch(args.weight_utils_py)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
