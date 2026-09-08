#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture one authenticated layer-3 QSA M35 state boundary."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "ee5de40742ad48a6064ea24b99a285ff69c47d57bbb170f57c4eef71567a1df3"
)


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("pinned QSA capture anchor changed")
    return source.replace(before, after, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINNED_SOURCE_SHA256:
        raise RuntimeError("pinned QSA source identity changed")
    source = payload.decode()
    source = replace_once(
        source,
        "from __future__ import annotations\n\nfrom typing import ClassVar, cast",
        "from __future__ import annotations\n\n"
        "import hashlib\nimport json\nimport os\nfrom pathlib import Path\n"
        "from typing import ClassVar, cast",
    )
    helper = r'''
_ROCKET_QSA_PREFILL_CAPTURED = False
_ROCKET_QSA_PREFILL_CHUNKS = []


def _rocket_capture_qsa_prefill_boundary(owner, positions, query, key, value,
                                          selected, main_metadata,
                                          raw_metadata, compressed_metadata):
    global _ROCKET_QSA_PREFILL_CAPTURED
    if _ROCKET_QSA_PREFILL_CAPTURED or owner.indexer.layer_id != 3:
        return
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.models.qwen3_8_flash_next.nvidia.model import _rocket_k0_oracle
    oracle = _rocket_k0_oracle()
    if oracle is None or not oracle.active_forward:
        return
    rank = get_tensor_model_parallel_rank()
    if rank != 0:
        return
    rows = query.shape[0]
    main_slots = main_metadata.slot_mapping[:rows].to(torch.int64)
    raw_slots = raw_metadata.slot_mapping[:rows].to(torch.int64)
    compressed_slots = compressed_metadata.slot_mapping[:rows].to(torch.int64)
    if (main_slots < 0).any():
        raise RuntimeError("authenticated QSA main slots changed")
    main_cache = owner.kv_cache.transpose(1, 2)
    main_key, main_value = main_cache.split(owner.head_dim, dim=-1)
    main_key = main_key.reshape(-1, owner.num_kv_heads, owner.head_dim)
    main_value = main_value.reshape(-1, owner.num_kv_heads, owner.head_dim)
    written_key = main_key.index_select(0, main_slots).contiguous()
    written_value = main_value.index_select(0, main_slots).contiguous()
    chunk = {
        "positions": positions[..., :rows].detach().contiguous().clone(),
        "query": query.detach().contiguous().clone(),
        "key": key.detach().contiguous().clone(),
        "value": value.detach().contiguous().clone(),
        "selected": selected.detach().contiguous().clone(),
        "main_slots": main_slots.detach().contiguous().clone(),
        "raw_slots": raw_slots.detach().contiguous().clone(),
        "compressed_slots": compressed_slots.detach().contiguous().clone(),
        "logical_positions": raw_metadata.logical_positions[:rows].detach().contiguous().clone(),
        "token_to_req": raw_metadata.token_to_req[:rows].detach().contiguous().clone(),
        "written_key": written_key.detach().contiguous().clone(),
        "written_value": written_value.detach().contiguous().clone(),
    }
    _ROCKET_QSA_PREFILL_CHUNKS.append(chunk)
    total_rows = sum(item["query"].shape[0] for item in _ROCKET_QSA_PREFILL_CHUNKS)
    if total_rows > 35:
        raise RuntimeError("authenticated QSA prefill rows changed")
    if total_rows < 35:
        return
    names = tuple(chunk)
    values = {}
    for name in names:
        axis = -1 if name == "positions" and chunk[name].ndim == 2 else 0
        values[name] = torch.cat(
            [item[name] for item in _ROCKET_QSA_PREFILL_CHUNKS], dim=axis
        ).contiguous()
    values["k_scale"] = owner._k_scale.detach().to(torch.float32).reshape(1).clone()
    values["v_scale"] = owner._v_scale.detach().to(torch.float32).reshape(1).clone()
    names = (*names, "k_scale", "v_scale")
    expected = {
        "query": ((35, 12, 256), torch.bfloat16),
        "key": ((35, 1, 256), torch.bfloat16),
        "value": ((35, 1, 256), torch.bfloat16),
        "selected": ((35, owner.indexer.output_width), torch.int32),
        "main_slots": ((35,), torch.int64),
        "raw_slots": ((35,), torch.int64),
        "compressed_slots": ((35,), torch.int64),
        "logical_positions": ((35,), torch.int64),
        "token_to_req": ((35,), torch.int32),
        "written_key": ((35, 1, 256), owner.kv_cache.dtype),
        "written_value": ((35, 1, 256), owner.kv_cache.dtype),
        "k_scale": ((1,), torch.float32),
        "v_scale": ((1,), torch.float32),
    }
    if any(tuple(values[name].shape) != shape or values[name].dtype != dtype
           for name, (shape, dtype) in expected.items()):
        raise RuntimeError("authenticated QSA prefill tensor layout changed")
    root = Path(os.environ["ROCKET_QWEN38_QSA_PREFILL_BOUNDARY_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / (".bundle.tmp." + str(os.getpid()))
    temporary.mkdir()
    entries = []
    for name in names:
        tensor = values[name]
        data = tensor.view(torch.uint8).cpu().numpy().tobytes()
        filename = name + ".bin"
        (temporary / filename).write_bytes(data)
        entries.append({"name": name, "file": filename,
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "shape": list(tensor.shape), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema": "rocket.qwen38.qsa-prefill-boundary.v1",
        "oracle_manifest_sha256":
            "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b",
        "implementation": "vllm:8e685d198:qwen38-qsa-prefill",
        "source_sha256":
            "ee5de40742ad48a6064ea24b99a285ff69c47d57bbb170f57c4eef71567a1df3",
        "rank": 0, "layer": 3, "rows": 35, "generation_index": 0,
        "tensors": entries,
    }
    canonical = json.dumps(manifest, sort_keys=True,
                           separators=(",", ":")).encode()
    key_hash = hashlib.sha256(canonical).hexdigest()
    manifest["artifact_key"] = key_hash
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    destination = root / key_hash
    if destination.exists():
        raise RuntimeError("QSA prefill boundary already published")
    os.replace(temporary, destination)
    _ROCKET_QSA_PREFILL_CAPTURED = True
'''
    source = replace_once(
        source,
        "from .indexer_qsa import QSAIndexer\n",
        "from .indexer_qsa import QSAIndexer\n" + helper,
    )
    source = replace_once(
        source,
        "        impl.forward_qsa(\n",
        "        compressed_metadata = cast(\n"
        "            QSAForwardMetadata,\n"
        "            metadata[self.indexer.compressed_key_cache.prefix],\n"
        "        )\n"
        "        _rocket_capture_qsa_prefill_boundary(\n"
        "            self, positions, query, key, value, selected,\n"
        "            main_metadata, side_metadata, compressed_metadata,\n"
        "        )\n"
        "        impl.forward_qsa(\n",
    )
    args.output.write_text(source)


if __name__ == "__main__":
    main()
