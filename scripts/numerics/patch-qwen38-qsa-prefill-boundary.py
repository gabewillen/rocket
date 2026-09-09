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
PINNED_INDEXER_SOURCE_SHA256 = (
    "e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066"
)


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("pinned QSA capture anchor changed")
    return source.replace(before, after, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--indexer-input", type=Path)
    parser.add_argument("--indexer-output", type=Path)
    args = parser.parse_args()
    if (args.indexer_input is None) != (args.indexer_output is None):
        raise RuntimeError("indexer input and output must be supplied together")
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
_ROCKET_QSA_PREFILL_V2_KEY = None


def _rocket_publish_qsa_c1_input(owner, positions):
    global _ROCKET_QSA_PREFILL_CAPTURED
    hidden = getattr(owner.indexer, "_rocket_c1_hidden", None)
    index_query = getattr(owner.indexer, "_rocket_c1_index_query", None)
    if hidden is None or index_query is None:
        raise RuntimeError("authenticated QSA c1 input is unavailable")
    values = {
        "row35_hidden": hidden.detach().contiguous().clone(),
        "row35_index_query": index_query.detach().contiguous().clone(),
        "row35_positions": positions.detach().contiguous().clone(),
    }
    expected = {
        "row35_hidden": ((1, 2560), torch.bfloat16),
        "row35_index_query": ((1, 4, 128), torch.bfloat16),
        "row35_positions": ((3, 1), torch.int64),
    }
    if any(tuple(values[name].shape) != shape or values[name].dtype != dtype
           for name, (shape, dtype) in expected.items()):
        raise RuntimeError("authenticated QSA c1 input layout changed")
    if not torch.equal(values["row35_positions"],
                       torch.full((3, 1), 35, dtype=torch.int64,
                                  device=positions.device)):
        raise RuntimeError("authenticated QSA c1 position changed")
    root = Path(os.environ["ROCKET_QWEN38_QSA_PREFILL_BOUNDARY_ROOT"])
    parent = root / _ROCKET_QSA_PREFILL_V2_KEY
    parent_manifest = json.loads((parent / "manifest.json").read_bytes())
    authenticated_parent = dict(parent_manifest)
    parent_key = authenticated_parent.pop("artifact_key", None)
    if (parent_key != _ROCKET_QSA_PREFILL_V2_KEY or
            hashlib.sha256(json.dumps(
                authenticated_parent, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest() != parent_key or
            parent_manifest.get("schema") !=
                "rocket.qwen38.qsa-prefill-boundary.v2"):
        raise RuntimeError("authenticated QSA v2 parent changed")
    temporary = root / (".bundle-v3.tmp." + str(os.getpid()))
    temporary.mkdir()
    entries = []
    for entry in parent_manifest["tensors"]:
        data = (parent / entry["file"]).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError("authenticated QSA v2 payload changed")
        (temporary / entry["file"]).write_bytes(data)
        entries.append(dict(entry))
    for name, tensor in values.items():
        data = tensor.view(torch.uint8).cpu().numpy().tobytes()
        filename = name + ".bin"
        (temporary / filename).write_bytes(data)
        entries.append({"name": name, "file": filename,
                        "dtype": str(tensor.dtype).removeprefix("torch."),
                        "shape": list(tensor.shape), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema": "rocket.qwen38.qsa-c1-input.v3",
        "parent_artifact_key": _ROCKET_QSA_PREFILL_V2_KEY,
        "oracle_manifest_sha256": parent_manifest["oracle_manifest_sha256"],
        "implementation": parent_manifest["implementation"],
        "source_sha256": parent_manifest["source_sha256"],
        "indexer_source_sha256":
            "e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066",
        "rank": 0, "layer": 3, "rows": 35, "c1_position": 35,
        "generation_index": 0, "tensors": entries,
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
        raise RuntimeError("QSA c1 input already published")
    os.replace(temporary, destination)
    _ROCKET_QSA_PREFILL_CAPTURED = True


def _rocket_capture_qsa_prefill_boundary(owner, positions, query, key, value,
                                          selected, main_metadata,
                                          raw_metadata, compressed_metadata):
    global _ROCKET_QSA_PREFILL_CAPTURED, _ROCKET_QSA_PREFILL_V2_KEY
    if owner.indexer.layer_id != 3:
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
    if _ROCKET_QSA_PREFILL_V2_KEY is not None:
        if _ROCKET_QSA_PREFILL_CAPTURED:
            return
        if rows != 1:
            raise RuntimeError("authenticated QSA c1 row count changed")
        _rocket_publish_qsa_c1_input(owner, positions[..., :rows])
        return
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
    raw_state_slots = torch.unique(
        values["raw_slots"][values["raw_slots"] >= 0], sorted=True
    )
    compressed_state_slots = torch.unique(
        values["compressed_slots"][values["compressed_slots"] >= 0], sorted=True
    )
    raw_cache = owner.indexer.raw_key_cache.kv_cache.reshape(
        -1, 1, owner.indexer.raw_key_cache.kv_cache.shape[-1]
    )
    compressed_cache = owner.indexer.compressed_key_cache.kv_cache.reshape(
        -1, 1, owner.indexer.index_head_dim
    )
    values["raw_state_slots"] = raw_state_slots
    values["raw_state"] = raw_cache.index_select(0, raw_state_slots).contiguous()
    values["compressed_state_slots"] = compressed_state_slots
    values["compressed_state"] = compressed_cache.index_select(
        0, compressed_state_slots
    ).contiguous()
    names = (
        *names, "k_scale", "v_scale", "raw_state_slots", "raw_state",
        "compressed_state_slots", "compressed_state",
    )
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
        "raw_state_slots": ((4,), torch.int64),
        "raw_state": ((4, 1, 140), torch.bfloat16),
        "compressed_state_slots": ((8,), torch.int64),
        "compressed_state": ((8, 1, 128), compressed_cache.dtype),
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
        "schema": "rocket.qwen38.qsa-prefill-boundary.v2",
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
    _ROCKET_QSA_PREFILL_V2_KEY = key_hash
'''
    source = replace_once(
        source,
        "from .indexer_qsa import QSAIndexer\n",
        "from .indexer_qsa import QSAIndexer\n" + helper,
    )
    source = replace_once(
        source,
        "        selected = self.indexer(\n",
        "        if self.indexer.layer_id == 3 and num_tokens == 1:\n"
        "            self.indexer._rocket_c1_hidden = (\n"
        "                hidden_states.detach().contiguous().clone()\n"
        "            )\n"
        "        selected = self.indexer(\n",
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
    if args.indexer_input is not None:
        indexer_payload = args.indexer_input.read_bytes()
        if hashlib.sha256(indexer_payload).hexdigest() != PINNED_INDEXER_SOURCE_SHA256:
            raise RuntimeError("pinned QSA indexer source identity changed")
        indexer = indexer_payload.decode()
        indexer = replace_once(
            indexer,
            "        if self.skip_topk:\n",
            "        if self.layer_id == 3 and num_tokens == 1:\n"
            "            self._rocket_c1_index_query = q.detach().contiguous().clone()\n\n"
            "        if self.skip_topk:\n",
        )
        args.indexer_output.write_text(indexer)


if __name__ == "__main__":
    main()
