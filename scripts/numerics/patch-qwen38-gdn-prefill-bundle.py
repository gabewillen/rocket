#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add one authenticated layer-0 GDN prefill tensor-bundle capture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "f606cd94485911c03badb27f466d53243dd787a37e2198347971afcc2f558f1b"
)


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("pinned GDN prefill capture anchor changed")
    return source.replace(before, after, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINNED_SOURCE_SHA256:
        raise RuntimeError("pinned GDN source identity changed")
    source = payload.decode()
    source = replace_once(
        source,
        "import os\nfrom typing import Literal",
        "import os\nimport hashlib\nimport json\nfrom pathlib import Path\n"
        "from typing import Literal",
    )
    source = replace_once(
        source,
        "    elif (\n"
        "        current_platform.is_device_capability_family(100)\n"
        "        and head_k_dim == 128\n"
        "        and current_platform.get_cuda_runtime_major() >= 13\n"
        "    ):\n",
        "    elif (\n"
        "        (current_platform.is_device_capability_family(100)\n"
        "         or current_platform.is_device_capability_family(120))\n"
        "        and head_k_dim == 128\n"
        "        and current_platform.get_cuda_runtime_major() >= 13\n"
        "    ):\n",
    )
    helper = r'''
_ROCKET_GDN_PREFILL_BUNDLE_CAPTURED = False


def _rocket_capture_gdn_prefill_bundle(q, k, v, log_decay, beta, initial_state):
    global _ROCKET_GDN_PREFILL_BUNDLE_CAPTURED
    if os.getenv("ROCKET_QWEN38_K0_GDN_CAPTURE_ACTIVE") != "1":
        return
    if _ROCKET_GDN_PREFILL_BUNDLE_CAPTURED:
        return
    from vllm.distributed import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    if rank != 0:
        return
    expected = {
        "q": ((35, 8, 128), torch.bfloat16),
        "k": ((35, 8, 128), torch.bfloat16),
        "v": ((35, 24, 128), torch.bfloat16),
        "log_decay": ((35, 24), torch.float32),
        "beta": ((35, 24), torch.float32),
        "initial_state": ((1, 24, 128, 128), torch.float32),
    }
    values = {"q": q, "k": k, "v": v, "log_decay": log_decay,
              "beta": beta, "initial_state": initial_state}
    if any(tuple(values[name].shape) != shape or values[name].dtype != dtype
           for name, (shape, dtype) in expected.items()):
        raise RuntimeError("authenticated GDN prefill tensor layout changed")
    root = Path(os.environ["ROCKET_QWEN38_GDN_PREFILL_BUNDLE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / (".bundle.tmp." + str(os.getpid()))
    if temporary.exists():
        raise RuntimeError("GDN prefill temporary bundle already exists")
    temporary.mkdir()
    entries = []
    for name in ("q", "k", "v", "log_decay", "beta", "initial_state"):
        value = values[name].detach().contiguous()
        data = value.view(torch.uint8).cpu().numpy().tobytes()
        filename = name + ".bin"
        (temporary / filename).write_bytes(data)
        entries.append({"name": name, "file": filename,
                        "dtype": str(value.dtype).removeprefix("torch."),
                        "shape": list(value.shape), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "schema": "rocket.qwen38.gdn-prefill-tensors.v1",
        "oracle_manifest_sha256":
            "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b",
        "implementation": "vllm:8e685d198:flashinfer:0.6.17:gdn-prefill-sm121a",
        "rank": 0, "layer": 0, "rows": 35, "tensors": entries,
    }
    canonical = json.dumps(manifest, sort_keys=True,
                           separators=(",", ":")).encode()
    key = hashlib.sha256(canonical).hexdigest()
    manifest["artifact_key"] = key
    (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    destination = root / key
    if destination.exists():
        raise RuntimeError("GDN prefill bundle already published")
    os.replace(temporary, destination)
    _ROCKET_GDN_PREFILL_BUNDLE_CAPTURED = True
'''
    source = replace_once(source, "logger = init_logger(__name__)\n", helper + "\nlogger = init_logger(__name__)\n")
    source = replace_once(
        source,
        "    if cu_seqlens is not None:\n"
        "        cu_seqlens = cu_seqlens.to(torch.int64)\n"
        "    result = chunk_gated_delta_rule_fi(\n",
        "    if cu_seqlens is not None:\n"
        "        cu_seqlens = cu_seqlens.to(torch.int64)\n"
        "    _rocket_capture_gdn_prefill_bundle(\n"
        "        q, k, v, fi_g, fi_beta, fi_state)\n"
        "    result = chunk_gated_delta_rule_fi(\n",
    )
    args.output.write_text(source)


if __name__ == "__main__":
    main()
