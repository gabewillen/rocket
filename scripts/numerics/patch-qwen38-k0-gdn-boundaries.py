#!/usr/bin/env python3
"""Add bounded first-row GDN captures to the pinned vLLM oracle overlay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ORACLE_SHA256 = "4714a2fa9994c728543fb4cff8a960aea9f834e62f617f6fb08ea28b71f9e358"
GDN_SHA256 = "81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0"


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("pinned GDN capture patch anchor differs")
    return source.replace(before, after, 1)


def read_pinned(path: Path, expected: str) -> str:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise RuntimeError("pinned GDN capture source identity differs")
    return payload.decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-input", type=Path, required=True)
    parser.add_argument("--oracle-output", type=Path, required=True)
    parser.add_argument("--gdn-input", type=Path, required=True)
    parser.add_argument("--gdn-output", type=Path, required=True)
    args = parser.parse_args()

    oracle = read_pinned(args.oracle_input, ORACLE_SHA256)
    oracle = replace_once(
        oracle,
        "        self.active_forward = True\n        self.forward_names = []",
        "        self.active_forward = True\n"
        "        os.environ['ROCKET_QWEN38_K0_GDN_CAPTURE_ACTIVE'] = '1'\n"
        "        self.forward_names = []",
    )
    args.oracle_output.write_text(oracle)

    gdn = read_pinned(args.gdn_input, GDN_SHA256)
    gdn = replace_once(
        gdn,
        "import os\nfrom typing import Literal",
        "import os\nimport hashlib\nimport json\nfrom pathlib import Path\nfrom typing import Literal",
    )
    helper = r'''
_ROCKET_K0_GDN_SEEN = set()
_ROCKET_K0_GDN_EXTENTS = {
    "qkvz": 8192,
    "ba": 48,
    "core": 3072,
    "normalized": 3072,
}


def _rocket_k0_gdn_save(prefix, name, tensor):
    if os.getenv("ROCKET_QWEN38_K0_GDN_CAPTURE_ACTIVE") != "1":
        return
    if ".layers.0." not in prefix or name in _ROCKET_K0_GDN_SEEN:
        return
    from vllm.distributed import get_tensor_model_parallel_rank
    rank = get_tensor_model_parallel_rank()
    if rank not in (0, 1) or name not in _ROCKET_K0_GDN_EXTENTS:
        raise RuntimeError("bounded GDN capture identity differs")
    value = tensor[:1].detach().contiguous()
    if value.dtype != torch.bfloat16 or value.numel() != _ROCKET_K0_GDN_EXTENTS[name]:
        raise RuntimeError("bounded GDN capture layout differs")
    payload = value.view(torch.uint8).cpu().numpy().tobytes()
    output = Path(os.environ["ROCKET_QWEN38_K0_GDN_CAPTURE_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    stem = f"rank{rank}-{name}"
    temporary = output / ("." + stem + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, output / (stem + ".bin"))
    record = {
        "schema": "rocket.qwen38.k0-gdn-boundaries.v1",
        "rank": rank,
        "name": name,
        "dtype": "bfloat16",
        "elements": value.numel(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    temporary = output / ("." + stem + ".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, output / (stem + ".json"))
    _ROCKET_K0_GDN_SEEN.add(name)
'''
    gdn = replace_once(gdn, "logger = init_logger(__name__)\n", helper + "\nlogger = init_logger(__name__)\n")
    gdn = replace_once(
        gdn,
        "        # ============================================================\n"
        "        # Part 2: Core Attention (Custom Op)\n",
        "        _rocket_k0_gdn_save(\n"
        "            self.prefix, 'qkvz',\n"
        "            torch.cat((mixed_qkv, z.reshape(num_tokens, -1)), dim=-1))\n"
        "        _rocket_k0_gdn_save(\n"
        "            self.prefix, 'ba', torch.cat((b, a), dim=-1))\n\n"
        "        # ============================================================\n"
        "        # Part 2: Core Attention (Custom Op)\n",
    )
    gdn = replace_once(
        gdn,
        "        # ============================================================\n"
        "        # Part 3: Output Projection\n"
        "        # ============================================================\n"
        "        return self._output_projection(core_attn_out, z)\n",
        "        _rocket_k0_gdn_save(self.prefix, 'core', core_attn_out)\n\n"
        "        # ============================================================\n"
        "        # Part 3: Output Projection\n"
        "        # ============================================================\n"
        "        return self._output_projection(core_attn_out, z)\n",
    )
    projection_start = gdn.index("    def _output_projection(")
    projection_end = gdn.index("    def forward_hip(", projection_start)
    projection = gdn[projection_start:projection_end]
    projection = replace_once(
        projection,
        "        core_attn_out = self.norm(core_attn_out, z)\n",
        "        core_attn_out = self.norm(core_attn_out, z)\n"
        "        _rocket_k0_gdn_save(self.prefix, 'normalized', core_attn_out)\n",
    )
    gdn = gdn[:projection_start] + projection + gdn[projection_end:]
    args.gdn_output.write_text(gdn)


if __name__ == "__main__":
    main()
