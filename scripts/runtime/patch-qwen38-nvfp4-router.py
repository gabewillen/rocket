#!/usr/bin/env python3
"""Make the pinned Qwen3.8 MoE router honor its NVFP4 policy."""

import argparse
import ast
from pathlib import Path

MARKER = "# ROCKET_QWEN38_NVFP4_ROUTER_V1\n"
IMPORT_ANCHOR = "from vllm.model_executor.layers.logits_processor import LogitsProcessor\n"
IMPORT_PATCH = (
    "from vllm.model_executor.layers.linear import ReplicatedLinear\n"
    + IMPORT_ANCHOR
)
INIT_ANCHOR = "        super().__init__(vllm_config=vllm_config, prefix=prefix)\n"
INIT_PATCH = INIT_ANCHOR + MARKER + '''        quant_config = vllm_config.quant_config
        router_prefix = f"{prefix}.gate"
        router_algo = (
            quant_config._resolve_quant_algo(router_prefix)
            if quant_config is not None
            and hasattr(quant_config, "_resolve_quant_algo")
            else None
        )
        if router_algo == "NVFP4":
            if not hasattr(self.experts, "gate") or getattr(
                self.experts, "_fse_fuse_gate", False
            ):
                raise ValueError(
                    f"Qwen3.8 router {router_prefix} cannot replace the MoE gate"
                )
            config = vllm_config.model_config.hf_text_config
            self.gate = ReplicatedLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                quant_config=quant_config,
                prefix=router_prefix,
            )
            self.experts.gate = self.gate
        elif router_algo is not None:
            raise ValueError(
                f"Qwen3.8 router {router_prefix} has unsupported policy {router_algo}"
            )
'''


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise SystemExit(f"vLLM source drift: expected one {label} anchor")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_py", type=Path)
    args = parser.parse_args()
    source = args.model_py.read_text()
    if MARKER.strip() in source:
        raise SystemExit("already patched")
    source = replace_once(source, IMPORT_ANCHOR, IMPORT_PATCH, "linear import")
    source = replace_once(source, INIT_ANCHOR, INIT_PATCH, "router construction")
    ast.parse(source, filename=str(args.model_py))
    args.model_py.write_text(source)


if __name__ == "__main__":
    main()
