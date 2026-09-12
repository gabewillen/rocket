#!/usr/bin/env python3
"""Apply the upstream dense-FP8 overlay to the EXL3 stack's vllm source.

Swaps vllm/model_executor/layers/quantization/exl3.py for the MiaAI-Lab
overlay version (GLM-5.3-Flash-EXL3-2x-DGX-Sparks, which adds the FP8
weight-only Marlin path for the KDA/MLA/dense projections) and patches the
glm5next KDA/MLA constructors to keep the exl3 quant config so the dense
projections reach it. Idempotent; writes .pre-fp8 backups next to each file.

Usage: python3 scripts/runtime/apply-exl3-dense-fp8.py [--source DIR] [--revert]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OVERLAY = REPO / "scripts/runtime/exl3-dense-fp8-overlay.py"
MARK = "# [glm53-dense-fp8]"

KDA_OLD = """        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config
"""
KDA_NEW = """        saved_quant_config = vllm_config.quant_config
        if getattr(saved_quant_config, "get_name", lambda: "")() != "exl3":  # [glm53-dense-fp8]
            vllm_config.quant_config = None
        try:
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config
"""
MLA_OLD = """                quant_config=None,  # MLA projections are BF16 in checkpoint
                prefix=f"{prefix}.self_attn",
"""
MLA_NEW = """                quant_config=(quant_config if getattr(quant_config, "get_name", lambda: "")() == "exl3" else None),  # [glm53-dense-fp8]
                prefix=f"{prefix}.self_attn",
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/home/glwillen/vllm-glm53/vllm")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    site = Path(args.source)
    exl3 = site / "model_executor/layers/quantization/exl3.py"
    kda = site / "models/glm5next/nvidia/kda.py"
    model = site / "models/glm5next/nvidia/model.py"

    if args.revert:
        for f in (exl3, kda, model):
            b = f.with_suffix(f.suffix + ".pre-fp8")
            if b.exists():
                shutil.copy(b, f)
                print(f"reverted {f.name}")
        return 0

    for f in (exl3, kda, model):
        b = f.with_suffix(f.suffix + ".pre-fp8")
        if not b.exists():
            shutil.copy(f, b)
    shutil.copy(OVERLAY, exl3)
    print(f"installed overlay exl3.py ({OVERLAY.name})")
    for path, old, new, label in ((kda, KDA_OLD, KDA_NEW, "kda"),
                                  (model, MLA_OLD, MLA_NEW, "mla")):
        text = path.read_text()
        if MARK in text:
            print(f"{path.name}: already patched")
            continue
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{path}: expected one {label} target, found {n}")
        path.write_text(text.replace(old, new, 1))
        print(f"patched {path.name} ({label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
