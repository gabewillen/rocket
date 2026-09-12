#!/usr/bin/env python3
"""Per-family weight audit for an NVFP4 checkpoint.

Reads safetensors headers only (zero tensor I/O), buckets tensor names into
weight families, and reports per-family bytes, dtypes, and per-tensor
aggregates. Optionally splits per decode-relevant projection type inside the
attention family.

Usage:
  python3 scripts/fuels/nvfp4-family-audit.py [--dir DIR]

Defaults to $ROCKET_FUEL_NVFP4_DIR when --dir is absent, else
~/.cache/rocket-fuels/glm-5.3-flash-nvfp4.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from collections import defaultdict

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "BF16": 2, "F16": 2,
    "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def family(name: str) -> str:
    n = name.lower()
    if ".experts." in n or "moe.experts" in n:
        return "moe_expert(fp4)"
    if "shared_expert" in n:
        return "shared_expert"
    if "self_attn" in n:
        return "attention"
    if "norm" in n:
        return "norms"
    if n.endswith(".mlp.gate") or "gate" in n or "router" in n or "b_proj" in n:
        return "router/gates"
    if "embed" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    return "other"


def attn_projection(name: str) -> str:
    n = name.lower()
    for tag, pat in (
        ("kda in_proj", "in_proj"),
        ("kda f_b/g_b", "f_b_proj"),
        ("kda g_b", "g_b_proj"),
        ("kda conv", "conv"),
        ("kda o_proj", "o_proj"),
        ("kda o_norm", "o_rms"),
        ("mla q_a", "q_a_proj"),
        ("mla q_b", "q_b_proj"),
        ("mla kv_a", "kv_a_proj"),
        ("mla kv_b", "kv_b_proj"),
        ("mla o_proj", "o_proj"),
        ("indexer", "index"),
    ):
        if pat in n:
            return tag
    return "other-attn"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = args.dir or os.environ.get("ROCKET_FUEL_NVFP4_DIR")
    if not d:
        d = os.path.expanduser("~/.cache/rocket-fuels/glm-5.3-flash-nvfp4")
    if not os.path.isdir(d):
        print(f"no fuel dir: {d}", file=sys.stderr)
        return 2

    meta: dict[str, tuple] = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(f, "rb") as fh:
            (hl,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(hl))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            dt, shape = info["dtype"], info["shape"]
            nb = 1
            for s in shape:
                nb *= s
            nb *= DTYPE_BYTES.get(dt, 1)
            meta[name] = (dt, shape, nb)

    fams: dict[str, dict] = defaultdict(
        lambda: {"bytes": 0, "n": 0, "dtypes": set(), "tensors": []}
    )
    attn_proj: dict[str, dict] = defaultdict(lambda: {"bytes": 0, "n": 0})
    for name, (dt, shape, nb) in meta.items():
        f = family(name)
        fams[f]["bytes"] += nb
        fams[f]["n"] += 1
        fams[f]["dtypes"].add(dt)
        fams[f]["tensors"].append(name)
        if f == "attention":
            p = attn_projection(name)
            attn_proj[p]["bytes"] += nb
            attn_proj[p]["n"] += 1

    total = sum(v["bytes"] for v in fams.values())
    print("== bytes per family ==")
    for f in sorted(fams, key=lambda x: -fams[x]["bytes"]):
        v = fams[f]
        print(
            f"  {f:24s} {v['bytes']/1e9:9.2f} GB  tensors={v['n']:6d}  "
            f"{sorted(v['dtypes'])}  share={100*v['bytes']/total:5.1f}%"
        )
    print(f"  {'TOTAL':24s} {total/1e9:9.2f} GB  tensors={len(meta)}")

    print("\n== attention family by projection ==")
    for p in sorted(attn_proj, key=lambda x: -attn_proj[x]["bytes"]):
        v = attn_proj[p]
        print(f"  {p:14s} {v['bytes']/1e9:8.3f} GB  tensors={v['n']:4d}")

    attn_layers = sorted(
        {int(name.split(".")[4]) for name in fams["attention"]["tensors"]
         if name.split(".")[4].isdigit()} or {0}
    )
    print(f"\nattention layers present: {len(attn_layers)} (first {attn_layers[:3]})")

    # Per-token read set: weights whose compute consumes the full matrix per
    # decode token (m=1 projections over every fired layer). Embed is a row
    # gather and is excluded; lm_head reads the vocab once per step.
    TOKEN_READ_WEIGHTS = {
        "attention": 1.0,
        "shared_expert": 1.0,
        "lm_head": 1.0,
        "router/gates": 1.0,
        "norms": 1.0,
        "other": 1.0,
        "embed": 0.0,     # row gather, ~8 kB/token
        "moe_expert(fp4)": 0.0,
    }
    per_token = sum(fams[f]["bytes"] * w for f, w in TOKEN_READ_WEIGHTS.items())
    fp4_bpe = 0.5 + 1.0 / 16  # 4-bit payload + fp8 e4m3 scale per 16-element block
    bf16_bpe = 2.0
    ratio = fp4_bpe / bf16_bpe
    print("\n== decode per-token weight read (c1, m=1) ==")
    print(f"  bf16 read      = {per_token/1e9:5.2f} GB/token")
    print(f"  after fp4 conv = {per_token*ratio/1e9:5.2f} GB/token (x{ratio:.2f}, 72% cut)")
    print(f"  c1 roofline    = {1000.0*(per_token/1e9)/236:6.1f} ms/token now -> "
          f"{1000.0*(per_token*ratio/1e9+5.1)/236:6.1f} ms/token fp4 "
          f"(@236 GB/s, +5.1 GB/token fp4 experts)")

    out = os.path.join(d, "nvfp4-family-audit.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "fuel_dir": d,
                "total_bytes": total,
                "families": {
                    f: {"bytes": fams[f]["bytes"], "tensors": fams[f]["n"],
                        "dtypes": sorted(fams[f]["dtypes"])}
                    for f in fams
                },
                "attention_projections": dict(attn_proj),
                "per_token_read_gb": per_token/1e9,
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
