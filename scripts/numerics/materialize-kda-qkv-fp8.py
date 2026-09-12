#!/usr/bin/env python3
"""Materialize FP8-per-row overlay objects for every KDA layer's q/k/v triple.

Reads per-tensor SHA-256 provenance from the inventory JSON that sits next to
the snapshot (scripts/numerics/nvfp4-overlay-inventory.py output), then calls
the materialize-nvfp4-overlay binary in --fp8-row mode per layer.

Usage:
  python3 scripts/numerics/materialize-kda-qkv-fp8.py [--snapshot DIR] [--out ROOT]
      [--inventory FILE] [--ids CSV]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE = os.path.join(REPO, "engines", "glm5-moe-nvfp4-2b")
BUILD = os.path.join(ENGINE, "build")

KDA_IDS = "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("ROCKET_FUEL_NVFP4_DIR")
                    or os.path.expanduser("~/.cache/rocket-fuels/glm-5.3-flash-nvfp4"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/rocket-fuels/overlays/kda-qkv-fp8"))
    ap.add_argument("--inventory", default=None)
    ap.add_argument("--ids", default=KDA_IDS)
    args = ap.parse_args()

    inv = args.inventory or os.path.join(os.path.dirname(args.snapshot.rstrip("/")), "inventory.json")
    if not os.path.exists(inv):
        print(f"inventory missing: {inv}; run nvfp4-overlay-inventory.py first", file=sys.stderr)
        return 2
    with open(inv) as fh:
        data = json.load(fh)
    by_name = {t["source_tensor"]: t for t in data["candidate_tensors"]}
    snapshot_key = data["source"]["snapshot_key"]

    subprocess.run(
        ["cmake", "--build", BUILD, "--target", "rocket_fuel", "-j", str(os.cpu_count())],
        cwd=REPO, check=True, capture_output=True)
    subprocess.run(
        ["c++", "-std=c++20", "-O3", "-I", os.path.join(ENGINE, "src"),
         os.path.join(REPO, "scripts", "numerics", "materialize-nvfp4-overlay.cc"),
         os.path.join(BUILD, "librocket_fuel.a"),
         "-o", os.path.join(BUILD, "materialize-nvfp4-overlay")],
        check=True)

    for layer in (int(x) for x in args.ids.split(",")):
        names = [f"model.language_model.layers.{layer}.self_attn.{p}_proj.weight"
                 for p in ("q", "k", "v")]
        missing = [n for n in names if n not in by_name]
        if missing:
            print(f"layer {layer}: inventory lacks {missing}", file=sys.stderr)
            return 2
        spec = ",".join(n + "=" + by_name[n]["source_sha256"] for n in names)
        out = os.path.join(args.out, f"l-{layer}")
        os.makedirs(out, exist_ok=True)
        r = subprocess.run(
            [os.path.join(BUILD, "materialize-nvfp4-overlay"), args.snapshot, out,
             snapshot_key, spec, "--fp8-row"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"layer {layer} failed: {r.stderr}", file=sys.stderr)
            return 1
        rel = r.stdout.strip().split("\t")[-1]
        print(f"l-{layer}: relL2={rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
