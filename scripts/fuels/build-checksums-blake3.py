#!/usr/bin/env python3
"""Emit a NIM-format checksums.blake3 for a hub-shaped safetensors snapshot.

The fuel download is a Hugging Face hub snapshot (no checksums.blake3); the
engine's ovelay inventory tooling expects the NIM manifest. This computes the
BLAKE3 digest of every file the manifest convention covers (safetensors shards,
config.json, model.safetensors.index.json) plus configs and writes the file
next to them. Digests are of actual local bytes, so the file is a valid
integrity manifest for what is on disk.

Usage: python3 scripts/fuels/build-checksums-blake3.py --dir DIR
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import blake3

NAMES = []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    d = args.dir or os.environ.get("ROCKET_FUEL_NVFP4_DIR")
    if not d:
        d = os.path.expanduser("~/.cache/rocket-fuels/glm-5.3-flash-nvfp4")
    files = (
        sorted(glob.glob(os.path.join(d, "*.safetensors")))
        + sorted(glob.glob(os.path.join(d, "*.json")))
        + sorted(glob.glob(os.path.join(d, "*.jinja")))
    )
    if not files:
        print(f"no files under {d}", file=sys.stderr)
        return 2
    out = os.path.join(d, "checksums.blake3")
    lines = []
    for f in files:
        h = blake3.blake3()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(4 << 20), b""):
                h.update(chunk)
        rel = os.path.relpath(f, d)
        lines.append(f"{h.hexdigest()}  {rel}")
        print(f"  {h.hexdigest()[:12]}...  {rel}")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
