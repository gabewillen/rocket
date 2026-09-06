#!/usr/bin/env python3
"""Print attention tensor shapes from the GLM-5.3-Flash NVFP4 checkpoint.

Reads only the safetensors JSON headers, so it never touches tensor bytes and
runs in under a second on a cold cache.

    scripts/attention/dump-attention-shapes.py [SNAPSHOT_DIR] [LAYER ...]

SNAPSHOT_DIR defaults to the NIM cache path used by this project. LAYER
defaults to 0 (KDA) and 3 (sparse MLA).
"""
import json
import os
import struct
import sys
from collections import defaultdict

DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/nim-glm53-2.1.2/ngc/hub/models--nim--zai-org--glm-5.3-flash/"
    "snapshots/nim-aa28e1f-nvfp4"
)


def read_header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def main(argv):
    snapshot = argv[1] if len(argv) > 1 and argv[1] else DEFAULT_SNAPSHOT
    layers = [int(a) for a in argv[2:]] or [0, 3]

    index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    wanted = []
    for name in weight_map:
        for layer in layers:
            if f".layers.{layer}." in name and ".self_attn." in name:
                wanted.append((layer, name))

    by_shard = defaultdict(list)
    for layer, name in wanted:
        by_shard[weight_map[name]].append((layer, name))

    headers = {shard: read_header(os.path.join(snapshot, shard)) for shard in by_shard}

    rows = []
    for shard, entries in by_shard.items():
        for layer, name in entries:
            meta = headers[shard][name]
            rows.append((layer, name, meta["dtype"], meta["shape"]))

    rows.sort(key=lambda r: (r[0], r[1]))
    width = max(len(r[1]) for r in rows) if rows else 0
    for layer, name, dtype, shape in rows:
        print(f"{name:<{width}}  {dtype:<5}  {shape}")

    if not rows:
        print("no self_attn tensors found for layers " + repr(layers), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
