#!/usr/bin/env python3
"""Bytes one decode token reads, straight from the NVFP4 checkpoint header.

fuels/glm-5.3-flash/fuel.yaml records single_token_active_gib as 9.6, derived
as "18B active parameters at ~4.25 bits". That derivation assumes the whole
model is 4-bit. This checkpoint is not: quantization_config.ignore lists
lm_head, the embeddings and every attention projection, and only
mlp.experts.* carries a weight_scale. So the real per-token read is the BF16
half at 2 bytes per parameter plus the routed-expert half at ~4.25 bits.

Usage:
  python3 scripts/runtime/active-set-bytes.py [snapshot_dir]
"""
import json
import os
import re
import sys
import struct

DEFAULT = os.path.expanduser(
    "~/.cache/nim-glm53-2.1.2/ngc/hub/models--nim--zai-org--glm-5.3-flash"
    "/snapshots/nim-aa28e1f-nvfp4")

DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1, "F16": 2, "I32": 4}


def main() -> int:
    snap = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    index = os.path.join(snap, "model.safetensors.index.json")
    if not os.path.exists(index):
        print(f"no checkpoint at {snap}", file=sys.stderr)
        return 77
    cfg = json.load(open(os.path.join(snap, "config.json")))["text_config"]
    top_k = cfg["num_experts_per_tok"]
    n_exp = cfg["n_routed_experts"]

    weight_map = json.load(open(index))["weight_map"]
    headers, sizes = {}, {}
    for name, shard in weight_map.items():
        if shard not in headers:
            with open(os.path.join(snap, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                headers[shard] = json.loads(fh.read(n))
        h = headers[shard][name]
        sizes[name] = h["data_offsets"][1] - h["data_offsets"][0]

    buckets = {"embed+lm_head": 0, "kda attention": 0, "mla attention": 0,
               "dense mlp": 0, "shared experts": 0, "router+hc+norms": 0,
               "routed experts (resident set)": 0}
    layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")
    text_layers = cfg["num_hidden_layers"]

    for name, nbytes in sizes.items():
        if name.startswith("model.visual."):
            continue                       # vision tower, not on the text path
        m = layer_re.match(name)
        if m is None:
            buckets["embed+lm_head"] += nbytes
            continue
        layer, rest = int(m.group(1)), m.group(2)
        if layer >= text_layers:
            continue                       # layer 45, the MTP draft layer
        if rest.startswith("mlp.experts."):
            buckets["routed experts (resident set)"] += nbytes
        elif rest.startswith("mlp.shared_experts."):
            buckets["shared experts"] += nbytes
        elif rest.startswith("mlp.gate."):
            buckets["router+hc+norms"] += nbytes
        elif rest.startswith("mlp."):
            buckets["dense mlp"] += nbytes
        elif rest.startswith("self_attn."):
            key = "mla attention" if cfg["layer_types"][layer] != "linear_attention" else "kda attention"
            buckets[key] += nbytes
        else:
            buckets["router+hc+norms"] += nbytes

    gib = 1 << 30
    experts_all = buckets.pop("routed experts (resident set)")
    experts_active = experts_all * top_k / n_exp
    resident = sum(buckets.values())

    print(f"snapshot {snap}")
    print(f"{'bucket':<34}{'GiB':>10}")
    for k, v in buckets.items():
        print(f"{k:<34}{v / gib:>10.2f}")
    print(f"{'resident (BF16, every token)':<34}{resident / gib:>10.2f}")
    print(f"{'routed experts, whole set':<34}{experts_all / gib:>10.2f}")
    print(f"{'routed experts, top-' + str(top_k) + ' per token':<34}{experts_active / gib:>10.2f}")
    print(f"{'':-<44}")
    total = resident + experts_active
    print(f"{'per-token weight traffic':<34}{total / gib:>10.2f}")
    print(f"{'fuel.yaml single_token_active_gib':<34}{9.6:>10.2f}")
    print(f"{'ratio':<34}{total / gib / 9.6:>10.2f}")
    print()
    print(f"whole checkpoint (text only)      {(resident + experts_all) / gib:>10.2f} GiB")
    for bw, label in ((238e9, "one booster, 238 GB/s"), (2 * 238e9, "two boosters")):
        print(f"roofline tok/s, {label:<22}{bw / total:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
