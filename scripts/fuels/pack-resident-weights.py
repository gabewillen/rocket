#!/usr/bin/env python3
"""Pack the engine's resident weight set into one contiguous load-order file.

The engine mmaps 33 safetensors shards and uploads ~72 GiB per launch; cold
reads at 4.6 GB/s cost ~16 s plus per-shard header churn. This script walks
the loader's upload order (embed, norms, per-layer attention/MLP weights,
lm_head, expert metadata) and writes one packed file with a small JSON
manifest of (name, offset, bytes). The loader's packed-file fast path
(ROCKET_PACKED_WEIGHTS) then mmaps one file and serves every upload from it.

Usage:
  python3 scripts/fuels/pack-resident-weights.py --fuel DIR --out FILE
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

REPO = "/home/glwillen/Development/rocket"

# Loader upload order, mirroring WeightStore::WeightStore in
# engines/glm5-moe-nvfp4-2b/src/weights.cu. Only tensors the loader reads
# as bf16/f32 resident (everything except routed-expert fp4 slots, which
# stream from the checkpoint on demand) belong here.
def resident_names(cfg_path: str) -> list[str]:
    with open(cfg_path) as fh:
        cfg = json.load(fh)["text_config"]
    layers = cfg["num_hidden_layers"]
    kda = set(cfg["linear_attn_config"]["kda_layers"])
    mla = set(cfg["linear_attn_config"]["full_attn_layers"])
    dense_mlp = {i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "dense"}
    P = "model.language_model."
    names = [P + "embed_tokens.weight", P + "norm.weight", "lm_head.weight"]
    for l in range(layers):
        p = f"{P}layers.{l}."
        names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                  p + "hc_attn_fn", p + "hc_attn_base", p + "hc_attn_scale",
                  p + "hc_ffn_fn", p + "hc_ffn_base", p + "hc_ffn_scale"]
        if l in kda:
            a = p + "self_attn."
            names += [a + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                      "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                                      "f_a_proj.weight", "f_b_proj.weight",
                                      "g_a_proj.weight", "g_b_proj.weight",
                                      "b_proj.weight", "A_log", "dt_bias",
                                      "o_norm.weight", "o_proj.weight")]
        elif l in mla:
            a = p + "self_attn."
            names += [a + n for n in ("q_a_proj.weight", "q_a_layernorm.weight",
                                      "q_b_proj.weight", "kv_a_proj_with_mqa.weight",
                                      "kv_a_layernorm.weight", "kv_b_proj.weight",
                                      "o_proj.weight", "indexer.wk.weight",
                                      "indexer.wq_b.weight", "indexer.k_norm.weight",
                                      "indexer.k_norm.bias", "indexer.weights_proj.weight",
                                      "indexer.index_kpool_compress_gate",
                                      "indexer.index_kpool_compress_ape")]
        if l in dense_mlp:
            names += [p + f"mlp.{x}_proj.{y}" for x in ("gate", "up", "down")
                      for y in ("weight", "weight_scale", "weight_scale_2", "input_scale")]
        else:
            names += [p + "mlp.gate.weight", p + "mlp.gate.e_score_correction_bias",
                      p + "mlp.shared_experts.gate_proj.weight",
                      p + "mlp.shared_experts.up_proj.weight",
                      p + "mlp.shared_experts.down_proj.weight"]
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuel", default=os.environ.get("ROCKET_FUEL_NVFP4_DIR")
                    or os.path.expanduser("~/.cache/rocket-fuels/glm-5.3-flash-nvfp4"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.fuel, "resident-packed.bin")

    with open(os.path.join(args.fuel, "model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]
    names = resident_names(os.path.join(args.fuel, "config.json"))

    # header map: name -> (shard, offset, bytes, dtype)
    shards: dict[str, tuple] = {}
    manifests: dict[str, dict] = {}

    def header(shard):
        if shard not in manifests:
            with open(os.path.join(args.fuel, shard), "rb") as fh:
                (hl,) = struct.unpack("<Q", fh.read(8))
                manifests[shard] = (hl, json.loads(fh.read(hl)))
        return manifests[shard][1]

    def header_len(shard):
        header(shard)
        return manifests[shard][0]

    plan = []
    total = 0
    missing = []
    for n in names:
        if n not in index:
            missing.append(n)
            continue
        shard = index[n]
        info = header(shard)[n]
        s, e = info["data_offsets"]
        plan.append((n, shard, s, e - s))
        total += e - s
    if missing:
        print(f"missing {len(missing)} tensors, e.g. {missing[:3]}", file=sys.stderr)
        return 2

    # Emit one safetensors file: JSON header with data_offsets, then the blob
    # in shard order (each shard's region read sequentially).
    plan.sort(key=lambda t: (t[1], t[2]))
    header_entries = {}
    blob_off = 0
    for n, shard, s, nb in plan:
        info = header(shard)[n]
        header_entries[n] = {"dtype": info["dtype"], "shape": info["shape"],
                             "data_offsets": [blob_off, blob_off + nb]}
        blob_off += nb
    header_json = json.dumps(header_entries, separators=(",", ":")).encode()
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad
    written = 0
    handles = {}
    with open(out, "wb") as w:
        w.write(struct.pack("<Q", len(header_json)))
        w.write(header_json)
        written = 8 + len(header_json)
        for n, shard, s, nb in plan:
            if shard not in handles:
                handles[shard] = open(os.path.join(args.fuel, shard), "rb")
            # Shard data_offsets are payload-relative (blob start = 8 + header_len).
            handles[shard].seek(8 + header_len(shard) + s)
            remaining = nb
            while remaining:
                chunk = handles[shard].read(min(remaining, 8 << 20))
                if not chunk:
                    print(f"short read on {n}", file=sys.stderr)
                    return 1
                w.write(chunk)
                remaining -= len(chunk)
            written += nb
    for h in handles.values():
        h.close()
    print(f"packed {written/2**30:.2f} GiB, {len(plan)} tensors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
