#!/usr/bin/env python3
"""Memory-bandwidth roofline for Qwen3.8-Flash-Next checkpoint headers."""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from collections import Counter
from pathlib import Path


def tensor_bytes(meta: dict) -> int:
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("every tensor needs two data_offsets")
    return int(offsets[1]) - int(offsets[0])


def inventory(headers: dict[str, dict]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for name, meta in headers.items():
        size = tensor_bytes(meta)
        if name.startswith("model.visual"):
            family = "vision_excluded"
        elif name.startswith("mtp."):
            family = "mtp_all_experts" if ".mlp.experts." in name else "mtp_dense"
        elif ".ple.ple_embedding.ngram_embedding." in name:
            family = "ple_table"
        elif ".mlp.experts." in name:
            family = "base_all_experts"
        elif name == "model.language_model.embed_tokens.weight":
            family = "embedding_table"
        elif name == "lm_head.weight":
            family = "lm_head"
        elif name.startswith("model.language_model."):
            family = "base_dense_other"
        else:
            family = "other"
        out[family] += size
    required = {
        "base_all_experts", "base_dense_other", "embedding_table", "lm_head",
        "mtp_all_experts", "mtp_dense", "ple_table",
    }
    missing = sorted(required - out.keys())
    if missing:
        raise ValueError("checkpoint family drift, missing: " + ", ".join(missing))
    return dict(out)


def expected_union(experts: int, top_k: int, batch: int) -> float:
    if not (0 < top_k <= experts and batch > 0):
        raise ValueError("require 0 < top_k <= experts and batch > 0")
    return experts * (1.0 - (1.0 - top_k / experts) ** batch)


def remote_headers(repo: str, revision: str) -> dict[str, dict]:
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    with urllib.request.urlopen(base + "model.safetensors.index.json") as response:
        index = json.load(response)
    headers: dict[str, dict] = {}
    for filename in sorted(set(index["weight_map"].values())):
        request = urllib.request.Request(
            base + filename, headers={"Range": "bytes=0-7", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request) as response:
            header_size = struct.unpack("<Q", response.read(8))[0]
        request = urllib.request.Request(
            base + filename,
            headers={"Range": f"bytes=8-{7 + header_size}", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request) as response:
            shard = json.loads(response.read(header_size))
        shard.pop("__metadata__", None)
        overlap = headers.keys() & shard.keys()
        if overlap:
            raise ValueError(f"duplicate tensors across shards: {sorted(overlap)[:3]}")
        headers.update(shard)
    return headers


def model(headers: dict[str, dict], batch: int, experts: int, top_k: int,
          nodes: int, gb_s_per_node: float, ple_rows: int,
          hidden_size: int, ple_bytes: int, embedding_bytes: int, route: str,
          mtp_proposals: int, accepted_tokens_per_step: float) -> dict:
    sizes = inventory(headers)
    union = float(top_k) if route == "same" else expected_union(experts, top_k, batch)
    shared = sizes["base_dense_other"] + sizes["lm_head"]
    routed = sizes["base_all_experts"] * union / experts
    row_gathers = batch * hidden_size * (embedding_bytes + ple_rows * ple_bytes)
    mtp_per_pass = sizes["mtp_dense"] + sizes["mtp_all_experts"] * union / experts
    mtp_traffic = mtp_proposals * mtp_per_pass
    step_bytes = shared + routed + row_gathers + mtp_traffic
    bandwidth = nodes * gb_s_per_node * 1e9
    steps_s = bandwidth / step_bytes
    accepted = batch * accepted_tokens_per_step
    return {
        "assumptions": {
            "batch": batch, "experts": experts, "top_k": top_k,
            "expected_unique_experts": union, "route_model": route,
            "nodes": nodes, "gb_s_per_node": gb_s_per_node,
            "ple_rows_per_token": ple_rows, "accepted_tokens_per_step": accepted_tokens_per_step,
            "mtp_proposals": mtp_proposals,
            "scope": "weight-read roofline; excludes compute, state, KV, launch, and fabric stalls",
        },
        "checkpoint_bytes": sizes,
        "step_bytes": {
            "shared_dense_and_lm_head": shared,
            "routed_expert_union": routed,
            "embedding_and_ple_rows": row_gathers,
            "mtp_proposal_weights_per_pass": mtp_per_pass,
            "mtp_proposal_traffic": mtp_traffic,
            "total": step_bytes,
        },
        "ceiling": {
            "batch_steps_per_s": steps_s,
            "aggregate_accepted_tokens_per_s": steps_s * accepted,
            "accepted_tokens_per_s_per_stream": steps_s * accepted_tokens_per_step,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--headers", type=Path)
    source.add_argument("--repo")
    ap.add_argument("--revision")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--gb-s-per-node", type=float, default=238.0)
    ap.add_argument("--ple-rows", type=int, default=16)
    ap.add_argument("--hidden-size", type=int, default=2560)
    ap.add_argument("--ple-bytes", type=int, default=1)
    ap.add_argument("--embedding-bytes", type=int, default=2)
    ap.add_argument("--route", choices=("uniform", "same"), default="uniform")
    ap.add_argument("--mtp-proposals", type=int, default=0)
    ap.add_argument("--accepted-tokens-per-step", type=float, default=1.0)
    ap.add_argument("--tsv", action="store_true")
    args = ap.parse_args()
    if args.repo:
        if not args.revision:
            ap.error("--repo requires --revision")
        headers = remote_headers(args.repo, args.revision)
    else:
        headers = json.loads(args.headers.read_text())
    result = model(headers, args.batch, args.experts,
                   args.top_k, args.nodes, args.gb_s_per_node, args.ple_rows,
                   args.hidden_size, args.ple_bytes, args.embedding_bytes, args.route,
                   args.mtp_proposals,
                   args.accepted_tokens_per_step)
    if args.tsv:
        a, s, c = result["assumptions"], result["step_bytes"], result["ceiling"]
        print("batch\troute\tunique_experts\tstep_GiB\tbatch_steps_s\taggregate_tok_s\tper_stream_tok_s")
        print(f'{a["batch"]}\t{a["route_model"]}\t{a["expected_unique_experts"]:.3f}\t'
              f'{s["total"]/2**30:.6f}\t{c["batch_steps_per_s"]:.3f}\t'
              f'{c["aggregate_accepted_tokens_per_s"]:.3f}\t'
              f'{c["accepted_tokens_per_s_per_stream"]:.3f}')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
