#!/usr/bin/env python3
"""Layer-0 forward reference for GLM-5.3-Flash on the hub NVFP4 fuel.

Implements the vLLM Glm5Next semantics (verified against
vllm/models/glm5next/nvidia/{model,kda}.py and the FLA fused_recurrent
kernel) in numpy, for one token, and reports the residual-stream RMS after
each sub-block so the engine's layer-0 RMS (0.118 on this fuel) can be
localized to the offending component.

Usage: python3 scripts/numerics/glm53-layer0-reference.py [--dense-off]
"""
from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
from tokenizers import Tokenizer

D = os.path.expanduser("~/.cache/rocket-fuels/glm-5.3-flash-nvfp4")
P = "model.language_model."

RMS_EPS = 1e-5
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
LOWER_BOUND = -5.0
HEADS = 64
HD = 128
H = 4096

_hdrs = {}
_idx = None


def _shard(sh):
    if sh not in _hdrs:
        with open(os.path.join(D, sh), "rb") as fh:
            (hl,) = struct.unpack("<Q", fh.read(8))
            _hdrs[sh] = (hl, json.loads(fh.read(hl)))
    return _hdrs[sh]


def T(name):
    global _idx
    if _idx is None:
        with open(os.path.join(D, "model.safetensors.index.json")) as fh:
            _idx = json.load(fh)["weight_map"]
    sh = _idx[name]
    hl, hdr = _shard(sh)
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(os.path.join(D, sh), "rb") as fh:
        fh.seek(8 + hl + s)
        raw = fh.read(e - s)
    dt, shape = info["dtype"], info["shape"]
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        f = np.frombuffer((u << 16).tobytes(), dtype=np.float32)
    elif dt == "F32":
        f = np.frombuffer(raw, dtype=np.float32)
    elif dt in ("U8", "F8_E4M3"):
        f = np.frombuffer(raw, dtype=np.uint8)
    else:
        raise ValueError(dt)
    return f.reshape(shape) if shape else f


def bf16(x):
    u = np.frombuffer(np.asarray(x, dtype=np.float32).tobytes(), dtype=np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return np.frombuffer(u.tobytes(), dtype=np.float32).copy()


def rmsnorm(x, w, eps=RMS_EPS):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def hc_site(streams, fn, scale, base, norm_w, label):
    """streams [4,4096] f32 -> (post, comb, normed, residual)."""
    x = streams.astype(np.float32)
    flat = x.reshape(-1)
    inv = 1.0 / np.sqrt(flat.dot(flat) / flat.size + RMS_EPS)
    mixes = fn @ (flat * inv)
    pre = sigmoid(mixes[:4] * scale[0] + base[:4]) + HC_EPS
    post = sigmoid(mixes[4:8] * scale[1] + base[4:8]) * POST_MULT
    cl = mixes[8:24].reshape(4, 4) * scale[2] + base[8:24].reshape(4, 4)
    cl = cl - cl.max(axis=-1, keepdims=True)
    e = np.exp(cl)
    comb = e / e.sum(axis=-1, keepdims=True) + HC_EPS
    comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    layer_input = (pre[:, None] * x).sum(0)
    normed = rmsnorm(bf16(layer_input), norm_w)
    print(f"  {label}: pre={pre} post={post}")
    print(f"  {label}: comb row sums={comb.sum(-1)} col sums={comb.sum(-2)}")
    return post, comb, normed.astype(np.float32), x


def hc_combine(post, comb, y, residual):
    out = np.empty_like(residual)
    for i in range(4):
        acc = post[i] * y
        for j in range(4):
            acc += comb[j, i] * residual[j]
        out[i] = bf16(acc)
    return out


def kda_step(layer, normed, state=None):
    qkv_w = np.concatenate(
        [T(f"{P}layers.{layer}.self_attn.{n}_proj.weight") for n in "qkv"], axis=0
    ).astype(np.float32)
    qkv = qkv_w @ normed  # [24576]
    # conv (first token: state zero, last tap hits the current sample)
    out = np.empty(3 * 8192, dtype=np.float32)
    for pi, part in enumerate(("q", "k", "v")):
        w = T(f"{P}layers.{layer}.self_attn.{part}_conv1d.weight").astype(np.float32)
        w = w.reshape(8192, 4)
        seg = qkv[pi * 8192 : (pi + 1) * 8192]
        out[pi * 8192 : (pi + 1) * 8192] = silu(w[:, 3] * seg)
    q, k, v = out[:8192], out[8192:16384], out[16384:]

    # per-head l2norm + q scale
    q = q.reshape(HEADS, HD)
    k = k.reshape(HEADS, HD)
    v = v.reshape(HEADS, HD)
    q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * (HD ** -0.5)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)

    f_a = T(f"{P}layers.{layer}.self_attn.f_a_proj.weight").astype(np.float32)
    f_b = T(f"{P}layers.{layer}.self_attn.f_b_proj.weight").astype(np.float32)
    b_w = T(f"{P}layers.{layer}.self_attn.b_proj.weight").astype(np.float32)
    a_log = T(f"{P}layers.{layer}.self_attn.A_log")
    dt_bias = T(f"{P}layers.{layer}.self_attn.dt_bias")
    fb = f_b @ (f_a @ normed)  # [8192]
    g = LOWER_BOUND * sigmoid(np.exp(a_log)[:, None] * fb.reshape(HEADS, HD) + dt_bias.reshape(HEADS, HD))
    beta = sigmoid(b_w @ normed)  # [64]

    o = np.empty((HEADS, HD), dtype=np.float32)
    for h in range(HEADS):
        S = np.zeros((HD, HD), dtype=np.float32) if state is None else state[h]
        S *= np.exp(g[h])[:, None]
        u = S.T @ k[h]
        d = beta[h] * (v[h] - u)
        S += np.outer(k[h], d)
        o[h] = S.T @ q[h]
        if state is not None:
            state[h] = S

    g_a = T(f"{P}layers.{layer}.self_attn.g_a_proj.weight").astype(np.float32)
    g_b = T(f"{P}layers.{layer}.self_attn.g_b_proj.weight").astype(np.float32)
    o_norm_w = T(f"{P}layers.{layer}.self_attn.o_norm.weight").astype(np.float32)
    g2 = g_b @ (g_a @ normed)  # [8192]
    on = np.empty((HEADS, HD), dtype=np.float32)
    for h in range(HEADS):
        x = o[h]
        inv = 1.0 / np.sqrt(np.mean(x * x) + RMS_EPS)
        on[h] = x * inv * o_norm_w * sigmoid(g2.reshape(HEADS, HD)[h])
    o_proj = T(f"{P}layers.{layer}.self_attn.o_proj.weight").astype(np.float32)
    return o_proj @ on.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-off", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(D, "tokenizer.json"))
    ids = tok.encode("The capital of France is").ids
    emb = T(f"{P}embed_tokens.weight")
    x0 = emb[ids[0]].astype(np.float32)
    print(f"prompt ids: {ids}")
    print(f"embed rms: {np.sqrt((x0**2).mean()):.6f}")

    streams = np.tile(x0, (4, 1))  # hc_expand replication
    state = np.zeros((HEADS, HD, HD), dtype=np.float32)

    L = 0
    fn = T(f"{P}layers.{L}.hc_attn_fn").astype(np.float32)
    base = T(f"{P}layers.{L}.hc_attn_base").astype(np.float32)
    scale = T(f"{P}layers.{L}.hc_attn_scale").astype(np.float32)
    in_norm = T(f"{P}layers.{L}.input_layernorm.weight").astype(np.float32)

    print("== attn site ==")
    post, comb, normed, residual = hc_site(streams, fn, scale, base, in_norm, "attn-pre")
    y = kda_step(L, normed, state)
    print(f"  kda out rms: {np.sqrt((y**2).mean()):.6f} amax: {np.abs(y).max():.6f}")
    streams = hc_combine(post, comb, y, residual)
    r = np.sqrt((streams.mean(0) ** 2).mean())
    print(f"  RMS after attn-site combine: {r:.6f}")

    print("== ffn site ==")
    ffn = T(f"{P}layers.{L}.hc_ffn_fn").astype(np.float32)
    fbase = T(f"{P}layers.{L}.hc_ffn_base").astype(np.float32)
    fscale = T(f"{P}layers.{L}.hc_ffn_scale").astype(np.float32)
    pa_norm = T(f"{P}layers.{L}.post_attention_layernorm.weight").astype(np.float32)
    post2, comb2, normed2, residual2 = hc_site(streams, ffn, fscale, fbase, pa_norm, "ffn-pre")
    if args.dense_off:
        y2 = np.zeros(H, dtype=np.float32)
    else:
        # dense MLP, fp4 dequant on the fly
        E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, -0, -.5, -1, -1.5, -2, -3, -4, -0, -0],
                        dtype=np.float32)

        def e4m3(b):
            s = -1.0 if (b >> 7) & 1 else 1.0
            e = (b >> 4) & 15
            m = b & 7
            if e == 15 and m == 7:
                return s * np.nan
            if e == 0:
                return s * (m / 8) * 2.0 ** -6
            return s * (2.0 ** (e - 7)) * (1 + m / 8)

        def dequant(base):
            w = T(base + ".weight")
            n, k2 = w.shape
            hi = ((w & 0xF0) >> 4).reshape(n, k2)
            lo = (w & 0xF).reshape(n, k2)
            q = np.empty((n, k2 * 2), dtype=np.float32)
            q[:, 0::2] = E2M1[hi]
            q[:, 1::2] = E2M1[lo]
            sc = T(base + ".weight_scale").reshape(n, k2 // 8)
            S = np.vectorize(e4m3, otypes=[np.float32])(sc)
            S = np.repeat(S, 16, axis=1)
            g2 = float(T(base + ".weight_scale_2"))
            return (q * S * g2)

        gate_w = dequant(f"{P}layers.{L}.mlp.gate_proj")
        up_w = dequant(f"{P}layers.{L}.mlp.up_proj")
        down_w = dequant(f"{P}layers.{L}.mlp.down_proj")
        gate = gate_w @ normed2
        up = up_w @ normed2
        lim = 10.0
        act = silu(np.minimum(gate, lim)) * np.clip(up, -lim, lim)
        y2 = down_w @ act
    streams = hc_combine(post2, comb2, y2, residual2)
    hmean = streams.mean(0)
    r = np.sqrt((hmean**2).mean())
    print(f"  RMS after layer-0 ffn-site combine: {r:.6f}")


if __name__ == "__main__":
    main()
