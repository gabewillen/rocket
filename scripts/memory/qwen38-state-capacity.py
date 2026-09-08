#!/usr/bin/env python3
"""Byte-count Qwen3.8 text serving state from its pinned config and vLLM source.

This is a capacity planner, not a CUDA allocation claim.  It mirrors the
state shapes and cache specs in NVIDIA's pinned vLLM image and emits the exact
payload that a later, model-resident allocation probe must allocate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
CONFIG_SHA256 = "deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80"
IMAGE = "vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
IMAGE_ID = "sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a"
SOURCE_CONTRACT = {
    "kv_cache_interface.py": {
        "path": "vllm/v1/kv_cache_interface.py",
        "sha256": "7452e823367960daf73520f80824aee0d5ed8ad2731d0404705695814d3bcff7",
        "anchors": ("class FullAttentionSpec", "class MLAAttentionSpec", "class CircularBufferSpec", "class MambaSpec"),
    },
    "model.py": {
        "path": "vllm/models/qwen3_8_flash_next/nvidia/model.py",
        "sha256": "d900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17",
        "anchors": ("get_gdn_mamba_state_shape_from_config", "get_ple_mamba_state_shape_from_config", "tp_replicated=True"),
    },
    "mtp.py": {
        "path": "vllm/models/qwen3_8_flash_next/nvidia/mtp.py",
        "sha256": "7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431",
        "anchors": ("layer_type=\"full_attention\"", "self.num_mtp_layers", "set_skip_topk"),
    },
    "qsa_cache.py": {
        "path": "vllm/models/qwen3_8_flash_next/common/qsa_cache.py",
        "sha256": "e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8",
        "anchors": ("class QSAKeyStateCache", "class QSACompressedKeyCache", "span = self.compress_ratio + vllm_config.num_speculative_tokens"),
    },
    "qsa.py": {
        "path": "vllm/models/qwen3_8_flash_next/nvidia/qsa.py",
        "sha256": "748addc85efaa8f7df940d1245bc900192f92e1f17af8fa774625758600751cb",
        "anchors": (
            "supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\"auto\", \"bfloat16\"]",
            "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache",
            "QSA does not support KV quantization",
        ),
    },
    "mamba_utils.py": {
        "path": "vllm/model_executor/layers/mamba/mamba_utils.py",
        "sha256": "e168adae4ac9a951f2566aa5fd84a4615aed97d1fbca3787be9acf9cc192c3eb",
        "anchors": ("gated_delta_net_state_shape", "short_conv_state_shape", "conv_kernel_size - 1 + num_spec"),
    },
    "platform_interface.py": {
        "path": "vllm/platforms/interface.py",
        "sha256": "7109cdf97649c1b7a3e471fc98df06be2eacb77501df94016a44f1327b01f55d",
        "anchors": ("attn_tokens_per_mamba_state", "attn_block_size = kernel_block_alignment_size", "mamba_page_size_padded"),
    },
    "kv_cache_utils.py": {
        "path": "vllm/v1/core/kv_cache_utils.py",
        "sha256": "f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4",
        "anchors": ("_get_kv_cache_groups_csa_linear", "page_size_padded=compressed_page", "page_size_padded=main_kv_page"),
    },
    "flash_attn.py": {
        "path": "vllm/v1/attention/backends/flash_attn.py",
        "sha256": "6bcd9c496e25abffef257aebb8a5efd332343af3da302560596aa9b2e2815935",
        "anchors": ("def get_supported_kernel_block_sizes", "return [MultipleOf(16)]"),
    },
}

DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "fp8_e4m3": 1}


class PlanError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ceil_div(n: int, d: int) -> int:
    if n < 0 or d <= 0:
        raise PlanError("ceil_div requires n >= 0 and d > 0")
    return (n + d - 1) // d


def round_up(n: int, alignment: int) -> int:
    return ceil_div(n, alignment) * alignment


def require_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise PlanError(f"missing or invalid positive integer text_config.{name}")
    return value


def require_dtype(value: str, name: str) -> str:
    if value not in DTYPE_BYTES:
        raise PlanError(f"missing or unsupported {name}: {value!r}")
    return value


def read_image_sources(image: str) -> tuple[str, dict[str, bytes]]:
    inspect = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True,
    )
    image_id = inspect.stdout.strip()
    root = "/usr/local/lib/python3.12/dist-packages/"
    script = "\n".join(
        f"printf '%s\\0' '{name}'; cat '{root}{spec['path']}'"
        for name, spec in SOURCE_CONTRACT.items()
    )
    result = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/sh", image, "-c", script],
        check=True, capture_output=True,
    )
    # Each file begins at its unique NUL-delimited contract name. Split using
    # the known ordered names so Python source may contain arbitrary bytes.
    payload = result.stdout
    sources: dict[str, bytes] = {}
    names = list(SOURCE_CONTRACT)
    for index, name in enumerate(names):
        marker = name.encode() + b"\0"
        if not payload.startswith(marker):
            raise PlanError(f"source stream missing marker for {name}")
        payload = payload[len(marker):]
        if index + 1 < len(names):
            next_marker = names[index + 1].encode() + b"\0"
            pos = payload.find(next_marker)
            if pos < 0:
                raise PlanError(f"source stream missing marker for {names[index + 1]}")
            sources[name], payload = payload[:pos], payload[pos:]
        else:
            sources[name] = payload
    return image_id, sources


def validate_sources(image_id: str, sources: dict[str, bytes]) -> dict[str, str]:
    if image_id != IMAGE_ID:
        raise PlanError(f"vLLM image ID drift: expected {IMAGE_ID}, got {image_id}")
    hashes: dict[str, str] = {}
    for name, contract in SOURCE_CONTRACT.items():
        data = sources.get(name)
        if data is None:
            raise PlanError(f"missing pinned implementation source {name}")
        digest = sha256(data)
        if digest != contract["sha256"]:
            raise PlanError(f"pinned implementation hash drift for {name}: {digest}")
        text = data.decode("utf-8")
        missing = [anchor for anchor in contract["anchors"] if anchor not in text]
        if missing:
            raise PlanError(f"pinned implementation anchors missing in {name}: {missing}")
        hashes[name] = digest
    return hashes


@dataclass(frozen=True)
class StateFamily:
    id: str
    owner: str
    family: str
    layers: int
    source_dtype: str
    serving_dtype: str
    logical_shape_per_layer_per_stream: list[int]
    logical_bytes_per_layer_per_stream: int
    logical_bytes_per_stream: int
    cuda_page_bytes_per_layer: int
    cuda_blocks_per_stream: int
    cuda_allocated_bytes_per_layer_per_stream: int
    cuda_allocated_bytes_per_stream: int
    cuda_allocated_bytes_c16: int
    cuda_allocation_accounted: bool
    shares_cuda_allocation_with: str | None
    nvme_padded_bytes_per_stream: int
    tp_ownership: str
    sequence_dependence: str
    restore_atomicity: str


def make_family(
    *, id: str, owner: str, family: str, layers: int, dtype: str,
    shape: list[int], cuda_blocks: int, cuda_page_bytes: int, concurrency: int,
    tp_ownership: str, sequence_dependence: str, host_page: int,
    shares_cuda_allocation_with: str | None = None,
) -> StateFamily:
    per_layer = math.prod(shape) * DTYPE_BYTES[dtype]
    logical = per_layer * layers
    allocated_per_layer = cuda_page_bytes * cuda_blocks
    allocated = allocated_per_layer * layers
    return StateFamily(
        id=id, owner=owner, family=family, layers=layers,
        source_dtype="runtime-produced", serving_dtype=dtype,
        logical_shape_per_layer_per_stream=shape,
        logical_bytes_per_layer_per_stream=per_layer,
        logical_bytes_per_stream=logical,
        cuda_page_bytes_per_layer=cuda_page_bytes,
        cuda_blocks_per_stream=cuda_blocks,
        cuda_allocated_bytes_per_layer_per_stream=allocated_per_layer,
        cuda_allocated_bytes_per_stream=allocated,
        cuda_allocated_bytes_c16=allocated * concurrency,
        cuda_allocation_accounted=shares_cuda_allocation_with is None,
        shares_cuda_allocation_with=shares_cuda_allocation_with,
        nvme_padded_bytes_per_stream=round_up(logical, host_page),
        tp_ownership=tp_ownership,
        sequence_dependence=sequence_dependence,
        restore_atomicity="restore with every required family at one accepted-token boundary",
    )


def build_plan(
    config_bytes: bytes, *, revision: str, context: int, concurrency: int,
    tp: int, block_size: int, kernel_block_alignment: int, kv_dtype: str, mamba_cache_dtype: str,
    mamba_ssm_dtype: str, speculative_tokens: int, mamba_cache_mode: str,
    host_page: int, image_id: str, source_hashes: dict[str, str],
) -> dict[str, Any]:
    if revision != REVISION:
        raise PlanError(f"checkpoint revision must be {REVISION}")
    digest = sha256(config_bytes)
    if digest != CONFIG_SHA256:
        raise PlanError(f"checkpoint config hash drift: expected {CONFIG_SHA256}, got {digest}")
    try:
        outer = json.loads(config_bytes)
        cfg = outer["text_config"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PlanError("config.json lacks an object text_config") from exc
    if not isinstance(cfg, dict):
        raise PlanError("config.json text_config is not an object")
    if context != require_int(cfg, "max_position_embeddings"):
        raise PlanError("context must equal pinned max_position_embeddings")
    if concurrency != 16 or tp != 2:
        raise PlanError("this ledger is pinned to concurrency 16 and TP2")
    if block_size <= 0 or kernel_block_alignment <= 0:
        raise PlanError("requested block size and kernel block alignment must be positive")
    if block_size != 16 or kernel_block_alignment != 16:
        raise PlanError("pinned FP8 FlashAttention contract requires requested/alignment block size 16")
    if host_page != 65536:
        raise PlanError("host page contract must be exactly 65536 bytes")
    if speculative_tokens != 3:
        raise PlanError("MTP parity contract requires exactly 3 speculative tokens")
    if mamba_cache_mode != "align":
        raise PlanError("pinned implementation requires mamba_cache_mode=align")
    require_dtype(kv_dtype, "KV cache dtype")
    require_dtype(mamba_cache_dtype, "Mamba convolution dtype")
    require_dtype(mamba_ssm_dtype, "Mamba recurrent dtype")
    if cfg.get("dtype") != "bfloat16" or outer.get("dtype") != "bfloat16":
        raise PlanError("source/model dtype must be explicitly bfloat16")
    if cfg.get("mamba_ssm_dtype") != "float32" or mamba_ssm_dtype != "float32":
        raise PlanError("recurrent state must remain explicit float32")
    # The pinned QSA kernel rejects quantized K/V and accepts BF16 only.
    # FP8 was an earlier capacity-planning input, not a working serving ABI.
    if kv_dtype != "bfloat16" or mamba_cache_dtype != "bfloat16":
        raise PlanError("working QSA kernel requires BF16 K/V and convolution state")

    layer_types = cfg.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != require_int(cfg, "num_hidden_layers"):
        raise PlanError("layer_types must cover every text layer")
    unknown = set(layer_types) - {"linear_attention", "full_attention"}
    if unknown:
        raise PlanError(f"unsupported text layer types: {sorted(unknown)}")
    linear_layers = layer_types.count("linear_attention")
    full_layers = layer_types.count("full_attention")
    mtp = cfg.get("mtp")
    if not isinstance(mtp, dict) or mtp.get("layer_types") != ["full_attention"]:
        raise PlanError("pinned MTP must contain one full-attention layer")
    mtp_layers = require_int(cfg, "mtp_num_hidden_layers")
    if mtp_layers != 1 or mtp.get("num_hidden_layers") != 1:
        raise PlanError("pinned MTP layer count drift")

    total_kv_heads = require_int(cfg, "num_key_value_heads")
    if total_kv_heads >= tp:
        if total_kv_heads % tp:
            raise PlanError("QSA KV heads are not divisible by TP")
        local_kv_heads = total_kv_heads // tp
        kv_ownership = "TP-sharded KV heads"
    else:
        if tp % total_kv_heads:
            raise PlanError("TP is not divisible by replicated QSA KV heads")
        local_kv_heads = 1
        kv_ownership = "KV head replicated because total KV heads < TP"
    head_dim = require_int(cfg, "head_dim")
    index_dim = require_int(cfg, "indexer_head_dim")
    index_kv_heads = require_int(cfg, "indexer_kv_heads")
    ratio = require_int(cfg, "indexer_compress_ratio")
    if index_kv_heads != 1:
        raise PlanError("QSA side-cache geometry requires one index KV head")
    raw_width = round_up(index_dim, 4) + 3 * 4  # packed int64 MRoPE axes in BF16 slots
    raw_capacity = ratio * ceil_div(ratio + speculative_tokens, ratio)
    compressed_rows = ceil_div(context, ratio)
    main_shape = [context, local_kv_heads, 2 * head_dim]
    raw_shape = [raw_capacity, 1, raw_width]
    compressed_shape = [compressed_rows, 1, index_dim]

    num_k = require_int(cfg, "linear_num_key_heads")
    num_v = require_int(cfg, "linear_num_value_heads")
    key_dim = require_int(cfg, "linear_key_head_dim")
    value_dim = require_int(cfg, "linear_value_head_dim")
    conv_kernel = require_int(cfg, "linear_conv_kernel_dim")
    if num_v % tp or (2 * key_dim * num_k + value_dim * num_v) % tp:
        raise PlanError("linear-attention state cannot be evenly TP-sharded")
    conv_width = (2 * key_dim * num_k + value_dim * num_v) // tp
    conv_len = conv_kernel - 1 + speculative_tokens
    recurrent_shape = [num_v // tp, value_dim, key_dim]

    if cfg.get("ple_layer_ids") != [2]:
        raise PlanError("PLE layer placement drift")
    hc_hidden = require_int(cfg, "hidden_size") * require_int(cfg, "hc_count")
    ple_len = (require_int(cfg, "ple_conv_kernel_size") - 1) * require_int(cfg, "ngram_size") + speculative_tokens
    cuda_state_blocks = 2  # MambaSpec.max_memory_usage_bytes in align mode.
    conv_page = math.prod([conv_len, conv_width]) * DTYPE_BYTES[mamba_cache_dtype]
    recurrent_page = math.prod(recurrent_shape) * DTYPE_BYTES[mamba_ssm_dtype]
    gdn_page = conv_page + recurrent_page
    ple_page = ple_len * hc_hidden * DTYPE_BYTES[mamba_cache_dtype]
    largest_mamba_page = max(gdn_page, ple_page)
    attention_bytes_per_token = local_kv_heads * 2 * head_dim * DTYPE_BYTES[kv_dtype]
    alignment_page_bytes = kernel_block_alignment * attention_bytes_per_token
    effective_block_size = kernel_block_alignment * ceil_div(largest_mamba_page, alignment_page_bytes)
    effective_block_size = max(block_size, effective_block_size)
    if effective_block_size % ratio or effective_block_size % raw_capacity:
        raise PlanError("derived effective block size violates QSA cache divisibility")
    main_page = effective_block_size * attention_bytes_per_token
    compressed_page = (effective_block_size // ratio) * index_dim * DTYPE_BYTES["bfloat16"]
    sequence_pages = ceil_div(context, effective_block_size)

    families = [
        make_family(id="target.full_attention.kv", owner="target", family="full-attention QSA main paged KV", layers=full_layers, dtype=kv_dtype, shape=main_shape, cuda_blocks=sequence_pages, cuda_page_bytes=main_page, concurrency=concurrency, tp_ownership=kv_ownership, sequence_dependence="one K and V row per token, paged to full context", host_page=host_page),
        make_family(id="target.qsa.raw", owner="target", family="QSA raw-key/MRoPE circular compressor state", layers=full_layers, dtype="bfloat16", shape=raw_shape, cuda_blocks=1, cuda_page_bytes=compressed_page, concurrency=concurrency, tp_ownership="replicated on both TP ranks", sequence_dependence="fixed 8-row ring per live stream; allocator pads its page to the compressed-cache page", host_page=host_page),
        make_family(id="target.qsa.compressed", owner="target", family="QSA compressed-key paged cache", layers=full_layers, dtype="bfloat16", shape=compressed_shape, cuda_blocks=sequence_pages, cuda_page_bytes=compressed_page, concurrency=concurrency, tp_ownership="replicated on both TP ranks", sequence_dependence="one key per complete 4-token group", host_page=host_page),
        make_family(id="target.linear.conv", owner="target", family="linear-attention convolution state", layers=linear_layers, dtype=mamba_cache_dtype, shape=[conv_len, conv_width], cuda_blocks=cuda_state_blocks, cuda_page_bytes=main_page, concurrency=concurrency, tp_ownership="TP-sharded projection width", sequence_dependence="fixed current state; align mode holds two pages padded to the main-KV page", host_page=host_page),
        make_family(id="target.linear.recurrent", owner="target", family="linear-attention recurrent matrix", layers=linear_layers, dtype=mamba_ssm_dtype, shape=recurrent_shape, cuda_blocks=cuda_state_blocks, cuda_page_bytes=main_page, concurrency=concurrency, tp_ownership="TP-sharded value heads", sequence_dependence="shares each padded GDN page with convolution state; allocated bytes are reported jointly", host_page=host_page, shares_cuda_allocation_with="target.linear.conv"),
        make_family(id="target.ple.conv", owner="target", family="PLE dilated short-convolution state", layers=1, dtype=mamba_cache_dtype, shape=[ple_len, hc_hidden], cuda_blocks=cuda_state_blocks, cuda_page_bytes=main_page, concurrency=concurrency, tp_ownership="replicated on both TP ranks (tp_replicated=True)", sequence_dependence="fixed 12-row state; align mode holds two pages padded to the main-KV page", host_page=host_page),
        make_family(id="mtp.full_attention.kv", owner="mtp", family="MTP parity QSA main paged KV", layers=mtp_layers, dtype=kv_dtype, shape=main_shape, cuda_blocks=sequence_pages, cuda_page_bytes=main_page, concurrency=concurrency, tp_ownership=kv_ownership, sequence_dependence="one K and V row per token for the full-attention draft layer", host_page=host_page),
        make_family(id="mtp.qsa.raw", owner="mtp", family="MTP parity QSA raw-key/MRoPE ring", layers=mtp_layers, dtype="bfloat16", shape=raw_shape, cuda_blocks=1, cuda_page_bytes=compressed_page, concurrency=concurrency, tp_ownership="replicated on both TP ranks", sequence_dependence="fixed 8-row ring per live stream; allocator pads its page to the compressed-cache page", host_page=host_page),
        make_family(id="mtp.qsa.compressed", owner="mtp", family="MTP parity QSA compressed-key cache", layers=mtp_layers, dtype="bfloat16", shape=compressed_shape, cuda_blocks=sequence_pages, cuda_page_bytes=compressed_page, concurrency=concurrency, tp_ownership="replicated on both TP ranks", sequence_dependence="one key per complete 4-token group", host_page=host_page),
    ]
    # GDN convolution and recurrent tensors occupy one shared padded Mamba page.
    # Avoid counting that physical page twice while preserving both logical rows.
    allocated = sum(x.cuda_allocated_bytes_per_stream for x in families if x.cuda_allocation_accounted)
    logical = sum(x.logical_bytes_per_stream for x in families)
    padded = sum(x.nvme_padded_bytes_per_stream for x in families)
    return {
        "schema": "rocket.qwen38.state-capacity.v1",
        "status": "allocation-plan-only",
        "checkpoint": {"revision": revision, "config_sha256": digest, "source_weight_dtype": cfg["dtype"]},
        "serving": {"context_tokens": context, "concurrency": concurrency, "tensor_parallel_size": tp, "requested_block_size_tokens": block_size, "kernel_block_alignment_tokens": kernel_block_alignment, "effective_block_size_tokens": effective_block_size, "sequence_pages": sequence_pages, "kv_cache_dtype": kv_dtype, "mamba_cache_dtype": mamba_cache_dtype, "mamba_ssm_cache_dtype": mamba_ssm_dtype, "mamba_cache_mode": mamba_cache_mode, "mtp_speculative_tokens": speculative_tokens, "host_page_bytes": host_page},
        "implementation": {"image": IMAGE, "image_id": image_id, "source_sha256": source_hashes},
        "families": [asdict(x) for x in families],
        "totals_per_rank": {"logical_bytes_per_stream": logical, "cuda_allocated_bytes_per_stream": allocated, "cuda_allocated_bytes_c16": allocated * concurrency, "nvme_padded_bytes_per_stream": padded, "nvme_padded_bytes_c16": padded * concurrency, "cuda_double_count_exclusion": "target.linear.recurrent shares the target.linear.conv GDN Mamba page"},
        "restore_contract": {"boundary": "one accepted target-token count shared by target, MTP, QSA, GDN, and PLE records", "publication": "write and checksum every family record before atomically publishing the boundary manifest", "partial_restore": "forbidden", "host_io_alignment_bytes": host_page},
        "parity_exclusions": [{"state": "MTP multi_hidden, logits, top-k indices, and forward metadata", "reason": "step-local scratch is recomputed after restore; pinned mtp.py persists sequence history only through its full-attention QSA caches"}],
        "proof": {"cuda_allocation": "not-run", "acceptance": "v2 status=passed after GPU fill, deterministic readback from every 65536-byte page, and allocator/resident deltas each cover the full extent", "remaining_gap": "allocate and bind every listed tensor inside one model-resident rank, then report allocator deltas and an evict/restore parity trace at the shared accepted-token boundary"},
        "next_allocation_command": "docker cp scripts/memory/qwen38-state-capacity.py rocket-qwen38-calibration-head:/rocket/run/qwen38-state-capacity.py && docker cp scripts/memory/qwen38-state-capacity-plan.json rocket-qwen38-calibration-head:/rocket/run/qwen38-state-capacity-plan.json && { rc=0; docker exec rocket-qwen38-calibration-head python3 /rocket/run/qwen38-state-capacity.py --cuda-allocate-plan /rocket/run/qwen38-state-capacity-plan.json --cuda-proof-output /rocket/run/qwen38-state-capacity-cuda-proof-v2.json --cuda-timeout-seconds 900 || rc=$?; docker cp rocket-qwen38-calibration-head:/rocket/run/qwen38-state-capacity-cuda-proof-v2.json /home/glwillen/calibration/qwen38-attention-nvfp4-live-20260907-01/qwen38-state-capacity-cuda-proof-v2.json; exit $rc; }",
    }


def validate_cuda_proof(proof: dict[str, Any], expected_bytes: int) -> None:
    if proof.get("schema") != "rocket.qwen38.state-capacity.cuda-allocation.v2":
        raise PlanError("CUDA proof must use touched-allocation schema v2")
    if proof.get("status") != "passed":
        raise PlanError(f"CUDA proof did not pass: {proof.get('status')!r}")
    for name in (
        "requested_bytes", "storage_bytes", "allocator_allocated_delta_bytes",
        "resident_delta_bytes", "touched_bytes", "verified_pages",
        "expected_pages",
    ):
        if type(proof.get(name)) is not int or proof[name] <= 0:
            raise PlanError(f"CUDA proof lacks positive {name}")
    if proof["requested_bytes"] != expected_bytes:
        raise PlanError("CUDA proof requested-byte total does not match the plan")
    if proof["storage_bytes"] != expected_bytes:
        raise PlanError("CUDA proof storage-byte total does not match the plan")
    if proof["touched_bytes"] != expected_bytes:
        raise PlanError("CUDA proof did not write the full byte extent")
    if proof["allocator_allocated_delta_bytes"] < expected_bytes:
        raise PlanError("CUDA allocator delta does not cover the full plan")
    if proof["resident_delta_bytes"] < expected_bytes:
        raise PlanError("CUDA resident-memory delta does not cover the full plan")
    if proof["verified_pages"] != proof["expected_pages"]:
        raise PlanError("CUDA proof did not read back every touched page")
    if not proof.get("deterministic_readback"):
        raise PlanError("CUDA proof lacks deterministic readback")


def _readback_pages(
    tensor: Any, pattern: int, page_bytes: int, chunk_pages: int,
) -> int:
    """Read one deterministic byte from every touched page using small copies."""
    size = tensor.numel()
    pages = ceil_div(size, page_bytes)
    for first_page in range(0, pages, chunk_pages):
        last_page = min(first_page + chunk_pages, pages)
        start = first_page * page_bytes
        stop = min(last_page * page_bytes, size)
        sample = tensor[start:stop:page_bytes].cpu().tolist()
        if len(sample) != last_page - first_page or any(x != pattern for x in sample):
            raise PlanError(f"CUDA deterministic page readback failed at page {first_page}")
    if int(tensor[-1].cpu().item()) != pattern:
        raise PlanError("CUDA deterministic tail-byte readback failed")
    return pages


def cuda_allocate_plan(
    plan_path: Path, output: Path, device: str, timeout_seconds: int,
    touch_page_bytes: int = 65536, readback_chunk_pages: int = 4096,
) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "rocket.qwen38.state-capacity.v1":
        raise PlanError("CUDA probe requires a Qwen3.8 state-capacity v1 plan")
    if plan.get("status") != "allocation-plan-only":
        raise PlanError("CUDA probe refuses an unknown plan status")
    if timeout_seconds <= 0:
        raise PlanError("CUDA probe timeout must be positive")
    if touch_page_bytes != plan.get("serving", {}).get("host_page_bytes"):
        raise PlanError("CUDA touch page must match the plan host-page contract")
    if readback_chunk_pages <= 0:
        raise PlanError("CUDA readback chunk must contain at least one page")
    try:
        import torch
    except ImportError as exc:
        raise PlanError("CUDA probe requires PyTorch in the model image") from exc
    if not torch.cuda.is_available():
        raise PlanError("CUDA probe requires an available CUDA device")
    torch.cuda.synchronize(device)
    free_before, total = torch.cuda.mem_get_info(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    requested_total = sum(
        row["cuda_allocated_bytes_c16"] for row in plan["families"]
        if row.get("cuda_allocation_accounted", True)
    )
    allocations = []
    held = []
    started = time.monotonic()
    failure: Exception | None = None
    final_verified_pages = 0
    try:
        for family_index, row in enumerate(plan["families"]):
            if not row.get("cuda_allocation_accounted", True):
                continue
            requested = row.get("cuda_allocated_bytes_c16")
            if type(requested) is not int or requested <= 0:
                raise PlanError(f"invalid CUDA byte count for {row.get('id')}")
            pattern = (family_index * 37 + 17) % 251 + 1
            family_started = time.monotonic()
            tensor = torch.empty(requested, dtype=torch.uint8, device=device)
            held.append(tensor)
            allocation = {
                "id": row["id"], "requested_bytes": requested,
                "storage_bytes": tensor.untyped_storage().nbytes(),
                "fill_pattern_uint8": pattern,
                "full_extent_gpu_write": False,
            }
            allocations.append(allocation)
            tensor.fill_(pattern)
            torch.cuda.synchronize(device)
            allocation["full_extent_gpu_write"] = True
            verified_pages = _readback_pages(
                tensor, pattern, touch_page_bytes, readback_chunk_pages
            )
            torch.cuda.synchronize(device)
            allocation["initial_verified_pages"] = verified_pages
            allocation["elapsed_seconds"] = time.monotonic() - family_started
            if time.monotonic() - started > timeout_seconds:
                raise PlanError("CUDA allocation probe exceeded its time bound")
        # Re-read every allocation after the last fill. Immediate per-family
        # checks alone can pass while later allocations evict or corrupt older
        # unified-memory pages.
        for tensor, allocation in zip(held, allocations):
            verified_pages = _readback_pages(
                tensor, allocation["fill_pattern_uint8"],
                touch_page_bytes, readback_chunk_pages,
            )
            allocation["final_verified_pages"] = verified_pages
            final_verified_pages += verified_pages
            if time.monotonic() - started > timeout_seconds:
                raise PlanError("CUDA allocation probe exceeded its time bound")
        torch.cuda.synchronize(device)
    except Exception as exc:  # Preserve an actionable artifact on CUDA OOM/fault.
        failure = exc
    try:
        torch.cuda.synchronize(device)
    except Exception as exc:
        failure = failure or exc
    free_after, total_after = torch.cuda.mem_get_info(device)
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    storage_total = sum(x["storage_bytes"] for x in allocations)
    touched_total = sum(
        x["requested_bytes"] for x in allocations
        if x["full_extent_gpu_write"]
    )
    expected_pages = sum(
        ceil_div(
            row["cuda_allocated_bytes_c16"], touch_page_bytes
        ) for row in plan["families"]
        if row.get("cuda_allocation_accounted", True)
    )
    deterministic_readback = (
        failure is None
        and storage_total == requested_total
        and touched_total == requested_total
        and final_verified_pages == expected_pages
    )
    resident_delta = free_before - free_after
    allocated_delta = allocated_after - allocated_before
    reserved_delta = reserved_after - reserved_before
    status = "passed"
    error = None
    if failure is not None:
        status = "failed"
        error = f"{type(failure).__name__}: {failure}"
    elif total_after != total:
        status, error = "failed", "CUDA device total changed during allocation probe"
    elif storage_total != requested_total:
        status, error = "failed", "not every planned allocation completed"
    elif touched_total != requested_total:
        status, error = "failed", "GPU writes did not touch the full byte extent"
    elif not deterministic_readback:
        status, error = "failed", "deterministic readback did not cover every touched page"
    elif allocated_delta < requested_total:
        status, error = "failed", "CUDA allocator delta does not cover the full byte extent"
    elif resident_delta < requested_total:
        status, error = "failed", "device-free memory did not fall by the full touched byte extent"
    result = {
        "schema": "rocket.qwen38.state-capacity.cuda-allocation.v2",
        "status": status,
        "plan_sha256": sha256(plan_path.read_bytes()),
        "device": str(device), "free_bytes_before": free_before,
        "free_bytes_after": free_after, "total_bytes": total,
        "resident_delta_bytes": resident_delta,
        "allocator_allocated_bytes_before": allocated_before,
        "allocator_allocated_bytes_after": allocated_after,
        "allocator_allocated_delta_bytes": allocated_delta,
        "allocator_reserved_bytes_before": reserved_before,
        "allocator_reserved_bytes_after": reserved_after,
        "allocator_reserved_delta_bytes": reserved_delta,
        "requested_bytes": requested_total,
        "storage_bytes": storage_total,
        "touched_bytes": touched_total,
        "touch_page_bytes": touch_page_bytes,
        "verified_pages": final_verified_pages,
        "expected_pages": expected_pages,
        "deterministic_readback": deterministic_readback,
        "elapsed_seconds": time.monotonic() - started,
        "allocations": allocations,
        "error": error,
        "scope": "fully touched CUDA capacity beside the resident model process; engine binding and restore parity remain separate proof gates",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "passed":
        raise PlanError(error or "CUDA allocation proof failed")
    validate_cuda_proof(result, requested_total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-allocate-plan", type=Path)
    parser.add_argument("--cuda-proof-output", type=Path)
    parser.add_argument("--cuda-device", default="cuda:0")
    parser.add_argument("--cuda-timeout-seconds", type=int, default=900)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--context", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--kernel-block-alignment", type=int)
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--mamba-cache-dtype")
    parser.add_argument("--mamba-ssm-cache-dtype")
    parser.add_argument("--mamba-cache-mode")
    parser.add_argument("--mtp-speculative-tokens", type=int)
    parser.add_argument("--host-page-bytes", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.cuda_allocate_plan is not None:
            if args.cuda_proof_output is None:
                raise PlanError("--cuda-allocate-plan requires --cuda-proof-output")
            cuda_allocate_plan(
                args.cuda_allocate_plan, args.cuda_proof_output, args.cuda_device,
                args.cuda_timeout_seconds,
            )
            return 0
        required = {
            "config": args.config, "revision": args.revision,
            "context": args.context, "concurrency": args.concurrency,
            "tensor_parallel_size": args.tensor_parallel_size,
            "block_size": args.block_size, "kv_cache_dtype": args.kv_cache_dtype,
            "kernel_block_alignment": args.kernel_block_alignment,
            "mamba_cache_dtype": args.mamba_cache_dtype,
            "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
            "mamba_cache_mode": args.mamba_cache_mode,
            "mtp_speculative_tokens": args.mtp_speculative_tokens,
            "host_page_bytes": args.host_page_bytes,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise PlanError(f"missing required planning arguments: {', '.join(missing)}")
        if args.image != IMAGE:
            raise PlanError(f"image must be pinned by digest to {IMAGE}")
        image_id, sources = read_image_sources(args.image)
        hashes = validate_sources(image_id, sources)
        plan = build_plan(
            args.config.read_bytes(), revision=args.revision, context=args.context,
            concurrency=args.concurrency, tp=args.tensor_parallel_size,
            block_size=args.block_size, kernel_block_alignment=args.kernel_block_alignment,
            kv_dtype=args.kv_cache_dtype,
            mamba_cache_dtype=args.mamba_cache_dtype,
            mamba_ssm_dtype=args.mamba_ssm_cache_dtype,
            speculative_tokens=args.mtp_speculative_tokens,
            mamba_cache_mode=args.mamba_cache_mode,
            host_page=args.host_page_bytes, image_id=image_id,
            source_hashes=hashes,
        )
    except (OSError, subprocess.SubprocessError, PlanError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
