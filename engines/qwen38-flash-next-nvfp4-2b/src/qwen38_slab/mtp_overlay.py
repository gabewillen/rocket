# SPDX-License-Identifier: Apache-2.0
"""MTP routed-expert source/serving traffic and quality gate.

The candidate keeps the pinned NVIDIA FP8 source immutable and derives a
ModelOpt NVFP4 serving overlay with the same layout consumed by the existing
FlashInfer SM121 b12x adapter. Traffic uses actual-router unique-local counts.
It makes no model-quality claim without every required per-family result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from .contract import MODEL_NVFP4_ABI, MTP_FP8_ABI

FP8_SOURCE_BYTES_PER_EXPERT = 4_915_800
NVFP4_SERVING_BYTES_PER_EXPERT = 2_764_824
MERGED_SLAB_K4_CURRENT_AGGREGATE_TOK_S = 457.524496688903
MERGED_SLAB_K4_CURRENT_STREAM_TOK_S = 28.59528104305644
MERGED_SLAB_K4_NVFP4_AGGREGATE_TOK_S = 469.3926469156607
MERGED_SLAB_K4_NVFP4_STREAM_TOK_S = 29.337040432228793
PROJECTION_FAMILIES = ("gate_proj", "up_proj", "down_proj")
INTERACTION_FAMILIES = (
    "target_attention_nvfp4",
    "target_router_nvfp4",
    "ple_nvfp4",
)


class MtpOverlayError(ValueError):
    """The serving overlay or its evidence is incomplete."""


@dataclass(frozen=True)
class QuantizationEvidence:
    activation_ranges: Mapping[str, tuple[float, float]]
    output_error: Mapping[str, float]
    logit_drift: float
    token_divergence: float
    bytes_per_accepted_token: float
    quantized_attention_interactions: Mapping[str, float]


@dataclass(frozen=True)
class MtpOverlayTraffic:
    source_dtype: str
    serving_dtype: str
    unique_local_experts_by_step: tuple[int, ...]
    source_weight_bytes: int
    serving_weight_bytes: int
    saved_weight_bytes: int
    current_control_tokens_per_second: float
    hypothetical_tokens_per_second: float
    current_control_stream_tokens_per_second: float
    hypothetical_stream_tokens_per_second: float
    quality_evidence_complete: bool


def compare_mtp_expert_overlay(
    unique_local_experts_by_step: tuple[int, ...],
    evidence: QuantizationEvidence | None = None,
) -> MtpOverlayTraffic:
    """Compare immutable FP8 source bytes with candidate NVFP4 serving bytes."""

    if (
        not isinstance(unique_local_experts_by_step, tuple)
        or not 1 <= len(unique_local_experts_by_step) <= 7
        or any(
            isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 256
            for count in unique_local_experts_by_step
        )
    ):
        raise MtpOverlayError("actual-router unique-local counts must cover K1 through K7")
    touches = sum(unique_local_experts_by_step)
    source = touches * FP8_SOURCE_BYTES_PER_EXPERT
    serving = touches * NVFP4_SERVING_BYTES_PER_EXPERT
    return MtpOverlayTraffic(
        MTP_FP8_ABI,
        MODEL_NVFP4_ABI,
        unique_local_experts_by_step,
        source,
        serving,
        source - serving,
        MERGED_SLAB_K4_CURRENT_AGGREGATE_TOK_S,
        MERGED_SLAB_K4_NVFP4_AGGREGATE_TOK_S,
        MERGED_SLAB_K4_CURRENT_STREAM_TOK_S,
        MERGED_SLAB_K4_NVFP4_STREAM_TOK_S,
        _complete_evidence(evidence),
    )


def _complete_evidence(evidence: QuantizationEvidence | None) -> bool:
    if evidence is None:
        return False
    if not isinstance(evidence.activation_ranges, Mapping) or not isinstance(
        evidence.output_error, Mapping
    ) or not isinstance(evidence.quantized_attention_interactions, Mapping):
        raise MtpOverlayError("quality evidence mappings are required")
    if (
        set(evidence.activation_ranges) != set(PROJECTION_FAMILIES)
        or set(evidence.output_error) != set(PROJECTION_FAMILIES)
        or set(evidence.quantized_attention_interactions) != set(INTERACTION_FAMILIES)
    ):
        raise MtpOverlayError("every MTP family and quantized interaction must be measured")
    scalars = (
        *evidence.output_error.values(),
        evidence.logit_drift,
        evidence.token_divergence,
        evidence.bytes_per_accepted_token,
        *evidence.quantized_attention_interactions.values(),
    )
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in scalars):
        raise MtpOverlayError("quality evidence values must be finite and nonnegative")
    if any(
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounds)
        or bounds[0] > bounds[1]
        for bounds in evidence.activation_ranges.values()
    ):
        raise MtpOverlayError("activation ranges are invalid")
    return True


MTP_OVERLAY_LAYOUT = MappingProxyType(
    {
        "weight": "packed E2M1 pairs in U8",
        "weight_scale": "FP8 E4M3, one scale per 16 values",
        "weight_scale_2": "F32 global projection scale",
        "input_scale": "F32 activation scale",
        "kernel": "FlashInfer 91bda04 SM121 b12x ModelOpt NVFP4",
    }
)
