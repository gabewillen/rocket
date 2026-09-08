"""Pinned Qwen3.8 rank-slab materialization and direct-I/O loading."""

from .contract import PINNED_CONTRACT, SlabContract, SlabError
from .cuda_slab_loader import CudaRankSlabLoader, CudaSlabLoadError
from .k0_composition import (
    K0CompositionBinding,
    K0CompositionError,
    K0CompositionRoot,
    K0_DOMAIN,
    REQUIRED_K0_PARTICIPANTS,
)
from .loader import DirectSlabLoader
from .materialize import materialize
from .whole_decoder import (
    DecoderSlabs,
    DraftArchitecture,
    GraphKey,
    WholeDecoderColdLoader,
    WholeDecoderError,
    WholeDecoderExecutor,
)

__all__ = [
    "DirectSlabLoader",
    "DecoderSlabs",
    "DraftArchitecture",
    "GraphKey",
    "K0CompositionBinding",
    "K0CompositionError",
    "K0CompositionRoot",
    "K0_DOMAIN",
    "REQUIRED_K0_PARTICIPANTS",
    "CudaRankSlabLoader",
    "CudaSlabLoadError",
    "PINNED_CONTRACT",
    "SlabContract",
    "SlabError",
    "WholeDecoderColdLoader",
    "WholeDecoderError",
    "WholeDecoderExecutor",
    "materialize",
]
