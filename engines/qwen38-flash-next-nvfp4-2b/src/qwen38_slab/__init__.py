"""Pinned Qwen3.8 rank-slab materialization and direct-I/O loading."""

from .contract import PINNED_CONTRACT, SlabContract, SlabError
from .cuda_slab_loader import CudaRankSlabLoader, CudaSlabLoadError
from .loader import DirectSlabLoader
from .materialize import materialize

__all__ = [
    "DirectSlabLoader",
    "CudaRankSlabLoader",
    "CudaSlabLoadError",
    "PINNED_CONTRACT",
    "SlabContract",
    "SlabError",
    "materialize",
]
