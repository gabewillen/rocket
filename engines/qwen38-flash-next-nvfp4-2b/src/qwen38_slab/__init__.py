"""Pinned Qwen3.8 rank-slab materialization and direct-I/O loading."""

from .contract import PINNED_CONTRACT, SlabContract, SlabError
from .loader import DirectSlabLoader
from .materialize import materialize

__all__ = [
    "DirectSlabLoader",
    "PINNED_CONTRACT",
    "SlabContract",
    "SlabError",
    "materialize",
]
