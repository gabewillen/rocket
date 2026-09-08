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
from .layer3_runtime import (
    REQUIRED_LAYER3_DEPENDENCIES,
    Layer3QsaStorage,
    Layer3RuntimeError,
    TwoRankLayer3Binding,
    TwoRankLayer3Factory,
    allocate_layer3_qsa_storage,
)
from .materialize import materialize
from .native_qsa import (
    NativeQsaBindingError,
    NativeQsaGraphHandle,
    NativeQsaSlabPointers,
    StateView,
    bind_slab_pointers,
    bind_state_view,
)
from .target_moe import (
    TargetMoeBinding,
    TargetMoeError,
    TargetMoeGeneration,
    TargetMoeLayerParticipant,
)
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
    "Layer3QsaStorage",
    "Layer3RuntimeError",
    "REQUIRED_K0_PARTICIPANTS",
    "REQUIRED_LAYER3_DEPENDENCIES",
    "CudaRankSlabLoader",
    "CudaSlabLoadError",
    "NativeQsaBindingError",
    "NativeQsaGraphHandle",
    "NativeQsaSlabPointers",
    "StateView",
    "PINNED_CONTRACT",
    "SlabContract",
    "SlabError",
    "TargetMoeBinding",
    "TargetMoeError",
    "TargetMoeGeneration",
    "TargetMoeLayerParticipant",
    "TwoRankLayer3Binding",
    "TwoRankLayer3Factory",
    "WholeDecoderColdLoader",
    "WholeDecoderError",
    "WholeDecoderExecutor",
    "bind_slab_pointers",
    "bind_state_view",
    "allocate_layer3_qsa_storage",
    "materialize",
]
