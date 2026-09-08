# SPDX-License-Identifier: Apache-2.0
"""Graph-owned layer-3 c1 storage and two-rank production dependency gate."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .k0_composition import K0_DOMAIN
from .native_qsa import ARENA_FIELDS, STATE_POINTER_FIELDS

HIDDEN = 2560
MAIN_WIDTH = 256
INDEX_WIDTH = 128
MAIN_PAGE = 1600
COMPRESSED_PAGE = 400
TABLE_PAGES = 164
RAW_ROWS = 8
RAW_WIDTH = 140

ARENA_SPECS = {
    "qkv_packed": ((HIDDEN // 2,), "uint8"),
    "qkv_sfa": ((128 * 40 * 4,), "uint8"),
    "raw_main_qkv": ((6656,), "bfloat16"),
    "index_projected_qk": ((640,), "bfloat16"),
    "main_query": ((12, MAIN_WIDTH), "bfloat16"),
    "attention_gate": ((12, MAIN_WIDTH), "bfloat16"),
    "index_query": ((4, INDEX_WIDTH), "bfloat16"),
    "index_logits": ((65536,), "float32"),
    "visible_blocks": ((1,), "int32"),
    "selected_blocks": ((512,), "int32"),
    "selected_tokens": ((2051,), "int32"),
    "attention_partial": ((32, 1, 12, MAIN_WIDTH), "float32"),
    "attention_lse": ((32, 1, 12), "float32"),
    "attention_output": ((12, MAIN_WIDTH), "bfloat16"),
    "gated_attention": ((3072,), "bfloat16"),
    "output_packed": ((3072 // 2,), "uint8"),
    "output_sfa": ((128 * 48 * 4,), "uint8"),
    "projected_output": ((HIDDEN,), "bfloat16"),
}

DEPENDENCY_SUFFIXES = (
    "target_slab", "indexer_sidecar_slab", "qsa_graph", "hyperconnection",
    "attention_pair_reduce", "target_router", "routed_experts",
    "shared_expert", "moe_pair_reduce",
)
REQUIRED_LAYER3_DEPENDENCIES = (
    "oracle_comparator",
    *(f"rank{rank}.{suffix}" for rank in (0, 1) for suffix in DEPENDENCY_SUFFIXES),
)


class Layer3RuntimeError(RuntimeError):
    def __init__(self, message: str, *, missing: tuple[str, ...] = ()):
        super().__init__(message)
        self.missing = missing


@dataclass
class Layer3QsaStorage:
    """All mutable c1 QSA buffers for one rank/layer-3 graph."""

    device: str
    max_tokens: int
    arena: Mapping[str, object]
    state: Mapping[str, object]
    main_blocks: int
    compressed_blocks: int
    generation: int = 0

    def advance(self, position: int) -> int:
        if not 0 <= position < self.max_tokens or position != self.generation:
            raise Layer3RuntimeError("layer-3 QSA position/generation changed")
        self.generation += 1
        values = {
            "positions": position,
            "main_slot_mapping": position,
            "raw_slot_mapping": position % RAW_ROWS,
            "compressed_slot_mapping": position // 4 if position % 4 == 3 else -1,
            "query_start_locations": 0,
            "logical_positions": position,
            "sequence_lengths": position + 1,
            "token_to_request": 0,
            "compression_work": 1 if position % 4 == 3 else 0,
        }
        for name, value in values.items():
            self.state[name].fill_(value)
        return self.generation


def allocate_layer3_qsa_storage(torch_api, *, device: str,
                                max_tokens: int) -> Layer3QsaStorage:
    """Allocate once before capture. Replay mutates only fixed tensor contents."""

    if (
        torch_api is None or not isinstance(device, str)
        or not device.startswith("cuda:") or not 1 <= max_tokens <= 262144
    ):
        raise Layer3RuntimeError("layer-3 QSA storage configuration changed")
    main_blocks = (max_tokens + MAIN_PAGE - 1) // MAIN_PAGE
    compressed_blocks = main_blocks
    arena = {
        name: torch_api.empty(shape, dtype=getattr(torch_api, dtype), device=device)
        for name, (shape, dtype) in ARENA_SPECS.items()
    }
    state = {
        "main_key_cache": torch_api.empty(
            (main_blocks, MAIN_PAGE, MAIN_WIDTH), dtype=torch_api.bfloat16,
            device=device,
        ),
        "main_value_cache": torch_api.empty(
            (main_blocks, MAIN_PAGE, MAIN_WIDTH), dtype=torch_api.bfloat16,
            device=device,
        ),
        "raw_key_cache": torch_api.empty(
            (1, RAW_ROWS, RAW_WIDTH), dtype=torch_api.bfloat16, device=device,
        ),
        "compressed_key_cache": torch_api.empty(
            (compressed_blocks, COMPRESSED_PAGE, INDEX_WIDTH),
            dtype=torch_api.bfloat16, device=device,
        ),
    }
    for name in STATE_POINTER_FIELDS[4:]:
        dtype = torch_api.int64 if name in ("positions", "logical_positions") else torch_api.int32
        length = TABLE_PAGES if name in ("main_block_table", "compressed_block_table") else 1
        state[name] = torch_api.full((length,), -1, dtype=dtype, device=device)
    state["raw_block_table"].fill_(0)
    for index in range(main_blocks):
        state["main_block_table"][index] = index
        state["compressed_block_table"][index] = index
    return Layer3QsaStorage(
        device, max_tokens, MappingProxyType(arena), MappingProxyType(state),
        main_blocks, compressed_blocks,
    )


@dataclass(frozen=True)
class TwoRankLayer3Binding:
    dependencies: Mapping[str, object]


class TwoRankLayer3Factory:
    """Publish the layer-3 launch root only with concrete production ports."""

    def bind(self, dependencies: Mapping[str, object]) -> TwoRankLayer3Binding:
        if not isinstance(dependencies, Mapping):
            raise Layer3RuntimeError(
                "layer-3 dependencies must be a mapping",
                missing=REQUIRED_LAYER3_DEPENDENCIES,
            )
        unknown = tuple(sorted(set(dependencies) - set(REQUIRED_LAYER3_DEPENDENCIES)))
        if unknown:
            raise Layer3RuntimeError(f"unknown layer-3 dependency: {unknown[0]}")
        missing = tuple(name for name in REQUIRED_LAYER3_DEPENDENCIES if dependencies.get(name) is None)
        if missing:
            raise Layer3RuntimeError(
                f"missing layer-3 dependency: {missing[0]}", missing=missing
            )
        for name in REQUIRED_LAYER3_DEPENDENCIES:
            dependency = dependencies[name]
            slab = name.endswith(("target_slab", "indexer_sidecar_slab"))
            comparator = name == "oracle_comparator"
            if slab:
                if (
                    "uint8" not in str(getattr(dependency, "dtype", ""))
                    or not str(getattr(dependency, "device", "")).startswith("cuda:")
                ):
                    raise Layer3RuntimeError(
                        f"layer-3 resident slab changed: {name}"
                    )
            elif (
                getattr(dependency, "execution_domain", None) != K0_DOMAIN
                or getattr(dependency, "production_eligible", None) is not True
                or (not comparator and
                    getattr(dependency, "graph_safe", None) is not True)
            ):
                raise Layer3RuntimeError(
                    f"layer-3 dependency is not production graph-safe: {name}"
                )
            method = _required_method(name)
            if not callable(getattr(dependency, method, None)):
                raise Layer3RuntimeError(
                    f"layer-3 dependency interface changed: {name}.{method}"
                )
            if name.endswith("hyperconnection") and not callable(
                getattr(dependency, "combine", None)
            ):
                raise Layer3RuntimeError(
                    f"layer-3 dependency interface changed: {name}.combine"
                )
            if name.startswith("rank") and not slab:
                expected_rank = int(name[4])
                if getattr(dependency, "rank", None) != expected_rank:
                    raise Layer3RuntimeError(
                        f"layer-3 dependency rank changed: {name}"
                    )
            if name.endswith(("qsa_graph", "hyperconnection", "target_router",
                              "routed_experts", "shared_expert")) and getattr(
                dependency, "layer", None
            ) != 3:
                raise Layer3RuntimeError(
                    f"layer-3 dependency layer changed: {name}"
                )
        return TwoRankLayer3Binding(MappingProxyType({
            name: dependencies[name] for name in REQUIRED_LAYER3_DEPENDENCIES
        }))


def _required_method(name: str) -> str:
    suffix = name.split(".")[-1]
    return {
        "oracle_comparator": "compare",
        "target_slab": "data_ptr",
        "indexer_sidecar_slab": "data_ptr",
        "qsa_graph": "launch",
        "hyperconnection": "mix",
        "attention_pair_reduce": "reduce",
        "target_router": "enqueue",
        "routed_experts": "enqueue",
        "shared_expert": "enqueue",
        "moe_pair_reduce": "reduce",
    }[suffix]


__all__ = ["ARENA_SPECS", "REQUIRED_LAYER3_DEPENDENCIES",
           "Layer3QsaStorage", "Layer3RuntimeError", "TwoRankLayer3Binding",
           "TwoRankLayer3Factory", "allocate_layer3_qsa_storage"]
