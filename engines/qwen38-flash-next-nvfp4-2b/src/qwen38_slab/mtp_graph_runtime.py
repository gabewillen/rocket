# SPDX-License-Identifier: Apache-2.0
"""Owned CUDA-graph runtime for the pinned one-layer Qwen3.8 MTP block.

Cold construction receives the already weight-loaded pinned vLLM MTP module,
fixed arena tensors, and the active attention metadata context. It captures the
entire K-step proposal chain once. The hot path performs one graph replay and
does not invoke Python model callbacks or loop over draft positions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .decode import Depth, GRAPH_BATCHES
from .mtp_source import NativeMtpSource

PINNED_MTP_SOURCE_SHA256 = "7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431"
PINNED_MTP_CLASS = "Qwen3_8FlashNextMTP"
HC_WIDTH = 10_240
VOCAB = 248_320


class MtpGraphRuntimeError(RuntimeError):
    """Pinned module, arena, graph residency, or generation changed."""


class Residency(str, Enum):
    RESIDENT = "resident"
    LAZY = "lazy"


@dataclass(frozen=True)
class MtpGraphKey:
    depth: Depth
    sequences: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.depth, bool)
            or self.depth is Depth.K0
            or self.sequences not in GRAPH_BATCHES
            or (int(self.depth) > 4 and self.sequences > 4)
        ):
            raise MtpGraphRuntimeError("MTP graph key is outside K1-K7 policy")


@dataclass(frozen=True)
class MtpArena:
    token_ids: object
    positions: object
    multi_hidden: object
    verification_tokens: object
    causal_snapshots: object
    accepted_widths: object
    inactive_causal_state: object


@dataclass(frozen=True)
class MtpDeviceDraftView:
    generation: int
    key: MtpGraphKey
    verification_tokens: object
    causal_snapshots: object


@dataclass
class _Captured:
    key: MtpGraphKey
    arena: MtpArena
    proposal_graph: object
    accept_graph: object
    generation: int = 0
    pending: bool = False


class MtpGraphRuntime:
    """Single-writer owner of immutable resident and lazy MTP graphs."""

    def __init__(self, source: NativeMtpSource, module: object, *, torch_api=None):
        if torch_api is None:
            import torch as torch_api  # type: ignore[no-redef]
        if (
            not isinstance(source, NativeMtpSource)
            or source.slab_bytes != 1_370_161_152
            or len(source.nonexpert_extents) != 29
            or len(source.expert_extents) != 1_536
            or type(module).__name__ != PINNED_MTP_CLASS
        ):
            raise MtpGraphRuntimeError("authenticated MTP source and pinned module are required")
        self._torch = torch_api
        self._source = source
        self._module = module
        self._graphs: dict[MtpGraphKey, _Captured] = {}
        self._faulted = False

    @staticmethod
    def authenticate_reference(path: Path) -> None:
        if not isinstance(path, Path):
            raise MtpGraphRuntimeError("pinned MTP source path is required")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise MtpGraphRuntimeError("cannot read pinned MTP implementation") from exc
        if digest != PINNED_MTP_SOURCE_SHA256:
            raise MtpGraphRuntimeError("pinned MTP implementation digest changed")

    def capture(self, key: MtpGraphKey, arena: MtpArena) -> None:
        """Capture an exact graph once; K1-K3 are resident, K4-K7 lazy."""

        if self._faulted or key in self._graphs:
            raise MtpGraphRuntimeError("MTP graph is faulted or already immutable")
        self._validate_arena(key, arena)
        torch = self._torch
        graph = torch.cuda.CUDAGraph()
        accept = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                token_ids = arena.token_ids
                positions = arena.positions
                hidden = arena.multi_hidden
                arena.verification_tokens[0].copy_(token_ids)
                # Cold capture unrolls K. Replay is one native graph launch.
                for step in range(int(key.depth)):
                    self._module.model.set_skip_topk(step > 0)
                    sample_hidden, hidden = self._module(
                        token_ids, positions, hidden_states=hidden,
                        spec_step_idx=step,
                    )
                    logits = self._module.compute_logits(sample_hidden, step)
                    token_ids = torch.argmax(logits, dim=-1).to(torch.int32)
                    arena.verification_tokens[step + 1].copy_(token_ids)
                    arena.causal_snapshots[step].copy_(hidden)
                    positions = positions + 1
            with torch.cuda.graph(accept):
                selected = (
                    torch.clamp(arena.accepted_widths, 1, int(key.depth)) - 1
                ).to(torch.int64)
                rows = torch.arange(key.sequences, device=selected.device)
                arena.inactive_causal_state.copy_(arena.causal_snapshots[selected, rows])
        except BaseException as exc:
            self._faulted = True
            raise MtpGraphRuntimeError("pinned MTP CUDA graph capture failed") from exc
        self._graphs[key] = _Captured(key, arena, graph, accept)

    def draft(self, key: MtpGraphKey, generation: int) -> MtpDeviceDraftView:
        """Hot path: exactly one replay, with no Python draft-step loop."""

        captured = self._bound(key, generation)
        try:
            captured.proposal_graph.replay()
        except BaseException:
            self._faulted = True
            raise
        captured.pending = True
        return MtpDeviceDraftView(
            generation, key, captured.arena.verification_tokens,
            captured.arena.causal_snapshots,
        )

    def stage_accept(self, key: MtpGraphKey, generation: int) -> object:
        captured = self._graphs.get(key)
        if self._faulted or captured is None or not captured.pending or generation != captured.generation + 1:
            raise MtpGraphRuntimeError("MTP accepted-state transaction changed")
        captured.accept_graph.replay()
        return captured.arena.inactive_causal_state

    def commit(self, key: MtpGraphKey, generation: int) -> None:
        captured = self._graphs.get(key)
        if captured is None or not captured.pending or generation != captured.generation + 1:
            self._faulted = True
            raise MtpGraphRuntimeError("MTP graph commit generation changed")
        captured.generation = generation
        captured.pending = False

    def discard(self, key: MtpGraphKey, generation: int) -> None:
        captured = self._graphs.get(key)
        if captured is not None and captured.pending and generation == captured.generation + 1:
            captured.pending = False

    def residency(self, key: MtpGraphKey) -> Residency:
        return Residency.RESIDENT if int(key.depth) <= 3 else Residency.LAZY

    def _bound(self, key: MtpGraphKey, generation: int) -> _Captured:
        captured = self._graphs.get(key)
        if self._faulted or captured is None or captured.pending or generation != captured.generation + 1:
            raise MtpGraphRuntimeError("MTP graph or generation is unavailable")
        return captured

    def _validate_arena(self, key: MtpGraphKey, arena: MtpArena) -> None:
        expected = (
            (arena.token_ids, (key.sequences,), "int32"),
            (arena.positions, (key.sequences,), "int64"),
            (arena.multi_hidden, (key.sequences, HC_WIDTH), "bfloat16"),
            (arena.verification_tokens, (int(key.depth) + 1, key.sequences), "int32"),
            (arena.causal_snapshots, (int(key.depth), key.sequences, HC_WIDTH), "bfloat16"),
            (arena.accepted_widths, (key.sequences,), "int32"),
            (arena.inactive_causal_state, (key.sequences, HC_WIDTH), "bfloat16"),
        )
        devices = set()
        for tensor, shape, dtype in expected:
            if tuple(getattr(tensor, "shape", ())) != shape or dtype not in str(getattr(tensor, "dtype", "")) or not getattr(tensor, "is_cuda", False):
                raise MtpGraphRuntimeError("fixed MTP arena shape, dtype, or device changed")
            devices.add(str(tensor.device))
        if len(devices) != 1:
            raise MtpGraphRuntimeError("MTP arena spans CUDA devices")
