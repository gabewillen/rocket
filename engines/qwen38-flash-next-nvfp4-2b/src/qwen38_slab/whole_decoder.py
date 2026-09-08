"""Single-owner whole-decoder topology for the specialized Qwen3.8 engine.

The executor composes fixed GDN/QSA, TP2 reduction, rank-local target MoE, and
greedy vocab-output adapters. It owns graph selection and execution order. The
adapters own CUDA buffers, streams, and kernels; values returned by adapters are
borrowed until the next synchronous adapter call.

``DecoderStateRestoreController.continuation`` is the accepted-prefix
capability. Execution requires the exact published object, so copied or stale
boundary data cannot open the launch gate. K0 binds the target slab only. K1
through K7 bind the same target and MTP slab objects. K4 through K7 are limited
to c1 through c4. Graph construction is lazy, cached by the bounded 8-depth by
5-batch by 2-draft-architecture key space, and never cached after failure.

OpenTelemetry cardinality is bounded. Decoder spans use phase (validate,
embedding, layers, sample), attention (none, gdn, qsa), depth (K0 through K7), graph batch
(1, 2, 4, 8, 16), draft architecture (native_mtp or external_draft), rank
(0 or 1), layer (-1 or 0 through 47), and outcome
(success or failure). Cold-load metrics use phase (six values), rank (0 or 1),
slab kind (all, target, mtp), and transfer direction (direct_read or h2d).
Request, session, token, digest, path, pointer, timestamp, and error values are
excluded from attributes.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterator, Mapping, Protocol

from .contract import PINNED_CONTRACT
from .controller_restore import DecoderContinuation
from .cuda_slab_loader import CudaRankSlabLoader, LoadedRankSlabs, RankLoadReceipt
from .decode import Depth, GRAPH_BATCHES
from .mtp_policy import PolicyDecision
from .routed_moe import MoeShape, SLAB_ARTIFACT_KEY

LAYERS = 48
_HEX_256 = re.compile(r"[0-9a-f]{64}\Z")


class WholeDecoderError(RuntimeError):
    """Validation or adapter failure; active-step failures fault the executor."""


class AttentionKind(str, Enum):
    GDN = "gdn"
    QSA = "qsa"


class ReductionKind(str, Enum):
    ATTENTION = "attention"
    MOE = "moe"


class DraftArchitecture(str, Enum):
    """Bounded verifier source selected before graph binding.

    ``EXTERNAL_DRAFT`` is an interface slot. It carries no acceptance or
    checkpoint-equivalence claim until a concrete adapter has its own proof.
    """

    NATIVE_MTP = "native_mtp"
    EXTERNAL_DRAFT = "external_draft"


class ExecutorPhase(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FAULTED = "faulted"


@dataclass(frozen=True)
class GraphKey:
    """One captured decoder shape with at most 128 verification rows."""

    depth: Depth
    graph_batch: int
    verify_width: int | None = None
    draft_architecture: DraftArchitecture = DraftArchitecture.NATIVE_MTP

    def __post_init__(self) -> None:
        try:
            if isinstance(self.depth, bool):
                raise ValueError("boolean depth")
            depth = Depth(self.depth)
        except (TypeError, ValueError) as exc:
            raise WholeDecoderError("graph depth must be K0 through K7") from exc
        if self.graph_batch not in GRAPH_BATCHES or isinstance(self.graph_batch, bool):
            raise WholeDecoderError("graph batch must be 1, 2, 4, 8, or 16")
        width = int(depth) + 1 if self.verify_width is None else self.verify_width
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise WholeDecoderError("verify width must be a positive integer")
        if self.graph_batch * width > 128:
            raise WholeDecoderError("sequences times verify width must be at most 128")
        if width != int(depth) + 1:
            raise WholeDecoderError("verify width must equal depth plus one")
        if depth >= Depth.K4 and self.graph_batch > 4:
            raise WholeDecoderError("K4 through K7 are lazy low-concurrency graphs")
        if not isinstance(self.draft_architecture, DraftArchitecture):
            raise WholeDecoderError("draft architecture is invalid")
        if depth is Depth.K0 and self.draft_architecture is not DraftArchitecture.NATIVE_MTP:
            raise WholeDecoderError("K0 has no draft architecture")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "verify_width", width)

    @property
    def sequences(self) -> int:
        return self.graph_batch

    @property
    def token_rows(self) -> int:
        assert self.verify_width is not None
        return self.graph_batch * self.verify_width


@dataclass(frozen=True)
class DecoderSlabs:
    """Exact immutable rank slab objects borrowed for the executor lifetime."""

    revision: str
    artifact_key: str
    rank: int
    target: object
    mtp: object
    draft_head: object | None = None

    def __post_init__(self) -> None:
        if (
            self.revision != PINNED_CONTRACT.revision
            or self.artifact_key != SLAB_ARTIFACT_KEY
            or self.rank not in (0, 1)
            or self.target is None
            or self.mtp is None
            or self.target is self.mtp
            or self.draft_head is self.target
            or self.draft_head is self.mtp
        ):
            raise WholeDecoderError("authenticated target/MTP slab identity changed")

    @classmethod
    def from_loaded(cls, loaded: LoadedRankSlabs) -> "DecoderSlabs":
        """Validate one loader publication and retain its exact CUDA objects."""

        if not isinstance(loaded, LoadedRankSlabs):
            raise WholeDecoderError("cold loader returned no rank slab publication")
        rank = loaded.receipt.rank
        target_key, mtp_key = f"rank{rank}-target", f"rank{rank}-mtp"
        if tuple(loaded.slabs) != (target_key, mtp_key):
            raise WholeDecoderError("cold loader slab publication order changed")
        return cls(
            PINNED_CONTRACT.revision,
            SLAB_ARTIFACT_KEY,
            rank,
            loaded.slabs[target_key],
            loaded.slabs[mtp_key],
        )


@dataclass(frozen=True)
class LayerOutput:
    """Attention-owned recurrent/HC state and rank-local output partial."""

    hidden: object
    local_partial: object


class BoundDecoderGraph(Protocol):
    """Immutable captured graph handle borrowed from the bounded graph cache."""

    @property
    def key(self) -> GraphKey: ...

    @property
    def target_slab(self) -> object: ...

    @property
    def mtp_slab(self) -> object | None: ...

    @property
    def draft_slab(self) -> object | None: ...


class DecoderGraphFactory(Protocol):
    """Initialization-safe graph builder; it must publish only complete graphs."""

    def bind(
        self, key: GraphKey, target_slab: object, mtp_slab: object | None,
        draft_slab: object | None,
    ) -> BoundDecoderGraph: ...


class AcceptedPrefixSource(Protocol):
    """Single source of truth for the current two-rank accepted publication."""

    @property
    def continuation(self) -> DecoderContinuation | None: ...


class AttentionExecutor(Protocol):
    """Synchronous layer adapter for one fixed GDN or QSA implementation."""

    @property
    def kind(self) -> AttentionKind: ...

    def execute(
        self, layer: int, hidden: object, graph: BoundDecoderGraph
    ) -> LayerOutput: ...


class InputEmbedder(Protocol):
    """Synchronous TP2 embedding lookup and pre-layer reduction adapter."""

    def embed(
        self, token_ids: tuple[int, ...], graph: BoundDecoderGraph
    ) -> object: ...


class MoeExecutor(Protocol):
    """Synchronous rank-local target-MoE graph adapter."""

    def execute(
        self, layer: int, hidden: object, shape: MoeShape,
        graph: BoundDecoderGraph,
    ) -> object: ...


class HiddenReducer(Protocol):
    """Synchronous PairReduce adapter returning the consumed TP2 value."""

    def reduce(
        self, layer: int, kind: ReductionKind, partial: object,
        graph: BoundDecoderGraph,
    ) -> object: ...


class OutputSampler(Protocol):
    """Synchronous final norm, vocab projection, and distributed sampling adapter."""

    def sample(
        self, hidden: object, graph: BoundDecoderGraph
    ) -> tuple[int, ...]: ...


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


class _Metric(Protocol):
    def add(self, amount: int, attributes: Mapping[str, str | int]) -> None: ...
    def record(self, amount: int, attributes: Mapping[str, str | int]) -> None: ...


class OtelMeter(Protocol):
    def create_counter(self, name: str, *, unit: str) -> _Metric: ...
    def create_histogram(self, name: str, *, unit: str) -> _Metric: ...


class WholeDecoderExecutor:
    """Run one exact layer-major decode step after accepted-prefix publication.

    The executor is synchronous, non-reentrant, and has one logical writer.
    Caller inputs are borrowed for the call. A successful tuple of sampled token
    IDs is owned by the caller. Validation and graph-bind failures leave the
    executor idle. Any failure after layer execution begins faults it because a
    remote write or recurrent-state mutation may have committed.
    """

    def __init__(
        self,
        *,
        slabs: DecoderSlabs,
        prefix_source: AcceptedPrefixSource,
        graph_factory: DecoderGraphFactory,
        embedder: InputEmbedder,
        gdn: AttentionExecutor,
        qsa: AttentionExecutor,
        moe: MoeExecutor,
        reducer: HiddenReducer,
        sampler: OutputSampler,
        tracer: OtelTracer,
    ) -> None:
        if not isinstance(slabs, DecoderSlabs):
            raise WholeDecoderError("authenticated decoder slabs are required")
        dependencies = (
            prefix_source, graph_factory, embedder, gdn, qsa, moe, reducer,
            sampler, tracer,
        )
        if any(value is None for value in dependencies):
            raise WholeDecoderError("whole-decoder dependencies are required")
        if getattr(gdn, "kind", None) is not AttentionKind.GDN:
            raise WholeDecoderError("GDN executor kind changed")
        if getattr(qsa, "kind", None) is not AttentionKind.QSA:
            raise WholeDecoderError("QSA executor kind changed")
        self._slabs = slabs
        self._prefix_source = prefix_source
        self._graph_factory = graph_factory
        self._embedder = embedder
        self._gdn = gdn
        self._qsa = qsa
        self._moe = moe
        self._reducer = reducer
        self._sampler = sampler
        self._tracer = tracer
        self._graphs: dict[GraphKey, BoundDecoderGraph] = {}
        self._phase = ExecutorPhase.IDLE
        self._last_prefix: DecoderContinuation | None = None

    @property
    def phase(self) -> ExecutorPhase:
        return self._phase

    @property
    def graph_count(self) -> int:
        return len(self._graphs)

    def graph_for(self, key: GraphKey) -> BoundDecoderGraph:
        """Return a complete graph, binding at most once for each bounded key."""

        if not isinstance(key, GraphKey):
            raise WholeDecoderError("whole decoder requires a GraphKey")
        graph = self._graphs.get(key)
        if graph is not None:
            return graph
        mtp = None if key.depth is Depth.K0 else self._slabs.mtp
        draft = None
        if (
            key.depth is not Depth.K0
            and key.draft_architecture is DraftArchitecture.EXTERNAL_DRAFT
        ):
            draft = self._slabs.draft_head
            if draft is None:
                raise WholeDecoderError(
                    "external-draft graph requires a draft-head slab"
                )
        try:
            candidate = self._graph_factory.bind(
                key, self._slabs.target, mtp, draft
            )
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            raise WholeDecoderError("decoder graph binding failed") from exc
        if (
            candidate is None
            or getattr(candidate, "key", None) != key
            or getattr(candidate, "target_slab", None) is not self._slabs.target
            or getattr(candidate, "mtp_slab", None) is not mtp
            or getattr(candidate, "draft_slab", None) is not draft
        ):
            raise WholeDecoderError("decoder graph violated slab identity contract")
        self._graphs[key] = candidate
        return candidate

    def execute(
        self, *, prefix: DecoderContinuation, decision: PolicyDecision,
        token_ids: tuple[int, ...],
        draft_architecture: DraftArchitecture = DraftArchitecture.NATIVE_MTP,
    ) -> tuple[int, ...]:
        """Execute the adaptive-policy depth and sample after 96 reductions.

        ``decision`` is borrowed from :class:`AdaptiveMtpPolicy`. Residency
        commands must be completed by the graph factory before ``bind`` returns.
        The executor does not mutate or publish ``decision.next_state``; that
        state joins the next accepted-prefix transaction owned by the controller.
        """

        if self._phase is not ExecutorPhase.IDLE:
            raise WholeDecoderError(f"whole decoder is {self._phase.value}")
        if not isinstance(decision, PolicyDecision):
            raise WholeDecoderError("adaptive MTP policy decision is required")
        if not isinstance(token_ids, tuple) or len(token_ids) not in GRAPH_BATCHES:
            raise WholeDecoderError("input tokens must use a captured sequence bucket")
        key = GraphKey(
            decision.selected_depth,
            len(token_ids),
            draft_architecture=draft_architecture,
        )
        with self._observed("validate", "none", key, -1):
            self._validate_prefix(prefix)
            if (
                not isinstance(token_ids, tuple)
                or len(token_ids) != key.sequences
                or any(
                    isinstance(token, bool) or not isinstance(token, int)
                    or not 0 <= token < 248_320
                    for token in token_ids
                )
            ):
                raise WholeDecoderError("one valid input token per sequence is required")
            graph = self.graph_for(key)
            shape = MoeShape(key.sequences, key.verify_width)

        self._phase = ExecutorPhase.ACTIVE
        try:
            with self._observed("embedding", "none", key, -1):
                hidden = self._embedder.embed(token_ids, graph)
                if hidden is None:
                    raise WholeDecoderError("embedding returned no hidden activation")
            for layer in range(LAYERS):
                attention = self._qsa if _attention_for(layer) is AttentionKind.QSA else self._gdn
                with self._observed("layers", attention.kind.value, key, layer):
                    result = attention.execute(layer, hidden, graph)
                    if (
                        not isinstance(result, LayerOutput)
                        or result.hidden is None
                        or result.local_partial is None
                    ):
                        raise WholeDecoderError("attention output contract changed")
                    hidden = self._reducer.reduce(
                        layer, ReductionKind.ATTENTION, result.local_partial, graph
                    )
                    if hidden is None:
                        raise WholeDecoderError("attention PairReduce returned no hidden state")
                    local_moe = self._moe.execute(layer, hidden, shape, graph)
                    if local_moe is None:
                        raise WholeDecoderError("MoE returned no rank-local partial")
                    hidden = self._reducer.reduce(
                        layer, ReductionKind.MOE, local_moe, graph
                    )
                    if hidden is None:
                        raise WholeDecoderError("MoE PairReduce returned no hidden state")
            with self._observed("sample", "none", key, -1):
                tokens = self._sampler.sample(hidden, graph)
                if (
                    not isinstance(tokens, tuple)
                    or len(tokens) != key.sequences
                    or any(
                        isinstance(token, bool) or not isinstance(token, int)
                        for token in tokens
                    )
                ):
                    raise WholeDecoderError("output sampler contract changed")
        except BaseException:
            self._phase = ExecutorPhase.FAULTED
            raise
        self._phase = ExecutorPhase.IDLE
        self._last_prefix = prefix
        return tokens

    def _validate_prefix(self, prefix: DecoderContinuation) -> None:
        try:
            current = self._prefix_source.continuation
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            raise WholeDecoderError("accepted-prefix source failed") from exc
        if (
            not isinstance(prefix, DecoderContinuation)
            or prefix is not current
            or prefix is self._last_prefix
            or isinstance(prefix.token_count, bool)
            or not 0 <= prefix.token_count <= 262_144
            or isinstance(prefix.generation_epoch, bool)
            or prefix.generation_epoch <= 0
            or not _HEX_256.fullmatch(prefix.token_hash)
            or not _HEX_256.fullmatch(prefix.commit_sha256)
            or not _HEX_256.fullmatch(prefix.policy_digest)
        ):
            raise WholeDecoderError("exact published accepted prefix is required")

    @contextmanager
    def _observed(
        self, phase: str, attention: str, key: GraphKey, layer: int
    ) -> Iterator[None]:
        depth = f"k{int(key.depth)}" if isinstance(key, GraphKey) else "k0"
        graph_batch = key.graph_batch if isinstance(key, GraphKey) else 1
        with self._tracer.start_as_current_span("rocket.qwen38.whole_decoder") as span:
            span.set_attribute("phase", phase)
            span.set_attribute("attention", attention)
            span.set_attribute("depth", depth)
            span.set_attribute("graph_batch", graph_batch)
            span.set_attribute("draft_architecture", key.draft_architecture.value)
            span.set_attribute("rank", self._slabs.rank)
            span.set_attribute("layer", layer)
            try:
                yield
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise
            else:
                span.set_attribute("outcome", "success")


@dataclass(frozen=True)
class ColdLoadedDecoder:
    """One immutable loader publication and derived exact decoder slab binding."""

    slabs: DecoderSlabs
    loaded: LoadedRankSlabs


class WholeDecoderColdLoader:
    """Load one rank and emit comparable phase-duration/copy-byte OTEL metrics."""

    def __init__(self, loader: CudaRankSlabLoader, meter: OtelMeter):
        if loader is None or not callable(getattr(loader, "load", None)) or meter is None:
            raise WholeDecoderError("CUDA rank loader and OpenTelemetry meter are required")
        self._loader = loader
        self._duration = meter.create_histogram(
            "rocket.qwen38.whole_decoder.cold_load.phase.duration", unit="ns"
        )
        self._bytes = meter.create_counter(
            "rocket.qwen38.whole_decoder.cold_load.transfer", unit="By"
        )

    def load(self) -> ColdLoadedDecoder:
        """Consume one loader result, record its immutable receipt, and return it."""

        try:
            loaded = self._loader.load()
            slabs = DecoderSlabs.from_loaded(loaded)
            try:
                self._record(loaded.receipt)
            except Exception:
                # OTEL export cannot revoke an already published CUDA slab set.
                pass
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, WholeDecoderError):
                raise
            raise WholeDecoderError("whole-decoder cold load failed") from exc
        return ColdLoadedDecoder(slabs, loaded)

    def _record(self, receipt: RankLoadReceipt) -> None:
        rank = receipt.rank
        fixed = {"rank": rank, "slab.kind": "all"}
        self._duration.record(receipt.load_to_publish_ns, {**fixed, "phase": "total"})
        self._duration.record(receipt.allocation_ns, {**fixed, "phase": "allocation"})
        self._duration.record(receipt.publish_ns, {**fixed, "phase": "publish"})
        for slab in (receipt.target, receipt.mtp):
            kind = "target" if slab.key.endswith("-target") else "mtp"
            attributes = {"rank": rank, "slab.kind": kind}
            self._bytes.add(
                slab.bytes_read, {**attributes, "direction": "direct_read"}
            )
            self._bytes.add(slab.h2d_bytes, {**attributes, "direction": "h2d"})
            for phase in ("direct_read", "sha256", "h2d_fence"):
                duration = sum(getattr(chunk, f"{phase}_ns") for chunk in slab.chunks)
                self._duration.record(duration, {**attributes, "phase": phase})


def _attention_for(layer: int) -> AttentionKind:
    """Return the frozen three-GDN then one-QSA 48-layer model topology."""

    if isinstance(layer, bool) or not 0 <= layer < LAYERS:
        raise WholeDecoderError("decoder layer must be 0 through 47")
    return AttentionKind.QSA if layer % 4 == 3 else AttentionKind.GDN


LAYER_TOPOLOGY: Mapping[int, AttentionKind] = MappingProxyType(
    {layer: _attention_for(layer) for layer in range(LAYERS)}
)


__all__ = [
    "AcceptedPrefixSource",
    "AttentionKind",
    "AttentionExecutor",
    "BoundDecoderGraph",
    "ColdLoadedDecoder",
    "DecoderSlabs",
    "DecoderGraphFactory",
    "DraftArchitecture",
    "ExecutorPhase",
    "GraphKey",
    "HiddenReducer",
    "InputEmbedder",
    "LAYER_TOPOLOGY",
    "LayerOutput",
    "MoeExecutor",
    "OutputSampler",
    "ReductionKind",
    "WholeDecoderColdLoader",
    "WholeDecoderError",
    "WholeDecoderExecutor",
]
