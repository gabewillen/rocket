from __future__ import annotations

import hashlib
import unittest
from dataclasses import dataclass
from types import MappingProxyType

from qwen38_slab.controller_restore import DecoderContinuation
from qwen38_slab.cuda_slab_loader import (
    ChunkTransferReceipt,
    LoadedRankSlabs,
    RankLoadReceipt,
    SlabTransferReceipt,
)
from qwen38_slab.decode import Depth
from qwen38_slab.mtp_policy import (
    AdaptiveMtpPolicy,
    PolicyEvent,
    SessionPhase,
    matched_live_policy_config,
)
from qwen38_slab.routed_moe import MoeShape, SLAB_ARTIFACT_KEY
from qwen38_slab.whole_decoder import (
    AttentionKind,
    DecoderSlabs,
    DraftArchitecture,
    ExecutorPhase,
    GraphKey,
    LayerOutput,
    ReductionKind,
    WholeDecoderColdLoader,
    WholeDecoderError,
    WholeDecoderExecutor,
)


class _Span:
    def __init__(self, name: str, sink: list[tuple[str, dict[str, object]]]):
        self.name = name
        self.attributes: dict[str, object] = {}
        self._sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._sink.append((self.name, dict(self.attributes)))

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        pass


class _Tracer:
    def __init__(self):
        self.spans: list[tuple[str, dict[str, object]]] = []

    def start_as_current_span(self, name: str) -> _Span:
        return _Span(name, self.spans)


class _Metric:
    def __init__(self, name: str, sink: list[tuple[str, int, dict[str, object]]]):
        self.name = name
        self.sink = sink

    def add(self, amount: int, attributes: dict[str, object]) -> None:
        self.sink.append((self.name, amount, dict(attributes)))

    def record(self, amount: int, attributes: dict[str, object]) -> None:
        self.sink.append((self.name, amount, dict(attributes)))


class _Meter:
    def __init__(self):
        self.points: list[tuple[str, int, dict[str, object]]] = []

    def create_counter(self, name: str, *, unit: str) -> _Metric:
        return _Metric(name, self.points)

    def create_histogram(self, name: str, *, unit: str) -> _Metric:
        return _Metric(name, self.points)


@dataclass(frozen=True)
class _Graph:
    key: GraphKey
    target_slab: object
    mtp_slab: object | None
    draft_slab: object | None


class _GraphFactory:
    def __init__(self, events: list[tuple[object, ...]]):
        self.events = events

    def bind(
        self, key: GraphKey, target_slab: object, mtp_slab: object | None,
        draft_slab: object | None,
    ) -> _Graph:
        self.events.append(
            ("bind", key.depth, key.graph_batch, target_slab, mtp_slab, draft_slab)
        )
        return _Graph(key, target_slab, mtp_slab, draft_slab)


class _PrefixSource:
    def __init__(self, continuation: DecoderContinuation):
        self.continuation = continuation


class _Loader:
    def __init__(self, loaded: LoadedRankSlabs):
        self.loaded = loaded

    def load(self) -> LoadedRankSlabs:
        return self.loaded


class _Attention:
    def __init__(self, kind: AttentionKind, events: list[tuple[object, ...]]):
        self.kind = kind
        self.events = events

    def execute(self, layer: int, hidden: object, graph: _Graph) -> LayerOutput:
        self.events.append((self.kind.value, layer, hidden, graph.key.depth))
        return LayerOutput(("attention", layer), ("attention_partial", layer))


class _FailingAttention(_Attention):
    def execute(self, layer: int, hidden: object, graph: _Graph) -> LayerOutput:
        raise RuntimeError("injected device failure")


class _Embedder:
    def __init__(self, events: list[tuple[object, ...]]):
        self.events = events

    def embed(self, token_ids: tuple[int, ...], graph: _Graph) -> object:
        self.events.append(("embed", token_ids, graph.key.depth))
        return ("embedding", token_ids)


class _Moe:
    def __init__(self, events: list[tuple[object, ...]]):
        self.events = events

    def execute(self, layer: int, hidden: object, shape: MoeShape, graph: _Graph) -> object:
        self.events.append(("moe", layer, hidden, shape.token_rows, graph.key.depth))
        return ("moe_partial", layer)


class _Reducer:
    def __init__(self, events: list[tuple[object, ...]]):
        self.events = events

    def reduce(self, layer: int, kind: ReductionKind, partial: object, graph: _Graph) -> object:
        self.events.append(("reduce", layer, kind.value, partial, graph.key.depth))
        return ("reduced", layer, kind.value)


class _Sampler:
    def __init__(self, events: list[tuple[object, ...]]):
        self.events = events

    def sample(self, hidden: object, graph: _Graph) -> tuple[int, ...]:
        self.events.append(("sample", hidden, graph.key.depth))
        return tuple(range(graph.key.sequences))


class WholeDecoderTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.tracer = _Tracer()
        self.continuation = DecoderContinuation(
            41,
            hashlib.sha256(b"accepted-prefix").hexdigest(),
            7,
            hashlib.sha256(b"commit").hexdigest(),
            hashlib.sha256(b"policy").hexdigest(),
        )
        self.prefixes = _PrefixSource(self.continuation)
        self.target = object()
        self.mtp = object()
        self.draft = object()
        self.slabs = DecoderSlabs(
            revision="fc694b54fb0174e0913e6adf86691ef85a4ead47",
            artifact_key=SLAB_ARTIFACT_KEY,
            rank=0,
            target=self.target,
            mtp=self.mtp,
            draft_head=self.draft,
        )
        self.executor = WholeDecoderExecutor(
            slabs=self.slabs,
            prefix_source=self.prefixes,
            graph_factory=_GraphFactory(self.events),
            embedder=_Embedder(self.events),
            gdn=_Attention(AttentionKind.GDN, self.events),
            qsa=_Attention(AttentionKind.QSA, self.events),
            moe=_Moe(self.events),
            reducer=_Reducer(self.events),
            sampler=_Sampler(self.events),
            tracer=self.tracer,
        )

    def test_exact_48_layer_topology_gate_lazy_graphs_and_sampling(self) -> None:
        # One lookup per depth proves the complete K0-K7 graph-key contract.
        for depth in Depth:
            graph = self.executor.graph_for(GraphKey(depth, 1))
            self.assertIs(graph.target_slab, self.target)
            self.assertIs(graph.mtp_slab, None if depth is Depth.K0 else self.mtp)
            self.assertIs(self.executor.graph_for(GraphKey(depth, 1)), graph)
        self.assertEqual(sum(event[0] == "bind" for event in self.events), 8)
        hybrid = self.executor.graph_for(
            GraphKey(
                Depth.K7, 1,
                draft_architecture=DraftArchitecture.EXTERNAL_DRAFT,
            )
        )
        self.assertIs(hybrid.target_slab, self.target)
        self.assertIs(hybrid.mtp_slab, self.mtp)
        self.assertIs(hybrid.draft_slab, self.draft)
        self.assertEqual(sum(event[0] == "bind" for event in self.events), 9)

        self.events.clear()
        policy = AdaptiveMtpPolicy(self.tracer, matched_live_policy_config())
        decision = policy.decide(
            policy.initial_state(),
            PolicyEvent(SessionPhase.EARLY_DECODE, 1, 100, Depth.K0),
        )
        self.tracer.spans.clear()
        tokens = self.executor.execute(
            prefix=self.continuation,
            decision=decision,
            token_ids=(17,),
        )
        self.assertEqual(tokens, (0,))

        attention = [event for event in self.events if event[0] in ("gdn", "qsa")]
        self.assertEqual(len(attention), 48)
        self.assertEqual(sum(event[0] == "gdn" for event in attention), 36)
        self.assertEqual(sum(event[0] == "qsa" for event in attention), 12)
        self.assertEqual(
            [event[0] for event in attention],
            ["qsa" if layer % 4 == 3 else "gdn" for layer in range(48)],
        )
        self.assertEqual(sum(event[0] == "moe" for event in self.events), 48)
        reductions = [event for event in self.events if event[0] == "reduce"]
        self.assertEqual(len(reductions), 96)
        self.assertEqual(
            [(event[1], event[2]) for event in reductions],
            [
                (layer, kind)
                for layer in range(48)
                for kind in ("attention", "moe")
            ],
        )
        for layer in range(48):
            group = self.events[1 + layer * 4 : 1 + layer * 4 + 4]
            expected_attention = "qsa" if layer % 4 == 3 else "gdn"
            self.assertEqual(
                [(event[0], event[1]) for event in group],
                [
                    (expected_attention, layer),
                    ("reduce", layer),
                    ("moe", layer),
                    ("reduce", layer),
                ],
            )
        self.assertEqual(self.events[-1][0], "sample")

        # The publication object is a one-step capability. Reuse and copied data fail.
        before = len(self.events)
        with self.assertRaisesRegex(WholeDecoderError, "published accepted prefix"):
            self.executor.execute(
                prefix=self.continuation,
                decision=decision,
                token_ids=(17,),
            )
        self.assertEqual(len(self.events), before)
        copied = DecoderContinuation(**self.continuation.__dict__)
        with self.assertRaisesRegex(WholeDecoderError, "published accepted prefix"):
            self.executor.execute(
                prefix=copied,
                decision=decision,
                token_ids=(17,),
            )
        self.assertEqual(len(self.events), before)

        with self.assertRaisesRegex(WholeDecoderError, "at most 128"):
            GraphKey(Depth.K7, 16, verify_width=9)
        for depth in (Depth.K4, Depth.K5, Depth.K6, Depth.K7):
            for sequences in (8, 16):
                with self.subTest(depth=depth, sequences=sequences), self.assertRaisesRegex(
                    WholeDecoderError, "low-concurrency"
                ):
                    GraphKey(depth, sequences)

        allowed_keys = {
            "phase", "attention", "depth", "graph_batch", "draft_architecture",
            "rank", "layer", "outcome"
        }
        allowed_values = {
            "phase": {"validate", "embedding", "layers", "sample"},
            "attention": {"none", "gdn", "qsa"},
            "depth": {f"k{depth}" for depth in range(8)},
            "graph_batch": {1, 2, 4, 8, 16},
            "draft_architecture": {"native_mtp", "external_draft"},
            "rank": {0, 1},
            "layer": set(range(-1, 48)),
            "outcome": {"success", "failure"},
        }
        for name, attributes in self.tracer.spans:
            self.assertEqual(name, "rocket.qwen38.whole_decoder")
            self.assertLessEqual(set(attributes), allowed_keys)
            for key, value in attributes.items():
                self.assertIn(value, allowed_values[key])

    def test_cold_load_records_phase_duration_and_copy_bytes(self) -> None:
        target_chunk = ChunkTransferReceipt(0, 64, 11, 13, 17)
        mtp_chunk = ChunkTransferReceipt(0, 32, 19, 23, 29)
        target = SlabTransferReceipt(
            "rank0-target", 64, 64, 1, 1, 100, 200, (target_chunk,)
        )
        mtp = SlabTransferReceipt(
            "rank0-mtp", 32, 32, 1, 1, 110, 210, (mtp_chunk,)
        )
        receipt = RankLoadReceipt(0, target, mtp, 7, 5, 127, 90)
        loaded = LoadedRankSlabs(
            MappingProxyType({"rank0-target": self.target, "rank0-mtp": self.mtp}),
            receipt,
        )
        meter = _Meter()

        result = WholeDecoderColdLoader(_Loader(loaded), meter).load()

        self.assertIs(result.slabs.target, self.target)
        self.assertIs(result.slabs.mtp, self.mtp)
        durations = [point for point in meter.points if point[0].endswith("duration")]
        transfers = [point for point in meter.points if point[0].endswith("transfer")]
        self.assertEqual(
            {(point[2]["phase"], point[2]["slab.kind"]) for point in durations},
            {
                ("total", "all"),
                ("allocation", "all"),
                ("publish", "all"),
                ("direct_read", "target"),
                ("sha256", "target"),
                ("h2d_fence", "target"),
                ("direct_read", "mtp"),
                ("sha256", "mtp"),
                ("h2d_fence", "mtp"),
            },
        )
        self.assertEqual(
            [(point[1], point[2]["slab.kind"], point[2]["direction"]) for point in transfers],
            [
                (64, "target", "direct_read"),
                (64, "target", "h2d"),
                (32, "mtp", "direct_read"),
                (32, "mtp", "h2d"),
            ],
        )
        for _, _, attributes in meter.points:
            self.assertEqual(attributes["rank"], 0)
            self.assertNotIn("path", attributes)
            self.assertNotIn("digest", attributes)

    def test_active_adapter_failure_faults_without_sampling(self) -> None:
        events: list[tuple[object, ...]] = []
        executor = WholeDecoderExecutor(
            slabs=self.slabs,
            prefix_source=self.prefixes,
            graph_factory=_GraphFactory(events),
            embedder=_Embedder(events),
            gdn=_FailingAttention(AttentionKind.GDN, events),
            qsa=_Attention(AttentionKind.QSA, events),
            moe=_Moe(events),
            reducer=_Reducer(events),
            sampler=_Sampler(events),
            tracer=self.tracer,
        )
        policy = AdaptiveMtpPolicy(self.tracer, matched_live_policy_config())
        decision = policy.decide(
            policy.initial_state(),
            PolicyEvent(SessionPhase.EARLY_DECODE, 1, 100, Depth.K0),
        )

        with self.assertRaisesRegex(RuntimeError, "injected device failure"):
            executor.execute(
                prefix=self.continuation,
                decision=decision,
                token_ids=(17,),
            )

        self.assertIs(executor.phase, ExecutorPhase.FAULTED)
        self.assertFalse(any(event[0] == "sample" for event in events))
        with self.assertRaisesRegex(WholeDecoderError, "faulted"):
            executor.execute(
                prefix=self.continuation,
                decision=decision,
                token_ids=(17,),
            )


if __name__ == "__main__":
    unittest.main()
