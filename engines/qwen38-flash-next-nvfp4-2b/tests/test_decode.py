from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from qwen38_slab.decode import (
    GRAPH_BATCHES,
    LAZY_DEPTHS,
    LAZY_MAX_STREAMS,
    MAX_CONTEXT_TOKENS,
    MAX_QUERY_ROWS,
    PAD,
    DecodeContractError,
    Depth,
    DepthBucket,
    DepthBucketPlanner,
    DepthZeroDecodeExecutor,
    QsaContinuationMetadata,
    StreamStep,
)


class Span:
    def __init__(self):
        self.attributes = {}
        self.exceptions = []

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exceptions.append(type(exception).__name__)


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


class DecodeContractTests(unittest.TestCase):
    def setUp(self):
        self.tracer = Tracer()

    def test_mixed_plan_is_immutable_stable_and_uses_smallest_graphs(self):
        planner = DepthBucketPlanner(Depth, self.tracer)
        inputs = [
            StreamStep(9, 100, Depth.K3),
            StreamStep(4, 200, Depth.K0),
            StreamStep(1, 300, Depth.K3),
            StreamStep(7, 400, Depth.K1),
        ]
        plan = planner.plan(inputs)
        inputs.reverse()
        self.assertEqual(
            [bucket.depth for bucket in plan.buckets],
            [Depth.K0, Depth.K1, Depth.K3],
        )
        self.assertEqual([bucket.graph_batch for bucket in plan.buckets], [1, 1, 2])
        self.assertEqual([step.slot for step in plan.buckets[-1].streams], [1, 9])
        self.assertEqual(plan.active_streams, 4)
        with self.assertRaises(FrozenInstanceError):
            plan.buckets[-1].graph_batch = 16

    def test_k0_through_k7_are_fixed_and_deep_depths_are_lazy_c2_only(self):
        self.assertEqual(tuple(int(depth) for depth in Depth), tuple(range(8)))
        self.assertEqual(LAZY_DEPTHS, frozenset(Depth(value) for value in range(4, 8)))
        self.assertEqual(LAZY_MAX_STREAMS, 2)
        planner = DepthBucketPlanner(Depth, self.tracer)
        deep = planner.plan(
            [StreamStep(1, 100, Depth.K7), StreamStep(4, 200, Depth.K7)]
        )
        bucket = deep.buckets[0]
        self.assertEqual((bucket.query_width, bucket.actual_rows, bucket.graph_rows), (8, 16, 16))
        with self.assertRaisesRegex(DecodeContractError, "lazy-only"):
            planner.plan(
                [
                    StreamStep(0, 1, Depth.K7),
                    StreamStep(1, 1, Depth.K0),
                    StreamStep(2, 1, Depth.K0),
                ]
            )
        hot = planner.plan(
            [StreamStep(slot, 1, Depth.K0) for slot in range(16)]
        )
        self.assertEqual((hot.buckets[0].graph_batch, hot.buckets[0].graph_rows), (16, 16))
        self.assertEqual(MAX_QUERY_ROWS, 128)

    def test_k7_metadata_carries_all_eight_continuation_positions(self):
        bucket = DepthBucketPlanner(Depth, self.tracer).plan(
            [StreamStep(3, 7, Depth.K7)]
        ).buckets[0]
        metadata = QsaContinuationMetadata(self.tracer)
        lease = metadata.update(bucket)
        self.assertEqual((lease.actual_rows, lease.graph_rows), (8, 8))
        self.assertEqual(list(metadata.buffers.logical_positions[:8]), list(range(7, 15)))
        self.assertEqual(list(metadata.buffers.token_to_req[:8]), [0] * 8)

    def test_depth_zero_executor_rejects_mtp_residency_and_speculation(self):
        with self.assertRaisesRegex(DecodeContractError, "resident MTP"):
            DepthZeroDecodeExecutor(self.tracer, mtp_resident=True)
        executor = DepthZeroDecodeExecutor(self.tracer)
        with self.assertRaisesRegex(DecodeContractError, "not resident"):
            executor.prepare([StreamStep(0, 10, Depth.K1)])
        prepared = executor.prepare([StreamStep(0, 10, Depth.K0)])
        self.assertEqual(prepared.lease.depth, Depth.K0)
        self.assertEqual(prepared.lease.actual_rows, 1)

    def test_qsa_positions_match_accepted_continuations_and_compression_boundaries(self):
        bucket = DepthBucketPlanner(Depth, self.tracer).plan([
            StreamStep(2, 3, Depth.K3),
            StreamStep(5, 8, Depth.K3),
        ]).buckets[0]
        metadata = QsaContinuationMetadata(self.tracer)
        lease = metadata.update(bucket)
        buffers = metadata.buffers
        self.assertEqual(
            (lease.actual_batch, lease.graph_batch, lease.actual_rows), (2, 2, 8)
        )
        self.assertEqual(list(buffers.query_start_loc[:3]), [0, 4, 8])
        self.assertEqual(list(buffers.seq_lens[:2]), [7, 12])
        self.assertEqual(list(buffers.stream_slots[:2]), [2, 5])
        self.assertEqual(
            list(buffers.token_to_req[:8]), [0, 0, 0, 0, 1, 1, 1, 1]
        )
        self.assertEqual(
            list(buffers.logical_positions[:8]), [3, 4, 5, 6, 8, 9, 10, 11]
        )
        self.assertEqual(
            list(buffers.raw_ring_offsets[:8]), [3, 4, 5, 6, 0, 1, 2, 3]
        )
        self.assertEqual(
            list(buffers.compressed_positions[:8]),
            [0, PAD, PAD, PAD, PAD, PAD, PAD, 2],
        )

    def test_metadata_reuses_views_and_clears_previous_wider_graph(self):
        planner = DepthBucketPlanner(Depth, self.tracer)
        metadata = QsaContinuationMetadata(self.tracer)
        buffers = metadata.buffers
        view_ids = tuple(
            id(getattr(buffers, name)) for name in buffers.__dataclass_fields__
        )
        wide = planner.plan(
            [StreamStep(slot, 20, Depth.K3) for slot in range(9)]
        ).buckets[0]
        first = metadata.update(wide)
        narrow = planner.plan([StreamStep(3, 50, Depth.K0)]).buckets[0]
        second = metadata.update(narrow)
        self.assertEqual(second.generation, first.generation + 1)
        self.assertIs(metadata.buffers, buffers)
        self.assertEqual(
            tuple(
                id(getattr(metadata.buffers, name))
                for name in buffers.__dataclass_fields__
            ),
            view_ids,
        )
        self.assertEqual(list(buffers.query_start_loc[:2]), [0, 1])
        self.assertTrue(all(value == PAD for value in buffers.logical_positions[1:]))
        self.assertTrue(all(value == PAD for value in buffers.stream_slots[1:]))

    def test_invalid_bucket_does_not_mutate_metadata(self):
        metadata = QsaContinuationMetadata(self.tracer)
        valid = DepthBucketPlanner(Depth, self.tracer).plan(
            [StreamStep(1, 10, Depth.K0)]
        ).buckets[0]
        metadata.update(valid)
        before = (
            metadata.generation,
            bytes(metadata.buffers.query_start_loc),
            bytes(metadata.buffers.logical_positions),
        )
        malformed = DepthBucket(Depth.K3, 16, (StreamStep(4, 10, Depth.K3),))
        with self.assertRaisesRegex(DecodeContractError, "smallest captured graph"):
            metadata.update(malformed)
        self.assertEqual(
            before,
            (
                metadata.generation,
                bytes(metadata.buffers.query_start_loc),
                bytes(metadata.buffers.logical_positions),
            ),
        )
        self.assertEqual(self.tracer.spans[-1].attributes["outcome"], "failure")
        malformed_depth = DepthBucket(99, 1, (StreamStep(4, 10, 99),))
        with self.assertRaisesRegex(DecodeContractError, "depth must be K0 through K7"):
            metadata.update(malformed_depth)

    def test_stream_and_context_bounds_fail_closed(self):
        planner = DepthBucketPlanner(Depth, self.tracer)
        invalid = (
            [],
            [StreamStep(0, 1, Depth.K0), StreamStep(0, 2, Depth.K0)],
            [StreamStep(16, 1, Depth.K0)],
            [StreamStep(0, 1, True)],
            [StreamStep(0, MAX_CONTEXT_TOKENS, Depth.K0)],
            [StreamStep(slot, 1, Depth.K0) for slot in range(16)]
            + [StreamStep(0, 1, Depth.K0)],
        )
        for streams in invalid:
            with self.subTest(streams=streams), self.assertRaises(DecodeContractError):
                planner.plan(streams)
        full = planner.plan(
            [
                StreamStep(slot, MAX_CONTEXT_TOKENS - 1, Depth.K0)
                for slot in range(16)
            ]
        )
        self.assertEqual(full.buckets[0].graph_batch, GRAPH_BATCHES[-1])
        with self.assertRaises(DecodeContractError):
            DepthBucketPlanner((False,), self.tracer)

    def test_otel_dimensions_are_bounded_and_failures_are_observed(self):
        planner = DepthBucketPlanner(Depth, self.tracer)
        planner.plan([StreamStep(0, 0, Depth.K0), StreamStep(1, 4, Depth.K1)])
        with self.assertRaises(DecodeContractError):
            planner.plan([])
        metadata = QsaContinuationMetadata(self.tracer)
        bucket = DepthBucketPlanner(Depth, self.tracer).plan(
            [StreamStep(0, 0, Depth.K0)]
        ).buckets[0]
        metadata.update(bucket)
        allowed = {"phase", "depth", "graph_batch", "outcome"}
        self.assertTrue(all(set(span.attributes) == allowed for span in self.tracer.spans))
        self.assertTrue(
            all(
                span.attributes["phase"] in {"plan", "metadata"}
                for span in self.tracer.spans
            )
        )
        self.assertTrue(
            all(
                span.attributes["depth"]
                in {"k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "mixed"}
                for span in self.tracer.spans
            )
        )
        self.assertTrue(
            all(
                span.attributes["graph_batch"] in {*GRAPH_BATCHES, "mixed"}
                for span in self.tracer.spans
            )
        )
        self.assertEqual(
            {span.attributes["outcome"] for span in self.tracer.spans},
            {"success", "failure"},
        )


if __name__ == "__main__":
    unittest.main()
