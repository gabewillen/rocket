from __future__ import annotations

import unittest

from qwen38_slab.decode import Depth, DepthZeroDecodeExecutor, StreamStep
from qwen38_slab.full_attention_layer import (
    FullAttentionLayerError,
    FullAttentionLayerExecutor,
)


class Span:
    def __init__(self): self.attributes = {}
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, exception): self.exception = type(exception).__name__


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, name):
        span = Span(); span.name = name; self.spans.append(span); return span


class Runtime:
    stream_pointer = 0x1000
    projected_attention_pointer = 0x2000
    rank_slab_identity = ("slab-a", 0, 3)
    def __init__(self, calls): self.calls = calls
    def bind_qkv_activation_device(self, source, rows):
        self.calls.append("activation_bind")
        self.bound = (source, rows)
    def read_qsa_projected_output(self): return b"a" * (16 * 2560 * 2)


class Binding:
    def __init__(self, calls): self.calls = calls
    def upload_and_launch(self, prepared):
        self.calls.append("qsa_attention")
        self.prepared = prepared


class HyperConnection:
    pointers = {"block_input": 0x3000, "reduced": 0x4000}
    rank_slab_identity = ("slab-a", 0, 3)
    def __init__(self, calls): self.calls = calls
    def launch_mix(self, m): self.calls.append(("attn_hc_mix", m))
    def reference_mix(self, m): self.calls.append(("attn_hc_mix_reference", m))
    def launch_combine(self, m): self.calls.append(("mlp_hc_combine_mix", m))
    def reference_combine(self, m): self.calls.append(("mlp_hc_combine_mix_reference", m))
    def synchronize(self): pass
    def hashes(self, m, names=None):
        del m
        hashes = {
            "block_input": "block", "injection": "injection",
            "reduced": "reduced", "updated_hidden": "updated",
            "next_block_input": "next_block", "next_injection": "next_injection",
        }
        return hashes if names is None else {name: hashes[name] for name in names}


class Reducer:
    def __init__(self, calls): self.fail = False; self.calls = calls
    def reduce(self, input_pointer, output_pointer, m, stream_pointer):
        if self.fail: raise FullAttentionLayerError("fault")
        self.calls.append("pair_reduce")
        self.call = (input_pointer, output_pointer, m, stream_pointer)


class FullAttentionLayerTests(unittest.TestCase):
    def setUp(self):
        self.tracer = Tracer()
        self.calls = []
        self.runtime = Runtime(self.calls)
        self.binding = Binding(self.calls)
        self.hc = HyperConnection(self.calls)
        self.reducer = Reducer(self.calls)
        self.layer = FullAttentionLayerExecutor(
            self.runtime, self.binding, self.hc, self.reducer, self.tracer
        )

    def test_exact_order_reference_hashes_and_bounded_stages(self):
        prepared = DepthZeroDecodeExecutor(Tracer()).prepare(
            [StreamStep(index, 64 + index, Depth.K0) for index in range(5)]
        )
        publication = self.layer.execute(prepared, verify_reference=True)
        self.assertEqual((publication.generation, publication.graph_batch), (1, 8))
        self.assertEqual(self.runtime.bound, (0x3000, 8))
        self.assertEqual(
            self.reducer.call, (0x2000, 0x4000, 8, 0x1000)
        )
        self.assertEqual(
            self.calls,
            [("attn_hc_mix", 8), "activation_bind", "qsa_attention",
             "pair_reduce", ("mlp_hc_combine_mix", 8),
             ("attn_hc_mix_reference", 8),
             ("mlp_hc_combine_mix_reference", 8)],
        )
        stages = {span.attributes["stage"] for span in self.tracer.spans}
        self.assertEqual(stages, set(FullAttentionLayerExecutor._STAGES))
        self.assertTrue(all(span.attributes["graph_batch"] == 8 for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["outcome"] == "success" for span in self.tracer.spans))

    def test_transport_failure_faults_without_publication(self):
        self.reducer.fail = True
        prepared = DepthZeroDecodeExecutor(Tracer()).prepare(
            [StreamStep(0, 64, Depth.K0)]
        )
        with self.assertRaisesRegex(FullAttentionLayerError, "fault"):
            self.layer.execute(prepared)
        with self.assertRaisesRegex(FullAttentionLayerError, "cannot retry"):
            self.layer.execute(prepared)
        self.assertEqual(self.tracer.spans[-1].attributes["outcome"], "failure")

    def test_mismatched_authenticated_slab_is_rejected(self):
        self.hc.rank_slab_identity = ("slab-b", 0, 3)
        with self.assertRaisesRegex(FullAttentionLayerError, "identity drift"):
            FullAttentionLayerExecutor(
                self.runtime, self.binding, self.hc, self.reducer, self.tracer
            )


if __name__ == "__main__":
    unittest.main()
