from __future__ import annotations

import unittest

from qwen38_slab.k0_composition import (
    K0CompositionError,
    K0CompositionRoot,
    K0_DOMAIN,
    REQUIRED_K0_PARTICIPANTS,
)
from qwen38_slab.routed_moe import SLAB_ARTIFACT_KEY
from qwen38_slab.whole_decoder import AttentionKind, DecoderSlabs


class _Span:
    def __init__(self, sink):
        self.attributes = {}
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.sink.append(self.attributes)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exception):
        pass


class _Tracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name):
        return _Span(self.spans)


class _Participant:
    execution_domain = K0_DOMAIN

    def __init__(self, kind=None):
        self.kind = kind

    def bind(self):
        pass

    def embed(self):
        pass

    def execute(self):
        pass

    def reduce(self):
        pass

    def sample(self):
        pass

    def encode_one(self):
        pass

    def compare(self):
        pass


def _slab(rank: int) -> DecoderSlabs:
    return DecoderSlabs(
        revision="fc694b54fb0174e0913e6adf86691ef85a4ead47",
        artifact_key=SLAB_ARTIFACT_KEY,
        rank=rank,
        target=object(),
        mtp=object(),
    )


def _complete_participants():
    result = {name: _Participant() for name in REQUIRED_K0_PARTICIPANTS}
    result["rank0.target_slab"] = _slab(0)
    result["rank1.target_slab"] = _slab(1)
    for name in result:
        if ".gdn.layer" in name:
            result[name] = _Participant(AttentionKind.GDN)
        elif ".qsa.layer" in name:
            result[name] = _Participant(AttentionKind.QSA)
    return result


class K0CompositionContractTests(unittest.TestCase):
    def test_empty_composition_names_every_missing_production_participant(self):
        tracer = _Tracer()
        with self.assertRaisesRegex(
            K0CompositionError, "missing K0 participant: tokenizer"
        ) as raised:
            K0CompositionRoot({}, tracer)
        self.assertEqual(raised.exception.missing, REQUIRED_K0_PARTICIPANTS)
        self.assertEqual(len(REQUIRED_K0_PARTICIPANTS), 394)
        self.assertEqual(tracer.spans[-1]["outcome"], "failure")
        self.assertEqual(tracer.spans[-1]["missing.bucket"], "65-394")

    def test_structurally_complete_contract_publishes_immutable_inventory(self):
        # This proves only the composition gate. It makes no kernel, numerical,
        # or physical-production claim for these test participants.
        participants = _complete_participants()
        root = K0CompositionRoot(participants, _Tracer())
        self.assertEqual(tuple(root.binding.participants), REQUIRED_K0_PARTICIPANTS)
        self.assertEqual(len(root.binding.layer_plan), 96)
        self.assertEqual(
            (root.binding.layer_plan[3].rank, root.binding.layer_plan[3].layer,
             root.binding.layer_plan[3].kind),
            (0, 3, AttentionKind.QSA),
        )
        self.assertIs(
            root.binding.layer_plan[3].attention,
            root.binding.participants["rank0.qsa.layer3"],
        )
        self.assertEqual(
            (root.binding.layer_plan[-1].rank, root.binding.layer_plan[-1].layer),
            (1, 47),
        )
        participants.pop("tokenizer")
        self.assertIn("tokenizer", root.binding.participants)
        with self.assertRaises(TypeError):
            root.binding.participants["tokenizer"] = _Participant()

    def test_mtp_or_wrong_attention_domain_cannot_open_k0_gate(self):
        participants = _complete_participants()
        participants["rank0.moe.layer0"] = _Participant()
        participants["rank0.moe.layer0"].execution_domain = "mtp_draft"
        with self.assertRaisesRegex(
            K0CompositionError,
            "K0 participant domain changed: rank0.moe.layer0",
        ):
            K0CompositionRoot(participants, _Tracer())

        participants = _complete_participants()
        participants["rank1.qsa.layer47"] = _Participant(AttentionKind.GDN)
        with self.assertRaisesRegex(
            K0CompositionError,
            "QSA participant kind changed: rank1.qsa.layer47",
        ):
            K0CompositionRoot(participants, _Tracer())


if __name__ == "__main__":
    unittest.main()
