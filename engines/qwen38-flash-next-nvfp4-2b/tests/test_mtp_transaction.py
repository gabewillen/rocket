# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from dataclasses import replace

from qwen38_slab.controller_restore import DecoderContinuation
from qwen38_slab.decode import Depth
from qwen38_slab.mtp_transaction import (
    DraftBatchSnapshot,
    DraftSequenceState,
    DraftStateLedger,
    DraftTransactionError,
    DraftTransactionPhase,
)


def prefix(epoch: int, tokens: int) -> DecoderContinuation:
    return DecoderContinuation(tokens, "a" * 64, epoch, "b" * 64, "c" * 64)


class PrefixSource:
    continuation: DecoderContinuation | None

    def __init__(self, value: DecoderContinuation): self.continuation = value


class Span:
    def __init__(self): self.attributes = {}
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def set_attribute(self, key, value): self.attributes[key] = value
    def record_exception(self, _error): pass


class Tracer:
    def __init__(self): self.spans = []
    def start_as_current_span(self, _name):
        span = Span(); self.spans.append(span); return span


class DraftStateTransactionTests(unittest.TestCase):
    def setUp(self):
        self.source = PrefixSource(prefix(3, 100))
        self.tracer = Tracer()
        self.ledger = DraftStateLedger(self.source, self.tracer)
        self.states = (
            DraftSequenceState(2, 17, 17, 9),
            DraftSequenceState(7, 41, 40, 4),
        )

    def test_accepts_only_the_current_prefix_and_publishes_selected_snapshots(self):
        snapshot = self.ledger.prepare(
            prefix=self.source.continuation,
            sequences=self.states,
            depth=Depth.K3,
            proposal_tokens=((11, 12, 13), (21, 22, 23)),
            native_generation=10,
        )
        self.assertIsInstance(snapshot, DraftBatchSnapshot)
        self.assertEqual(self.ledger.phase, DraftTransactionPhase.PENDING)

        next_prefix = prefix(4, 104)
        self.source.continuation = next_prefix
        publication = self.ledger.commit(
            snapshot,
            accepted_widths=(3, 1),
            published_mtp_tokens=(20, 41),
            prefix=next_prefix,
        )
        self.assertEqual(self.ledger.phase, DraftTransactionPhase.IDLE)
        self.assertIs(publication.prefix, next_prefix)
        self.assertEqual(
            publication.sequences,
            (
                DraftSequenceState(2, 20, 20, 10),
                DraftSequenceState(7, 42, 41, 10),
            ),
        )
        self.assertEqual(publication.accepted_proposals, ((11, 12), ()))

    def test_unaccepted_suffix_never_reaches_publication(self):
        snapshot = self.ledger.prepare(
            prefix=self.source.continuation,
            sequences=self.states,
            depth=Depth.K2,
            proposal_tokens=((11, 12), (21, 22)),
            native_generation=10,
        )
        next_prefix = prefix(4, 103)
        self.source.continuation = next_prefix
        publication = self.ledger.commit(
            snapshot,
            accepted_widths=(1, 2),
            published_mtp_tokens=(18, 42),
            prefix=next_prefix,
        )
        self.assertEqual(publication.accepted_proposals, ((), (21,)))
        self.assertNotIn(12, publication.accepted_proposals[0])
        self.assertNotIn(22, publication.accepted_proposals[1])

    def test_stale_duplicate_and_forged_publications_fail_closed(self):
        snapshot = self.ledger.prepare(
            prefix=self.source.continuation,
            sequences=self.states,
            depth=Depth.K1,
            proposal_tokens=((11,), (21,)),
            native_generation=10,
        )
        with self.assertRaisesRegex(DraftTransactionError, "new accepted prefix"):
            self.ledger.commit(
                snapshot,
                accepted_widths=(1, 1),
                published_mtp_tokens=(18, 42),
                prefix=self.source.continuation,
            )
        self.assertEqual(self.ledger.phase, DraftTransactionPhase.PENDING)

        next_prefix = prefix(4, 102)
        self.source.continuation = next_prefix
        forged = replace(snapshot, native_generation=11)
        with self.assertRaisesRegex(DraftTransactionError, "pending snapshot"):
            self.ledger.commit(
                forged,
                accepted_widths=(1, 1),
                published_mtp_tokens=(18, 42),
                prefix=next_prefix,
            )
        self.ledger.commit(
            snapshot,
            accepted_widths=(1, 1),
            published_mtp_tokens=(18, 41),
            prefix=next_prefix,
        )
        with self.assertRaisesRegex(DraftTransactionError, "pending"):
            self.ledger.commit(
                snapshot,
                accepted_widths=(1, 1),
                published_mtp_tokens=(18, 42),
                prefix=next_prefix,
            )

    def test_invalid_native_publication_faults_ledger_and_blocks_reuse(self):
        snapshot = self.ledger.prepare(
            prefix=self.source.continuation,
            sequences=self.states,
            depth=Depth.K3,
            proposal_tokens=((11, 12, 13), (21, 22, 23)),
            native_generation=10,
        )
        next_prefix = prefix(4, 102)
        self.source.continuation = next_prefix
        with self.assertRaisesRegex(DraftTransactionError, "MTP causal boundary"):
            self.ledger.commit(
                snapshot,
                accepted_widths=(1, 1),
                published_mtp_tokens=(99, 42),
                prefix=next_prefix,
            )
        self.assertEqual(self.ledger.phase, DraftTransactionPhase.FAULTED)
        with self.assertRaisesRegex(DraftTransactionError, "faulted"):
            self.ledger.prepare(
                prefix=next_prefix,
                sequences=self.states,
                depth=Depth.K1,
                proposal_tokens=((1,), (2,)),
                native_generation=12,
            )

    def test_prepare_validation_has_no_state_effect(self):
        with self.assertRaisesRegex(DraftTransactionError, "proposal width"):
            self.ledger.prepare(
                prefix=self.source.continuation,
                sequences=self.states,
                depth=Depth.K2,
                proposal_tokens=((1,), (2,)),
                native_generation=10,
            )
        self.assertEqual(self.ledger.phase, DraftTransactionPhase.IDLE)
        self.assertEqual(
            set(self.tracer.spans[-1].attributes),
            {"operation", "depth", "batch", "outcome"},
        )
        self.assertEqual(self.tracer.spans[-1].attributes["outcome"], "failure")


if __name__ == "__main__":
    unittest.main()
