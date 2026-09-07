from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qwen38_slab.state_txn import (
    PAGE_BYTES,
    STATE_FAMILIES,
    AcceptedBoundary,
    AuthenticatedState,
    FamilyPayload,
    StateIdentity,
    StateTransactionError,
    StateTransactionStore,
    Transition,
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


class StateTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.stores = (self.root / "rank0", self.root / "rank1")
        self.tracer = Tracer()
        self.identity = StateIdentity("fc694-synthetic", hashlib.sha256(b"precision").hexdigest())
        self.store = StateTransactionStore(self.stores, self.identity, self.tracer)
        self.boundary = AcceptedBoundary(37, hashlib.sha256(b"accepted tokens").hexdigest(), True)
        self.payloads = {
            rank: {
                family: FamilyPayload(
                    f"rank={rank};family={family};T=37".encode(),
                    f"discard speculative rank={rank} family={family}".encode(),
                )
                for family in STATE_FAMILIES
            }
            for rank in range(2)
        }

    def tearDown(self):
        self.temp.cleanup()

    def _commit(self, txn="txn-001", injector=None):
        return self.store.commit("session-1", txn, self.boundary, self.payloads, injector)

    def test_two_rank_commit_and_staged_restore_publish_once(self):
        commit = self._commit()
        self.assertEqual(commit, (self.stores[0] / "transactions/txn-001/COMMITTED.json").read_bytes())
        self.assertEqual(commit, (self.stores[1] / "transactions/txn-001/COMMITTED.json").read_bytes())
        self.assertEqual(
            (self.stores[0] / "sessions/session-1/index.json").read_bytes(),
            (self.stores[1] / "sessions/session-1/index.json").read_bytes(),
        )
        for rank in range(2):
            prepared = json.loads((self.stores[rank] / "transactions/txn-001/PREPARED.json").read_text())
            self.assertEqual([item["family"] for item in prepared["family_table"]], list(STATE_FAMILIES))
            for extent in prepared["family_table"]:
                self.assertEqual(extent["offset_bytes"] % PAGE_BYTES, 0)
                self.assertEqual(extent["length_bytes"] % PAGE_BYTES, 0)
                path = self.stores[rank] / "transactions/txn-001/families" / extent["file"]
                self.assertEqual(path.stat().st_size, PAGE_BYTES)
                self.assertNotIn(b"discard speculative", path.read_bytes())

        publications = []
        restored = self.store.restore("session-1", publications.append)
        self.assertEqual(len(publications), 1)
        self.assertIs(publications[0], restored)
        self.assertEqual(restored.boundary, self.boundary)
        self.assertEqual(
            {
                rank: {
                    family: payload.accepted
                    for family, payload in restored.rank_payload(rank).items()
                }
                for rank in range(2)
            },
            {
                rank: {
                    family: payload.accepted
                    for family, payload in self.payloads[rank].items()
                }
                for rank in range(2)
            },
        )
        with self.assertRaises(TypeError):
            AuthenticatedState()
        with self.assertRaises(TypeError):
            restored._boundary = AcceptedBoundary(38, self.boundary.token_hash, True)
        with self.assertRaises(TypeError):
            restored.rank_payload(0)[STATE_FAMILIES[0]] = FamilyPayload(b"forged")
        allowed = {"phase", "rank", "family", "outcome"}
        self.assertTrue(self.tracer.spans)
        self.assertTrue(all(set(span.attributes) <= allowed for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["phase"] in {"prepare", "write", "commit", "index", "read", "publish"}
                            for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["rank"] in {-1, 0, 1} for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["family"] in {*STATE_FAMILIES, "none"}
                            for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["outcome"] in {"success", "failure"}
                            for span in self.tracer.spans))

    def test_fault_after_each_transition_never_exposes_partial_state(self):
        for transition in Transition:
            with self.subTest(transition=transition):
                nested = self.root / transition.value
                tracer = Tracer()
                store = StateTransactionStore((nested / "rank0", nested / "rank1"), self.identity, tracer)

                def fail_at(observed):
                    if observed is transition:
                        raise InjectedFault(transition.value)

                with self.assertRaises(InjectedFault):
                    store.commit("session", "txn", self.boundary, self.payloads, fail_at)
                published = []
                if transition is Transition.AFTER_INDEX_RANK1:
                    restored = store.restore("session", published.append)
                    self.assertEqual(len(restored.rank_payloads), 2)
                    self.assertEqual(len(published), 1)
                else:
                    with self.assertRaises(StateTransactionError):
                        store.restore("session", published.append)
                    self.assertEqual(published, [])

    def test_tamper_is_rejected_before_publish(self):
        self._commit()
        target = self.stores[1] / "transactions/txn-001/families/00-target_full_attention_main_kv.state"
        with target.open("r+b") as stream:
            first = stream.read(1); stream.seek(0); stream.write(bytes([first[0] ^ 1]))
        published = []
        with self.assertRaisesRegex(StateTransactionError, "checksum"):
            self.store.restore("session-1", published.append)
        self.assertEqual(published, [])

    def test_prepared_garbage_and_rank_index_disagreement_are_ineligible(self):
        def fail_after_prepare(transition):
            if transition is Transition.AFTER_PREPARE_RANK1:
                raise InjectedFault("prepared")

        with self.assertRaises(InjectedFault): self._commit(injector=fail_after_prepare)
        with self.assertRaises(StateTransactionError): self.store.restore("session-1", lambda _state: None)

        other = StateTransactionStore((self.root / "other0", self.root / "other1"), self.identity, Tracer())
        other.commit("session-2", "txn-a", self.boundary, self.payloads)
        index = other._stores[1] / "sessions/session-2/index.json"
        value = json.loads(index.read_text()); value["transaction_id"] = "txn-b"
        index.write_text(json.dumps(value))
        with self.assertRaisesRegex(StateTransactionError, "indexes are not identical"):
            other.restore("session-2", lambda _state: self.fail("must not publish"))

    def test_boundary_inventory_and_identity_fail_closed_before_payload_io(self):
        bad_boundary = AcceptedBoundary(37, self.boundary.token_hash, False)
        with self.assertRaisesRegex(StateTransactionError, "quiesced"):
            self.store.commit("session", "txn", bad_boundary, self.payloads)
        missing = {rank: dict(values) for rank, values in self.payloads.items()}
        missing[0].pop(STATE_FAMILIES[-1])
        with self.assertRaisesRegex(StateTransactionError, "nine-family"):
            self.store.commit("session", "txn", self.boundary, missing)
        with self.assertRaises(StateTransactionError):
            StateTransactionStore(self.stores, StateIdentity("rev", "x" * 64), Tracer())
        self.assertFalse(any(store.exists() for store in self.stores))

    def test_direct_io_failure_has_no_buffered_or_mmap_fallback(self):
        real_open = os.open

        def reject_direct(path, flags, *args):
            if flags & os.O_DIRECT:
                raise OSError(22, "direct rejected")
            return real_open(path, flags, *args)

        with mock.patch("qwen38_slab.state_txn.os.open", side_effect=reject_direct):
            with self.assertRaisesRegex(StateTransactionError, "fallback is forbidden"):
                self._commit()
        transaction = self.stores[0] / "transactions/txn-001"
        self.assertFalse((transaction / "PREPARED.json").exists())

    def test_all_payload_mmaps_are_anonymous_and_direct_offsets_are_aligned(self):
        from qwen38_slab import state_txn

        real_mmap = state_txn.mmap.mmap
        real_pwritev = state_txn.os.pwritev
        real_preadv = state_txn.os.preadv
        mappings = []
        io_calls = []

        def anonymous_only(fileno, length, *args, **kwargs):
            mappings.append((fileno, length))
            self.assertEqual(fileno, -1)
            self.assertEqual(length % PAGE_BYTES, 0)
            return real_mmap(fileno, length, *args, **kwargs)

        def capture_write(fd, buffers, offset):
            io_calls.append(("write", offset, len(buffers[0])))
            return real_pwritev(fd, buffers, offset)

        def capture_read(fd, buffers, offset):
            io_calls.append(("read", offset, len(buffers[0])))
            return real_preadv(fd, buffers, offset)

        with mock.patch("qwen38_slab.state_txn.mmap.mmap", side_effect=anonymous_only), \
             mock.patch("qwen38_slab.state_txn.os.pwritev", side_effect=capture_write), \
             mock.patch("qwen38_slab.state_txn.os.preadv", side_effect=capture_read), \
             mock.patch("qwen38_slab.state_txn.IO_CHUNK_BYTES", PAGE_BYTES):
            first = STATE_FAMILIES[0]
            for rank in range(2):
                self.payloads[rank][first] = FamilyPayload(b"x" * (PAGE_BYTES + 1), b"discarded")
            self._commit()
            self.store.restore("session-1", lambda _state: None)
        self.assertTrue(mappings)
        self.assertTrue(io_calls)
        self.assertTrue(all(offset % PAGE_BYTES == 0 and length % PAGE_BYTES == 0
                            for _operation, offset, length in io_calls))
        self.assertIn(("write", PAGE_BYTES, PAGE_BYTES), io_calls)
        self.assertIn(("read", PAGE_BYTES, PAGE_BYTES), io_calls)

    def test_restore_checks_runtime_revision_precision_topology_and_token_boundary(self):
        self._commit()
        mismatched = StateTransactionStore(
            self.stores,
            StateIdentity("different-revision", self.identity.precision_digest),
            Tracer(),
        )
        with self.assertRaisesRegex(StateTransactionError, "identity"):
            mismatched.restore("session-1", lambda _state: self.fail("must not publish"))

    def test_every_commit_restore_identity_dimension_is_authenticated(self):
        mutations = {
            "revision": lambda value: value.__setitem__("revision", "other"),
            "precision": lambda value: value.__setitem__("precision_digest", "0" * 64),
            "topology": lambda value: value.__setitem__("topology", "tp4"),
            "T": lambda value: value.__setitem__("token_count", 38),
            "token hash": lambda value: value.__setitem__("token_hash", "0" * 64),
            "family table": lambda value: value["families"].reverse(),
        }
        for index, (dimension, mutate) in enumerate(mutations.items()):
            with self.subTest(dimension=dimension):
                root = self.root / f"identity-{index}"
                stores = (root / "rank0", root / "rank1")
                transaction = StateTransactionStore(stores, self.identity, Tracer())
                transaction.commit("session", "txn", self.boundary, self.payloads)
                commit_path = stores[0] / "transactions/txn/COMMITTED.json"
                value = json.loads(commit_path.read_text())
                mutate(value)
                raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for store in stores:
                    (store / "transactions/txn/COMMITTED.json").write_bytes(raw)
                    index_path = store / "sessions/session/index.json"
                    index_value = json.loads(index_path.read_text())
                    index_value["commit_sha256"] = hashlib.sha256(raw).hexdigest()
                    index_path.write_text(json.dumps(index_value, sort_keys=True, separators=(",", ":")) + "\n")
                published = []
                with self.assertRaises(StateTransactionError):
                    transaction.restore("session", published.append)
                self.assertEqual(published, [])


class InjectedFault(RuntimeError):
    pass


if __name__ == "__main__":
    unittest.main()
