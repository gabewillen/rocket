# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import unittest

from qwen38_slab.controller_restore import DecoderContinuation
from qwen38_slab.controller_rpc import (
    AuthenticatedRpcChannel,
    AuthenticatedRpcWorker,
    ControllerRpcError,
    PhysicalDecoderRestoreController,
    RemoteAllocation,
    RpcCudaRuntime,
)


KEY = hashlib.sha256(b"ephemeral-test-controller-key").digest()


def decode_request(encoded, rank, sequence, command):
    envelope = json.loads(encoded)
    signature = envelope.pop("hmac_sha256")
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    if not hmac.compare_digest(signature, hmac.new(KEY, canonical, hashlib.sha256).hexdigest()):
        raise AssertionError("request HMAC changed")
    assert envelope["rank"] == rank
    assert envelope["sequence"] == sequence
    assert envelope["command"] == command


def response(rank, sequence, command, result, *, sequence_override=None):
    envelope = {
        "schema": "rocket.qwen38.controller-worker.v1",
        "direction": "response",
        "rank": rank,
        "sequence": sequence if sequence_override is None else sequence_override,
        "command": command,
        "payload": {"ok": True, "result": result},
    }
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    envelope["hmac_sha256"] = hmac.new(KEY, canonical, hashlib.sha256).hexdigest()
    return json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()


class ScriptedTransport:
    def __init__(self, handler):
        self.handler = handler
        self.calls = 0

    def exchange(self, request, timeout_seconds):
        self.calls += 1
        return self.handler(request, timeout_seconds, self.calls)


class LoopTransport:
    def __init__(self, worker):
        self.worker = worker

    def exchange(self, request, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        return self.worker.handle(request)


class Owner:
    def __init__(self, rank):
        self.rank = rank


class Channel:
    def __init__(self, rank, fail_command=None):
        self.rank = rank
        self.fail_command = fail_command
        self.commands = []

    def request(self, command, payload):
        self.commands.append((command, payload))
        if command == self.fail_command:
            raise ControllerRpcError("injected worker fault")
        return {}


class Controller:
    def __init__(self, continuation):
        self.continuation = continuation

    def restore(self, session_id):
        if session_id != "session":
            raise RuntimeError("session changed")
        return self.continuation


class PhysicalRuntime:
    def __init__(
        self, rank, preflight=None, release=None, release_hook=None, open_hook=None
    ):
        self.owner = Owner(rank)
        self.preflight = preflight
        self.release = release
        self.release_hook = release_hook
        self.open_hook = open_hook
        self.faults = 0
        self.gate_open = False
        self.launches = []

    def accept_common(self, continuation):
        if isinstance(self.preflight, BaseException):
            raise self.preflight
        return self.preflight or {
            "rank": self.owner.rank,
            "phase": "accepted",
            "token_hash": continuation.token_hash,
            "generation_epoch": continuation.generation_epoch,
        }

    def release_common(self, continuation):
        if self.release_hook is not None:
            self.release_hook()
        if isinstance(self.release, BaseException):
            raise self.release
        return self.release or {
            "rank": self.owner.rank,
            "phase": "armed",
            "token_hash": continuation.token_hash,
            "generation_epoch": continuation.generation_epoch,
        }

    def open_common(self, continuation):
        if self.open_hook is not None:
            self.open_hook()
        self.gate_open = True
        return {
            "rank": self.owner.rank,
            "phase": "open",
            "token_hash": continuation.token_hash,
            "generation_epoch": continuation.generation_epoch,
        }

    def launch(self, continuation, generation):
        if not self.gate_open:
            raise ControllerRpcError("worker gate is closed")
        self.launches.append((continuation, generation))
        return {"generation": generation}

    def fault_physical(self, continuation=None):
        del continuation
        self.faults += 1


class ControllerRpcTests(unittest.TestCase):
    def setUp(self):
        self.continuation = DecoderContinuation(
            262_144,
            hashlib.sha256(b"accepted-continuation").hexdigest(),
            7,
            hashlib.sha256(b"commit").hexdigest(),
            hashlib.sha256(b"policy").hexdigest(),
        )

    def test_authenticated_round_trip_and_duplicate_response_fail_closed(self):
        def handler(request, _timeout, call):
            decode_request(request, 0, call, "status")
            return response(0, call, "status", {"rank": 0}, sequence_override=1)

        channel = AuthenticatedRpcChannel(0, KEY, ScriptedTransport(handler), 1)
        self.assertEqual(channel.request("status", {}), {"rank": 0})
        with self.assertRaisesRegex(ControllerRpcError, "authentication"):
            channel.request("status", {})

    def test_worker_and_controller_codecs_bind_rank_command_and_sequence(self):
        worker = AuthenticatedRpcWorker(
            1, KEY, lambda command, payload: {"command": command, **payload}
        )
        transport = LoopTransport(worker)
        channel = AuthenticatedRpcChannel(1, KEY, transport, 2)
        self.assertEqual(
            channel.request("status", {"value": 7}),
            {"command": "status", "value": 7},
        )
        self.assertEqual(transport.timeout_seconds, 2.0)

    def test_worker_timeout_fails_closed(self):
        def timeout(_request, timeout_seconds, _call):
            self.assertEqual(timeout_seconds, 0.25)
            raise TimeoutError("peer disappeared")

        channel = AuthenticatedRpcChannel(0, KEY, ScriptedTransport(timeout), 0.25)
        with self.assertRaisesRegex(ControllerRpcError, "timed out"):
            channel.request("status", {})

    def test_oversized_request_is_rejected_before_transport(self):
        transport = ScriptedTransport(lambda *_args: self.fail("transport was called"))
        channel = AuthenticatedRpcChannel(0, KEY, transport, 1)
        with self.assertRaisesRegex(ControllerRpcError, "exceeds 65536 bytes"):
            channel.request("status", {"padding": "x" * 65_536})
        self.assertEqual(transport.calls, 0)

    def test_peer_loss_before_common_release_faults_both_workers(self):
        ranks = (
            PhysicalRuntime(0),
            PhysicalRuntime(1, preflight=TimeoutError("peer lost")),
        )
        controller = PhysicalDecoderRestoreController(
            Controller(self.continuation), ranks
        )
        with self.assertRaisesRegex(ControllerRpcError, "physical TP2"):
            controller.restore("session")
        self.assertEqual(tuple(rank.faults for rank in ranks), (1, 1))

    def test_duplicate_rank_receipt_fails_before_release(self):
        ranks = (
            PhysicalRuntime(0),
            PhysicalRuntime(1, preflight={
                "rank": 0,
                "phase": "accepted",
                "token_hash": self.continuation.token_hash,
                "generation_epoch": 7,
            }),
        )
        with self.assertRaisesRegex(ControllerRpcError, "acknowledgement"):
            PhysicalDecoderRestoreController(
                Controller(self.continuation), ranks
            ).restore("session")
        self.assertEqual(tuple(rank.faults for rank in ranks), (1, 1))

    def test_mismatched_continuation_hash_fails_before_release(self):
        ranks = (
            PhysicalRuntime(0),
            PhysicalRuntime(1, preflight={
                "rank": 1,
                "phase": "accepted",
                "token_hash": hashlib.sha256(b"wrong").hexdigest(),
                "generation_epoch": 7,
            }),
        )
        with self.assertRaisesRegex(ControllerRpcError, "acknowledgement"):
            PhysicalDecoderRestoreController(
                Controller(self.continuation), ranks
            ).restore("session")
        self.assertEqual(tuple(rank.faults for rank in ranks), (1, 1))

    def test_rank1_release_failure_faults_both_and_never_publishes(self):
        ranks = (
            PhysicalRuntime(0),
            PhysicalRuntime(1, release=ControllerRpcError("release fault")),
        )
        controller = PhysicalDecoderRestoreController(
            Controller(self.continuation), ranks
        )
        with self.assertRaisesRegex(ControllerRpcError, "release fault"):
            controller.restore("session")
        self.assertEqual(tuple(rank.faults for rank in ranks), (1, 1))
        self.assertFalse(any(rank.gate_open for rank in ranks))
        with self.assertRaisesRegex(ControllerRpcError, "not published"):
            controller.admit(self.continuation)

    def test_admission_waits_until_both_open_triggers_complete(self):
        first_released = threading.Event()
        allow_second = threading.Event()

        def mark_first():
            first_released.set()

        def hold_second():
            self.assertTrue(allow_second.wait(2))

        ranks = (
            PhysicalRuntime(0, open_hook=mark_first),
            PhysicalRuntime(1, open_hook=hold_second),
        )
        controller = PhysicalDecoderRestoreController(
            Controller(self.continuation), ranks
        )
        restore_result = []
        restore = threading.Thread(
            target=lambda: restore_result.append(controller.restore("session"))
        )
        restore.start()
        self.assertTrue(first_released.wait(2))
        admission_complete = threading.Event()
        admission_result = []
        admission = threading.Thread(
            target=lambda: (
                admission_result.append(controller.launch(0, 8)),
                admission_complete.set(),
            )
        )
        admission.start()
        self.assertFalse(admission_complete.wait(0.05))
        allow_second.set()
        restore.join(2)
        admission.join(2)
        self.assertEqual(restore_result, [self.continuation])
        self.assertTrue(admission_complete.is_set())
        self.assertEqual(admission_result, [{"generation": 8}])
        self.assertEqual(ranks[0].launches, [(self.continuation, 8)])

    def test_rank1_post_allocation_copy_fault_does_not_prepare(self):
        channel = Channel(1, fail_command="copy")
        runtime = RpcCudaRuntime(Owner(1), channel)
        allocation = RemoteAllocation(1, "target_gdn_conv", 64, "allocation-1")
        runtime._allocations[allocation.token] = allocation

        class Extent:
            family = "target_gdn_conv"
            logical_bytes = 64
            length_bytes = 65_536
            path = "/rank1/03-target_gdn_conv.state"
            logical_sha256 = hashlib.sha256(b"logical").hexdigest()
            padded_sha256 = hashlib.sha256(b"padded").hexdigest()

        with self.assertRaisesRegex(ControllerRpcError, "injected worker fault"):
            runtime.copy_extent_to_device(allocation, Extent())
        self.assertEqual([item[0] for item in channel.commands], ["copy"])


if __name__ == "__main__":
    unittest.main()
