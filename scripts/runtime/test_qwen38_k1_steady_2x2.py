#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qwen38-k1-steady-2x2.py")
SPEC = importlib.util.spec_from_file_location("qwen38_k1_steady_2x2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Steady2x2ContractTest(unittest.TestCase):
    def test_four_cells_vary_only_hca_and_cpuset(self):
        self.assertEqual(
            MODULE.CELLS,
            {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)},
        )
        source = (MODULE.FIXTURE / "launch-head.sh").read_text()
        for name, (dual, pinned) in MODULE.CELLS.items():
            rendered = MODULE.render_launch(source, MODULE.Cell(name, dual, pinned))
            self.assertEqual("--cpuset-cpus" in rendered, pinned)
            self.assertEqual("NCCL_IB_QPS_PER_CONNECTION=4" in rendered, dual)
            expected_hca = ",".join(MODULE.HCA) if dual else MODULE.HCA[0]
            self.assertIn(f"NCCL_IB_HCA={expected_hca}", rendered)
            self.assertIn("num_speculative_tokens\":1", rendered)

    def test_render_rejects_premodified_launch(self):
        source = (MODULE.FIXTURE / "launch-head.sh").read_text()
        with self.assertRaisesRegex(MODULE.ContractError, "already modified"):
            MODULE.render_launch(source.replace("--ipc host", "--ipc host --cpuset-cpus 0"), MODULE.Cell("A", False, False))

    def test_runner_overlay_keeps_fixed_cohort_after_eos(self):
        source = (MODULE.RUNNER / MODULE.RUNNER_REL).read_text()
        rendered = MODULE.render_runner(source)
        compile(rendered, MODULE.RUNNER_REL, "exec")
        self.assertNotIn('"ignore_eos": True', source)
        self.assertEqual(rendered.count('"ignore_eos": True'), 1)
        self.assertEqual(rendered.count("w.abort()"), 1)
        self.assertEqual(rendered.count("join_deadline = time.monotonic() + 30"), 1)
        with self.assertRaisesRegex(MODULE.ContractError, "anchor changed"):
            MODULE.render_runner(rendered)

    def test_runner_overlay_closes_blocked_sse_and_records_all_workers(self):
        namespace = {"__name__": "rendered_runner", "__file__": MODULE.RUNNER / MODULE.RUNNER_REL}
        exec(MODULE.render_runner((MODULE.RUNNER / MODULE.RUNNER_REL).read_text()), namespace)
        worker_type = namespace["Worker"]
        opened = threading.Event()

        peers = []

        class BlockedResponse:
            def __init__(self, client):
                self.socket = client
                self.fp = client.makefile("rb")

            def __enter__(self):
                opened.set()
                return self

            def __exit__(self, *_):
                self.close()

            def __iter__(self):
                return iter(self.fp)

            def close(self):
                self.fp.close()
                self.socket.close()

        responses = []

        def fake_urlopen(*_, **__):
            client, peer = socket.socketpair()
            peers.append(peer)
            response = BlockedResponse(client)
            responses.append(response)
            return response

        original_urlopen = namespace["urllib"].request.urlopen
        namespace["urllib"].request.urlopen = fake_urlopen
        args = types.SimpleNamespace(
            tag="test", model="model", temperature=0.6, top_p=0.95,
            max_tokens=32768, thinking="off", url="http://test", token="", timeout=30,
        )
        stop = threading.Event()
        log = []
        try:
            workers = [worker_type(i, args, "prompt", stop, log) for i in range(16)]
            for worker in workers:
                worker.start()
            self.assertTrue(opened.wait(1))
            deadline = time.monotonic() + 1
            while len(responses) != 16 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(len(responses), 16)
            stop.set()
            for worker in workers:
                worker.abort()
            join_deadline = time.monotonic() + 1
            for worker in workers:
                worker.join(max(0.0, join_deadline - time.monotonic()))
            self.assertFalse(any(worker.is_alive() for worker in workers))
            self.assertEqual(len(log), 16)
            self.assertEqual({record["finish"] for record in log}, {"aborted"})
        finally:
            namespace["urllib"].request.urlopen = original_urlopen
            for peer in peers:
                peer.close()

    def test_sources_and_fixture_are_exact(self):
        record = MODULE.validate_sources()
        self.assertEqual(record["mtp_depth"], 1)
        self.assertEqual(record["model_revision"], MODULE.MODEL_REVISION)

    def test_prepared_contract_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = MODULE.Cell("D", True, True)
            cell_dir = MODULE.prepare_cell(root, cell)
            contract = json.loads((cell_dir / "contract.json").read_text())
            self.assertEqual(contract["windows"], 3)
            self.assertEqual(contract["window_seconds"], 10)
            self.assertEqual(contract["otel_cardinality"], {"cell": 4, "rank": 2, "hca": 2})
            with self.assertRaisesRegex(MODULE.ContractError, "already exists"):
                MODULE.prepare_cell(root, cell)


if __name__ == "__main__":
    unittest.main()
