#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import asyncio
import json
import sys
import tempfile
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
        self.assertEqual(rendered.count("AsyncOpenAI("), 1)
        self.assertEqual(rendered.count("await stream.close()"), 1)
        self.assertEqual(rendered.count("await client.close()"), 1)
        self.assertEqual(rendered.count("max_retries=0"), 1)
        self.assertNotIn("class Worker(threading.Thread)", rendered)
        self.assertNotIn("r.fp.raw._sock", rendered)
        self.assertNotIn("sock.shutdown", rendered)
        self.assertNotIn("Stdlib only", rendered)
        with self.assertRaisesRegex(MODULE.ContractError, "anchor changed"):
            MODULE.render_runner(rendered)

    def test_sdk_cancellation_closes_all_streams_and_client(self):
        instances = []

        class BlockedStream:
            def __init__(self):
                self.closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()

            async def close(self):
                self.closed = True

        class Completions:
            def __init__(self, client):
                self.client = client

            async def create(self, **request):
                self.client.requests.append(request)
                stream = BlockedStream()
                self.client.streams.append(stream)
                return stream

        class AsyncClient:
            def __init__(self, **options):
                self.options = options
                self.requests = []
                self.streams = []
                self.closed = False
                self.chat = types.SimpleNamespace(completions=Completions(self))
                instances.append(self)

            async def close(self):
                self.closed = True

        namespace = {"__name__": "rendered_runner", "__file__": MODULE.RUNNER / MODULE.RUNNER_REL}
        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = AsyncClient
        fake_openai.__version__ = "3.3.1"
        previous_openai = sys.modules.get("openai")
        sys.modules["openai"] = fake_openai
        try:
            exec(MODULE.render_runner((MODULE.RUNNER / MODULE.RUNNER_REL).read_text()), namespace)
        finally:
            if previous_openai is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = previous_openai
        args = types.SimpleNamespace(
            tag="test", model="model", temperature=0.6, top_p=0.95,
            max_tokens=32768, thinking="off", url="http://test", token="", timeout=30,
        )
        namespace["sample_loop"] = lambda *_: None
        log = asyncio.run(
            namespace["run_stream_cohort"](args, "prompt", 16, 0, 1, [])
        )
        self.assertEqual(len(instances), 1)
        client = instances[0]
        self.assertTrue(client.closed)
        self.assertEqual(len(client.requests), 16)
        self.assertTrue(all(stream.closed for stream in client.streams))
        self.assertEqual(len(log), 16)
        self.assertEqual({record["finish"] for record in log}, {"aborted"})
        for request in client.requests:
            self.assertTrue(request["stream"])
            self.assertEqual(request["extra_body"], {
                "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False},
            })
        self.assertEqual(client.options, {
            "base_url": "http://test",
            "api_key": "rocket-benchmark-dummy",
            "timeout": 30,
            "max_retries": 0,
        })

    def test_generated_runner_rejects_another_sdk_version(self):
        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = object
        fake_openai.__version__ = "3.2.0"
        previous_openai = sys.modules.get("openai")
        sys.modules["openai"] = fake_openai
        try:
            with self.assertRaisesRegex(RuntimeError, "openai 3.3.1 required"):
                exec(
                    MODULE.render_runner(
                        (MODULE.RUNNER / MODULE.RUNNER_REL).read_text()
                    ),
                    {"__name__": "rendered_runner", "__file__": MODULE.RUNNER / MODULE.RUNNER_REL},
                )
        finally:
            if previous_openai is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = previous_openai

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
            self.assertEqual(contract["openai_python_version"], "3.3.1")
            self.assertEqual(
                contract["openai_python_commit"],
                "753ab5c1a81cd85e8bf0aef4c04c51a2e8dae6cd",
            )
            self.assertEqual(
                contract["openai_async_example_sha256"],
                "50468f6737372e9dc3d6f46e74225605d20e02f80200efff317517cba0ea0f28",
            )
            self.assertEqual(
                contract["otel_scope"],
                "benchmark-only external client; runtime/service telemetry boundary unchanged",
            )
            with self.assertRaisesRegex(MODULE.ContractError, "already exists"):
                MODULE.prepare_cell(root, cell)


if __name__ == "__main__":
    unittest.main()
