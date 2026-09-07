#!/usr/bin/env python3
"""Unit tests for qwen38-attention-calibration.py."""

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PATH = pathlib.Path(__file__).with_name("qwen38-attention-calibration.py")
SPEC = importlib.util.spec_from_file_location("qwen38_attention_calibration", PATH)
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


class Handler(BaseHTTPRequestHandler):
    requests = []
    speculative_config = None
    expose_speculative_metrics = False
    draft_attempts = 0
    accepted_drafts = 0
    lock = threading.Lock()

    def log_message(self, *_args):
        pass

    def send_json(self, value):
        encoded = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/server_info":
            self.send_json({"server_args": {"speculative_config": self.speculative_config}})
        elif self.path == "/v1/models":
            self.send_json({"data": [{"id": "qwen3.8-flash-next"}]})
        elif self.path == "/metrics" and self.expose_speculative_metrics:
            encoded = (
                f"vllm:spec_decode_num_drafts_total {self.draft_attempts}\n"
                f"vllm:spec_decode_num_accepted_tokens_total {self.accepted_drafts}\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        with self.lock:
            self.requests.append(body)
            if self.speculative_config is not None and self.expose_speculative_metrics:
                type(self).draft_attempts += 3
                type(self).accepted_drafts += 2
        self.send_json({
            "choices": [{"message": {"content": "calibration response"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 5},
        })


class CalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        Handler.requests = []
        Handler.speculative_config = None
        Handler.expose_speculative_metrics = False
        Handler.draft_attempts = 0
        Handler.accepted_drafts = 0

    def args(self, **overrides):
        values = {
            "endpoint": self.endpoint,
            "model": "qwen3.8-flash-next",
            "seed": 1234,
            "timeout": 5,
            "long_decode_tokens": 64,
            "concurrent_streams": 3,
            "mtp": False,
            "mtp_tokens": 32,
            "speculative_config": None,
            "api_key": None,
            "out": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_manifest_covers_every_required_mechanism(self):
        cases = calibration.build_cases(1234, 64)
        concurrent = calibration.concurrent_cases(1234, 3)
        manifest = calibration.coverage_manifest(cases, concurrent, True)
        required = {
            "recurrent_linear_attention", "long_decode", "repetition", "alternation",
            "state_transition", "full_attention", "exact_copy", "induction",
            "distant_needle", "near_match_distractors", "beginning_retrieval",
            "middle_retrieval", "end_retrieval", "ple", "overlapping_ngrams",
            "code_structure", "json_structure", "rare_tokens", "multilingual",
            "concurrent_variable_length_streams", "mtp_speculation",
        }
        self.assertTrue(required.issubset(manifest["mechanisms"]))
        self.assertEqual(manifest["schema"], calibration.SCHEMA)

    def test_manifest_records_checkpoint_mechanism_facts(self):
        manifest = calibration.coverage_manifest(
            calibration.build_cases(1234, 64),
            calibration.concurrent_cases(1234, 3),
            True,
        )
        facts = manifest["checkpoint_facts"]
        self.assertEqual(facts["text_layers"], {
            "count": 48,
            "pattern": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            "linear_attention_count": 36,
            "full_attention_count": 12,
        })
        self.assertEqual(facts["ple"], {"layer_ids": [2]})
        self.assertEqual(facts["qsa"], {
            "applies_to": "full_attention", "indexer_n_heads": 4,
        })
        self.assertEqual(facts["moe"], {"experts": 512, "top_k": 10})
        self.assertEqual(facts["mtp"], {
            "hybrid": True, "num_hidden_layers": 1, "layer_type": "full_attention",
        })

    def test_cases_and_request_seeds_are_deterministic(self):
        first = calibration.build_cases(77, 128)
        second = calibration.build_cases(77, 128)
        self.assertEqual(first, second)
        report = calibration.run(self.args(seed=77))
        self.assertEqual(report["seed"], 77)
        self.assertEqual(report["summary"], {
            "total": 13, "completed": 13, "failed": 0, "unavailable": 0,
            "configured_but_unverified": 0,
        })
        seeds = [request["seed"] for request in Handler.requests]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertTrue(all(request["temperature"] == 0 for request in Handler.requests))

    def test_concurrent_cases_have_variable_prompt_and_decode_lengths(self):
        calibration.run(self.args())
        requests = Handler.requests[-3:]
        self.assertEqual(len({request["max_tokens"] for request in requests}), 3)
        self.assertEqual(len({len(request["messages"][0]["content"]) for request in requests}), 3)

    def test_mtp_is_unavailable_when_speculative_config_is_none(self):
        report = calibration.run(self.args(mtp=True))
        self.assertEqual(report["results"]["mtp_speculative_decode"]["status"], "unavailable")
        self.assertEqual(
            report["results"]["mtp_speculative_decode"]["reason"],
            "speculative_config is None",
        )
        self.assertFalse(report["mtp"]["speculative_config_available"])
        self.assertEqual(report["mtp"]["runtime_coverage"], "unavailable")
        self.assertEqual(report["mtp"]["checkpoint_capable"]["layer_type"], "full_attention")

    def test_caller_config_cannot_prove_mtp_on_non_speculative_server(self):
        report = calibration.run(self.args(
            mtp=True,
            speculative_config={"method": "arbitrary", "num_speculative_tokens": 99},
        ))
        self.assertEqual(report["results"]["mtp_speculative_decode"]["status"], "unavailable")
        self.assertEqual(report["mtp"]["runtime_coverage"], "unavailable")
        self.assertFalse(report["mtp"]["runtime_config_discovered"])

    def test_mtp_runs_when_server_reports_speculative_config(self):
        Handler.speculative_config = {"method": "qwen3_next_mtp", "num_speculative_tokens": 3}
        report = calibration.run(self.args(mtp=True))
        self.assertEqual(
            report["results"]["mtp_speculative_decode"]["status"],
            "configured_but_unverified",
        )
        self.assertTrue(report["mtp"]["speculative_config_available"])
        self.assertEqual(report["mtp"]["runtime_coverage"], "configured_but_unverified")

    def test_mtp_completes_with_runtime_config_and_metric_delta(self):
        Handler.speculative_config = {"method": "qwen3_next_mtp", "num_speculative_tokens": 3}
        Handler.expose_speculative_metrics = True
        report = calibration.run(self.args(mtp=True))
        result = report["results"]["mtp_speculative_decode"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["speculation_metric_deltas"], {
            "vllm:spec_decode_num_drafts_total": 3.0,
            "vllm:spec_decode_num_accepted_tokens_total": 2.0,
        })

    def test_mtp_runtime_coverage_is_not_requested_without_option(self):
        report = calibration.run(self.args())
        self.assertEqual(report["mtp"]["runtime_coverage"], "not_requested")
        self.assertIsNone(report["mtp"]["speculative_config_available"])

    def test_endpoint_accepts_full_chat_completions_url(self):
        report = calibration.run(self.args(endpoint=self.endpoint + "/v1/chat/completions"))
        self.assertEqual(report["summary"]["failed"], 0)

    def test_main_writes_only_explicit_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.json"
            code = calibration.main([
                "--endpoint", self.endpoint,
                "--long-decode-tokens", "32",
                "--concurrent-streams", "1",
                "--out", str(output),
            ])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())["schema"], calibration.SCHEMA)


if __name__ == "__main__":
    unittest.main()
