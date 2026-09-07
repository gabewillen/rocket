#!/usr/bin/env python3
"""Focused output-contract tests for the forked-prefix benchmark."""

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("openai-forked-prefix.py")
SPEC = importlib.util.spec_from_file_location("forked_prefix", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class ForkedPrefixTest(unittest.TestCase):
    def test_json_cases_carry_utc_and_monotonic_boundaries(self):
        row = {"prompt_tokens": 10, "completion_tokens": 4, "ttft_s": 0.1,
               "decode_s": 0.5, "first_at": 10.0, "finished_at": 10.5}
        output = io.StringIO()
        with mock.patch.object(BENCHMARK, "request", return_value=row), \
             mock.patch.object(sys, "argv", [str(SCRIPT), "--concurrency", "1", "--json"]), \
             contextlib.redirect_stdout(output):
            BENCHMARK.main()
        case = json.loads(output.getvalue())[0]
        self.assertLess(case["started_unix_ns"], case["finished_unix_ns"])
        self.assertTrue(case["started_at"].endswith("Z"))
        self.assertTrue(case["finished_at"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
