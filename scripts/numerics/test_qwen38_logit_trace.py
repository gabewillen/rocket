#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

PATH = pathlib.Path(__file__).with_name("qwen38-logit-trace.py")
SPEC = importlib.util.spec_from_file_location("logit_trace", PATH)
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def token(text, score, alternatives=None):
    alternatives = alternatives or [(text, score)]
    return {"token": text, "bytes": list(text.encode()), "logprob": score,
            "top_logprobs": [{"token": t, "bytes": list(t.encode()), "logprob": s} for t, s in alternatives]}


def report(tokens):
    return {"cases": {"x": {"tokens": tokens}}}


class LogitTraceTest(unittest.TestCase):
    def test_identical_tokens_report_drift(self):
        result = M.compare(report([token("a", -1, [("a", -1), ("b", -2)])]),
                           report([token("a", -1.25, [("a", -1.25), ("b", -3)])]))
        self.assertEqual(result["token_parity"], 1)
        self.assertEqual(result["cases"]["x"]["max_chosen_logprob_drift"], .25)
        self.assertEqual(result["cases"]["x"]["max_common_top_logprob_drift"], 1)

    def test_first_divergence_is_recorded(self):
        result = M.compare(report([token("a", 0), token("b", 0)]),
                           report([token("a", 0), token("c", 0)]))
        self.assertEqual(result["cases"]["x"]["first_divergence"], 1)
        self.assertEqual(result["cases"]["x"]["shared_prefix_tokens"], 1)

    def test_case_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "case mismatch"):
            M.compare(report([]), {"cases": {"y": {"tokens": []}}})


if __name__ == "__main__": unittest.main()
