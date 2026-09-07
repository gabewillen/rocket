#!/usr/bin/env python3
import importlib.util
import pathlib
import unittest

PATH = pathlib.Path(__file__).with_name("qwen38-compare-precision.py")
SPEC = importlib.util.spec_from_file_location("compare_precision", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(value=2.0):
    return {"telemetry": {"layer.0.x.output": {"kind": "projection_output", "rms": value, "abs_p99": value,
            "absmax": value, "histogram_log2": [1, 3]}}}


def quality(ok=True, tail="x"):
    return {"results": {"case": {"ok": ok, "answer_tail": tail}}}


class ComparePrecisionTest(unittest.TestCase):
    def test_reports_ratios_histogram_and_parity(self):
        result = MODULE.compare(summary(2), summary(4), quality(), quality())
        row = result["channels"]["layer.0.x.output"]
        self.assertEqual(row["rms_ratio"], 2)
        self.assertEqual(row["histogram_tv"], 0)
        self.assertEqual(result["quality"]["answer_tail_equal"], 1)
        self.assertEqual(result["quality"]["regressions"], [])

    def test_fails_closed_on_channel_mismatch(self):
        candidate = summary()
        candidate["telemetry"]["extra"] = candidate["telemetry"].pop("layer.0.x.output")
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            MODULE.compare(summary(), candidate, quality(), quality())

    def test_reports_quality_regression(self):
        result = MODULE.compare(summary(), summary(), quality(), quality(False, "y"))
        self.assertEqual(result["quality"]["regressions"], ["case"])


if __name__ == "__main__":
    unittest.main()
