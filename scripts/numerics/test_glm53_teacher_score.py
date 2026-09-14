import importlib.util
import json
import math
import pathlib
import struct
import tempfile
import unittest
from array import array

PATH = pathlib.Path(__file__).with_name("glm53-teacher-score.py")
SPEC = importlib.util.spec_from_file_location("glm53_teacher_score", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def write_trace(path, logprobs, top1):
    with path.open("w") as handle:
        handle.write(json.dumps({"type": "metadata", "schema": "rocket.glm53.teacher-score.v1", "vocab_size": 3}) + "\n")
        for i, (lp, best) in enumerate(zip(logprobs, top1)):
            handle.write(json.dumps({"type": "token", "sequence": 1, "position": i,
                                     "input_id": i, "target_id": 1,
                                     "target_logprob": lp, "logsumexp": 0.0,
                                     "top5": [{"id": best, "logprob": -0.1},
                                              {"id": 1 if best != 1 else 2, "logprob": -1.0},
                                              {"id": 0, "logprob": -2.0},
                                              {"id": 3, "logprob": -3.0},
                                              {"id": 4, "logprob": -4.0}]}) + "\n")
        handle.write(json.dumps({"type": "summary", "sequences": 1, "tokens": len(logprobs)}) + "\n")


def write_logits(path, rows):
    with path.open("wb") as handle:
        handle.write(struct.pack("<8sIQ", MOD.MAGIC, len(rows[0]), len(rows)))
        values = array("f", [x for row in rows for x in row])
        values.tofile(handle)


class TeacherScoreTest(unittest.TestCase):
    def test_summary_and_comparison(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            a, b = root / "a.jsonl", root / "b.jsonl"
            al, bl = root / "a.bin", root / "b.bin"
            write_trace(a, [-1.0, -2.0], [1, 2])
            write_trace(b, [-1.1, -1.9], [1, 0])
            write_logits(al, [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
            write_logits(bl, [[1.9, 1.1, 0.0], [0.1, 1.0, 1.9]])
            _, ar, _ = MOD.read_trace(a)
            _, br, _ = MOD.read_trace(b)
            result = MOD.compare(ar, br, MOD.read_logits(al), MOD.read_logits(bl))
            self.assertEqual(result["tokens"], 2)
            self.assertEqual(result["top1_agreement"], 0.5)
            self.assertEqual(result["first_top1_divergence_row"], 2)
            self.assertAlmostEqual(result["perplexity_ratio"], 1.0)
            self.assertGreater(result["mean_kl_divergence_nats"], 0.0)
            self.assertGreater(result["centered_logit_rms_error"], 0.0)

    def test_identical_logits_have_zero_drift(self):
        rows = [{"sequence": 1, "position": 0, "input_id": 1, "target_id": 2,
                 "target_logprob": -0.5,
                 "top5": [{"id": x, "logprob": -x} for x in range(5)]}]
        logits = (3, 1, array("f", [0.0, 1.0, 2.0]))
        result = MOD.compare(rows, rows, logits, logits)
        self.assertAlmostEqual(result["mean_kl_divergence_nats"], 0.0)
        self.assertAlmostEqual(result["logit_rms_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
