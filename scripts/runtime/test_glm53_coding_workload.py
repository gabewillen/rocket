import copy
import json
import pathlib
import subprocess
import tempfile
import unittest

from glm53_coding_workload import load_workload, materialize_phase, useful_tokens, validate


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "scripts/runtime/fixtures/glm53-coding-sessions.jsonl"


class CodingWorkloadTest(unittest.TestCase):
    def setUp(self):
        self.sessions = load_workload(WORKLOAD)

    def test_committed_workload(self):
        stats = validate(self.sessions)
        self.assertEqual(64, stats["sessions"])
        self.assertEqual(64, stats["distinct_prompts"])
        self.assertGreaterEqual(sum(s["response_tokens"] for p in range(4)
                                    for s in self.sessions[p * 16:p * 16 + 8]), 4096)

    def test_repeated_source_block_fails(self):
        bad = copy.deepcopy(self.sessions)
        marker = "\nFILE duplicate.py SHA256 deadbeef\n"
        bad[0]["turns"][0]["content"] += marker + marker
        with self.assertRaisesRegex(ValueError, "repeats a source block"):
            validate(bad)

    def test_replicated_prompt_fails(self):
        bad = copy.deepcopy(self.sessions)
        bad[1]["turns"][0]["content"] = bad[0]["turns"][0]["content"]
        with self.assertRaisesRegex(ValueError, "replicated prompt"):
            validate(bad)

    def test_useful_token_accounting_rejects_padding(self):
        result = {"generated_token_ids": [[1, 2], [3, 4]], "useful_output_tokens": 5}
        with self.assertRaisesRegex(ValueError, "padding or rejected drafts"):
            useful_tokens(result)

    def test_adaptive_spec_map_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([
                str(ROOT / "scripts/runtime/run-glm53-coding-workload.py"),
                "--workload", str(WORKLOAD), "--batch", "8", "--spec-map", "7,4,3,2",
                "--out", td, "--dry-run"], check=True, stdout=subprocess.DEVNULL)
            summary = json.loads((pathlib.Path(td) / "summary.json").read_text())
            self.assertEqual([7, 4, 3, 2], summary["spec_map"])
            self.assertEqual([7, 4, 3, 2], [x["spec_k"] for x in summary["runs"]])

    def test_phase_materialization_is_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            selected, listing = materialize_phase(self.sessions, 1, 8, td)
            paths = listing.read_text().splitlines()
            self.assertEqual(8, len(paths))
            self.assertEqual(8, len({pathlib.Path(x).read_bytes() for x in paths}))
            self.assertTrue(all(s["context_tokens_target"] == 8192 for s in selected))


if __name__ == "__main__":
    unittest.main()
