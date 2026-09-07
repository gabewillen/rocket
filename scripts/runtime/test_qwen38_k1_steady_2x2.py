#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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
