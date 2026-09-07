#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("qwen38_flashinfer_n640_tail.py")
SPEC = importlib.util.spec_from_file_location("qwen38_flashinfer_n640_tail", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

REFERENCE = Path("/home/glwillen/Development/inference-references/flashinfer")


class ExactN640TailTest(unittest.TestCase):
    def test_shape_contract_has_no_sixth_slice(self):
        self.assertEqual(MODULE.retained_group_count(640), 3)
        self.assertEqual(
            [MODULE.retained_slice_count(640, group) for group in range(3)],
            [2, 2, 1],
        )
        touched = [
            group * 2 + lane
            for group in range(3)
            for lane in range(MODULE.retained_slice_count(640, group))
        ]
        self.assertEqual(touched, [0, 1, 2, 3, 4])

    def test_padded_control_is_unchanged(self):
        self.assertEqual(MODULE.retained_group_count(768), 3)
        self.assertEqual(
            [MODULE.retained_slice_count(768, group) for group in range(3)],
            [2, 2, 2],
        )

    def test_invalid_shapes_fail_closed(self):
        for value in (0, 129, True, 640.0):
            with self.assertRaises(ValueError):
                MODULE.retained_group_count(value)
        for group in (3, 1.0, True):
            with self.assertRaises(ValueError):
                MODULE.retained_slice_count(640, group)

    def test_transformed_sources_compile_and_guard_four_pipelines(self):
        static_source = (REFERENCE / MODULE.STATIC_REL).read_text()
        transformed = MODULE.transform_static(static_source)
        compile(transformed, str(MODULE.STATIC_REL), "exec")
        self.assertEqual(transformed.count("if current_slice < gate_tile_cnt:"), 4)
        self.assertIn("Int32(retained_slice_idx)", transformed)
        self.assertIn("or current_slice + Int32(1) == gate_tile_cnt", transformed)

    def test_dispatch_keeps_exact_tail_opt_in(self):
        source = (REFERENCE / MODULE.DISPATCH_REL).read_text()
        transformed = MODULE.transform_dispatch(source)
        compile(transformed, str(MODULE.DISPATCH_REL), "exec")
        self.assertIn("_EXACT_N640_RETAINED_TAIL = False", transformed)
        self.assertIn("_EXACT_N640_RETAINED_TAIL and n == 640", transformed)
        self.assertEqual(
            transformed.count(
                "(n + _STATIC_RETAINED_GROUP_N - 1) // _STATIC_RETAINED_GROUP_N"
            ),
            2,
        )
        self.assertIn(
            "if not (_EXACT_N640_RETAINED_TAIL and n == 640):\n"
            "        n = _align_up(n, _STATIC_RETAINED_GROUP_N)",
            transformed,
        )
        self.assertIn(
            "(n + _STATIC_RETAINED_GROUP_N - 1)\n"
            "            // _STATIC_RETAINED_GROUP_N",
            transformed,
        )
        self.assertIn("barrier_phase_clock: torch.Tensor", transformed)
        self.assertIn("(_BARRIER_PHASES, _BARRIER_TRACE_SLOTS, 2)", transformed)

    def test_dynamic_barrier_trace_has_five_bounded_phases(self):
        source = (REFERENCE / MODULE.GENERIC_REL).read_text()
        transformed = MODULE.transform_dynamic_instrumentation(source)
        compile(transformed, str(MODULE.GENERIC_REL), "exec")
        self.assertEqual(transformed.count("barrier_phase_clock[trace_base]"), 1)
        self.assertEqual(
            transformed.count("barrier_phase_clock[trace_base + Int32(1)]"), 1
        )
        for phase in range(5):
            self.assertIn(
                f"barrier_phase_clock,\n            Int32({phase}),", transformed
            )
        self.assertNotIn(
            "barrier_phase_clock,\n            Int32(5),", transformed
        )

    def test_materialized_overlay_does_not_mutate_reference(self):
        before = {
            relative: (REFERENCE / relative).read_bytes()
            for relative in MODULE.REFERENCE_SHA256
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "flashinfer-overlay"
            MODULE.materialize(REFERENCE, output)
            MODULE.materialize(REFERENCE, output)
            for relative in MODULE.REFERENCE_SHA256:
                self.assertTrue((output / relative).is_file())
                self.assertNotEqual(
                    (output / relative).read_bytes(),
                    before[relative],
                )
        for relative, content in before.items():
            self.assertEqual((REFERENCE / relative).read_bytes(), content)

    def test_source_and_overlay_must_be_disjoint(self):
        with self.assertRaisesRegex(MODULE.OverlayContractError, "disjoint"):
            MODULE.materialize(REFERENCE, REFERENCE / "overlay")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            with self.assertRaisesRegex(MODULE.OverlayContractError, "disjoint"):
                MODULE.materialize(parent / "source", parent)

    def test_existing_overlay_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "flashinfer-overlay"
            MODULE.materialize(REFERENCE, output)
            target = output / MODULE.STATIC_REL
            target.write_text(target.read_text() + "\n# drift\n")
            with self.assertRaisesRegex(MODULE.OverlayContractError, "drift"):
                MODULE.materialize(REFERENCE, output)


if __name__ == "__main__":
    unittest.main()
