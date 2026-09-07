#!/usr/bin/env python3
"""CPU tests for the Qwen3.8 linear FP8 overlay contract."""

import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("qwen38-linear-fp8-overlay-contract.py")
SPEC = importlib.util.spec_from_file_location("contract", PATH)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def fixtures():
    headers, telemetry = {}, {}
    for layer in range(36):
        for projection in contract.PROJECTIONS:
            name = f"model.language_model.layers.{layer}.linear_attn.{projection}.weight"
            headers[name] = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}
            channel = contract.input_channel(layer, projection)
            telemetry[channel] = {"absmax": float(layer + 1)}
    trace = {
        "schema": contract.TRACE_SCHEMA,
        "coverage": dict(contract.EXPECTED_COVERAGE),
        "telemetry": telemetry,
        "input_sha256": "0" * 64,
    }
    modelopt = "\n".join(contract.DENSE_FP8_ABI_MARKERS)
    return headers, trace, modelopt


class ContractTest(unittest.TestCase):
    def test_stock_loader_blocks_fake_overlay(self):
        headers, trace, modelopt = fixtures()
        result = contract.build_contract(headers, trace, modelopt, "stock loader")
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["loader_abi"]["dense_fp8_valid"])
        self.assertFalse(result["loader_abi"]["overlay_valid"])
        self.assertEqual(result["artifact"]["selected_matrices"], 180)

    def test_ready_requires_hashes_and_exact_overlay_abi(self):
        headers, trace, modelopt = fixtures()
        hashes = {name: "a" * 64 for name in headers}
        result = contract.build_contract(
            headers, trace, modelopt, contract.OVERLAY_ABI_MARKER, hashes
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["artifact"]["encoded_weight_bytes"], 720)
        self.assertAlmostEqual(result["tensors"][0]["input_scale"], 1 / 448)

    def test_missing_matrix_fails_closed(self):
        headers, trace, modelopt = fixtures()
        headers.pop(next(iter(headers)))
        with self.assertRaisesRegex(contract.ContractError, "179/180"):
            contract.build_contract(headers, trace, modelopt, "stock loader")

    def test_dtype_and_shape_drift_fail_closed(self):
        for field, value, message in (
            ("dtype", "F16", "dtype drift"),
            ("shape", [4], "shape drift"),
        ):
            headers, trace, modelopt = fixtures()
            headers[next(iter(headers))][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(contract.ContractError, message):
                contract.build_contract(headers, trace, modelopt, "stock loader")

    def test_missing_telemetry_fails_closed(self):
        headers, trace, modelopt = fixtures()
        trace["telemetry"].pop("layer.0.linear_attn.in_proj_qkvz.input")
        with self.assertRaisesRegex(contract.ContractError, "missing calibrated activation"):
            contract.build_contract(headers, trace, modelopt, "stock loader")


if __name__ == "__main__":
    unittest.main()
