#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import types
import unittest


SCRIPT = pathlib.Path(__file__).with_name("qwen38-gdn-prefill-phase.py")


def load_attribution_function():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "attribute_projection_backend"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace["attribute_projection_backend"]


class GdnPrefillPhaseContract(unittest.TestCase):
    def test_projection_result_names_concrete_selected_kernel(self) -> None:
        kernel_type = type(
            "FlashInferCutlassNvFp4LinearKernel",
            (),
            {"__module__": "vllm.model_executor.kernels.linear.nvfp4.flashinfer"},
        )
        result: dict[str, object] = {}
        method = types.SimpleNamespace(kernel=kernel_type())

        load_attribution_function()(result, method)

        self.assertEqual(
            result["nvfp4_kernel"],
            "vllm.model_executor.kernels.linear.nvfp4.flashinfer."
            "FlashInferCutlassNvFp4LinearKernel",
        )

    def test_both_projection_measurements_are_attributed(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "attribute_projection_backend"
        ]
        attributed_results = {
            call.args[0].id
            for call in calls
            if call.args and isinstance(call.args[0], ast.Name)
        }
        self.assertEqual(attributed_results, {"input_result", "output_result"})


if __name__ == "__main__":
    unittest.main()
