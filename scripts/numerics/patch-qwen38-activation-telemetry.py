#!/usr/bin/env python3
"""Add opt-in Qwen3.8 linear-attention activation telemetry to vLLM."""

import argparse
from pathlib import Path

IMPORT_ANCHOR = "from itertools import islice\n\nimport torch\n"
IMPORT_PATCH = "from itertools import islice\nimport os\n\nimport torch\n"
CLASS_ANCHOR = "class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):\n"
HELPER = '''_ROCKET_CALIBRATION_MAXIMA = {}\n\n\n+def _rocket_install_activation_telemetry(model):\n+    if os.getenv("ROCKET_NVFP4_CALIBRATE") != "1":\n+        return\n+    for layer_index, layer in enumerate(model.layers):\n+        attention = getattr(layer, "linear_attn", None)\n+        if attention is None:\n+            continue\n+        for projection_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):\n+            projection = getattr(attention, projection_name)\n+            key = f"layer.{layer_index}.linear_attn.{projection_name}"\n+            def capture(module, inputs, name=key):\n+                del module\n+                value = float(inputs[0].detach().float().abs().amax().item())\n+                if value > _ROCKET_CALIBRATION_MAXIMA.get(name, 0.0):\n+                    _ROCKET_CALIBRATION_MAXIMA[name] = value\n+                    print(f"ROCKET_NVFP4_CALIBRATION\\t{name}\\t{value:.9g}", flush=True)\n+            projection.register_forward_pre_hook(capture)\n+\n+\n+'''.replace("\n+", "\n")
INIT_ANCHOR = "        enable_qwen38next_low_latency_gemm(self, self.model_config.dtype)\n"
INIT_PATCH = INIT_ANCHOR + "        _rocket_install_activation_telemetry(self.model)\n"


def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise SystemExit(f"vLLM source drift: expected one {label} anchor")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_py", type=Path)
    args = parser.parse_args()
    source = args.model_py.read_text()
    if "ROCKET_NVFP4_CALIBRATION" in source:
        raise SystemExit("already patched")
    source = replace_once(source, IMPORT_ANCHOR, IMPORT_PATCH, "import")
    source = replace_once(source, CLASS_ANCHOR, HELPER + CLASS_ANCHOR, "class")
    source = replace_once(source, INIT_ANCHOR, INIT_PATCH, "initialization")
    args.model_py.write_text(source)


if __name__ == "__main__":
    main()
