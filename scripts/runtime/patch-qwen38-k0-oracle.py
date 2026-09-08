#!/usr/bin/env python3
"""Patch the pinned Qwen3.8 target model with a one-shot K0 oracle capture."""

import argparse
from pathlib import Path


IMPORT_OLD = "from itertools import islice\n\nimport torch\n"
IMPORT_NEW = """from itertools import islice
import hashlib
import json
import os
from pathlib import Path

import torch
"""
CLASS_ANCHOR = "@support_torch_compile(\n    dynamic_arg_dims={\n        \"input_ids\": 0,\n"
HELPER = r'''_ROCKET_K0_ORACLE = None


class _RocketK0Oracle:
    schema = "rocket.qwen38.k0-target-oracle.v1"

    def __init__(self):
        self.output = Path(os.environ["ROCKET_QWEN38_K0_ORACLE_DIR"])
        self.output.mkdir(parents=True, exist_ok=False)
        self.expected_ids = json.loads(os.environ["ROCKET_QWEN38_K0_EXPECTED_IDS"])
        self.identity = json.loads(os.environ["ROCKET_QWEN38_K0_IDENTITY"])
        self.artifacts = []
        self.started = False
        self.complete = False

    def fail(self, phase, error):
        record = {
            "schema": self.schema,
            "valid": False,
            "complete": False,
            "phase": phase,
            "reason": f"{type(error).__name__}: {error}"[:1024],
            "identity": self.identity,
            "completed": [item["name"] for item in self.artifacts],
        }
        temporary = self.output / ".failure.json.tmp"
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, self.output / "failure.json")
        print("ROCKET_QWEN38_K0_ORACLE_FAILURE\t" + json.dumps(record, sort_keys=True), flush=True)

    def save(self, name, tensor):
        if any(item["name"] == name for item in self.artifacts):
            raise RuntimeError(f"duplicate oracle artifact: {name}")
        value = tensor.detach().contiguous()
        payload = value.view(torch.uint8).cpu().numpy().tobytes()
        filename = name.replace(".", "-") + ".bin"
        temporary = self.output / ("." + filename + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, self.output / filename)
        self.artifacts.append({
            "name": name,
            "file": filename,
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "strides": list(value.stride()),
            "numel": value.numel(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    def begin(self, input_ids, embedding):
        if self.started or self.complete:
            raise RuntimeError("oracle expected exactly one target forward")
        actual = input_ids.detach().cpu().tolist()
        if actual != self.expected_ids:
            raise RuntimeError(f"input token IDs differ: expected {self.expected_ids}, got {actual}")
        self.started = True
        self.save("embedding", embedding)

    def finish(self, logits):
        if not self.started or self.complete:
            raise RuntimeError("oracle logits arrived outside the one target forward")
        self.save("logits", logits)
        last = logits[-1].detach().float()
        values, indices = torch.topk(last, min(20, last.numel()), sorted=True)
        manifest = {
            "schema": self.schema,
            "valid": True,
            "complete": True,
            "identity": self.identity,
            "input_token_ids": self.expected_ids,
            "artifacts": self.artifacts,
            "top_k": [
                {"token_id": int(index), "logit": float(value)}
                for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
            ],
            "greedy_token_id": int(indices[0]),
        }
        names = [item["name"] for item in self.artifacts]
        expected = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm", "logits"]
        if names != expected:
            raise RuntimeError(f"oracle artifact sequence differs: {names}")
        temporary = self.output / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.output / "manifest.json")
        self.complete = True
        print("ROCKET_QWEN38_K0_ORACLE_COMPLETE\t" + json.dumps({
            "schema": self.schema,
            "greedy_token_id": manifest["greedy_token_id"],
            "artifact_count": len(self.artifacts),
        }, sort_keys=True), flush=True)


def _rocket_k0_oracle():
    global _ROCKET_K0_ORACLE
    if os.getenv("ROCKET_QWEN38_K0_ORACLE") != "1":
        return None
    output = Path(os.environ["ROCKET_QWEN38_K0_ORACLE_DIR"])
    if not (output.parent / "ARMED").is_file():
        return None
    from vllm.distributed import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank() != 0:
        return None
    if _ROCKET_K0_ORACLE is None:
        _ROCKET_K0_ORACLE = _RocketK0Oracle()
    return _ROCKET_K0_ORACLE


def _rocket_k0_emergency_failure(phase, error):
    output = Path(os.environ["ROCKET_QWEN38_K0_ORACLE_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": _RocketK0Oracle.schema,
        "valid": False,
        "complete": False,
        "phase": phase,
        "reason": f"{type(error).__name__}: {error}"[:1024],
        "identity": json.loads(os.environ.get("ROCKET_QWEN38_K0_IDENTITY", "{}")),
        "completed": sorted(path.stem for path in output.glob("*.bin"))[:51],
    }
    temporary = output / ".failure.json.tmp"
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, output / "failure.json")
    print("ROCKET_QWEN38_K0_ORACLE_FAILURE\t" + json.dumps(record, sort_keys=True), flush=True)


def _rocket_k0_guard(phase, callback):
    try:
        oracle = _rocket_k0_oracle()
    except Exception as error:
        _rocket_k0_emergency_failure(phase, error)
        raise
    if oracle is None:
        return
    try:
        callback(oracle)
    except Exception as error:
        oracle.fail(phase, error)
        raise


'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"{label} anchor count is {source.count(old)}, expected 1")
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source = path.read_text()
    source = replace_once(source, IMPORT_OLD, IMPORT_NEW, "import")
    source = replace_once(source, CLASS_ANCHOR, HELPER + CLASS_ANCHOR, "class decorator")
    source = replace_once(
        source,
        "                hidden_states = self.embed_input_ids(input_ids)\n            hidden_states = hidden_states.repeat(1, self.config.hc_count)\n",
        "                hidden_states = self.embed_input_ids(input_ids)\n"
        "                _rocket_k0_guard(\"embedding\", lambda oracle: oracle.begin(input_ids, hidden_states))\n"
        "            hidden_states = hidden_states.repeat(1, self.config.hc_count)\n",
        "embedding",
    )
    source = replace_once(
        source,
        "            if deepstack_input_embeds is not None and layer_idx < len(\n",
        "            _rocket_k0_guard(\n"
        "                f\"layer.{layer_idx:02d}\",\n"
        "                lambda oracle, layer=layer, hidden_states=hidden_states, block_output=block_output, injection=injection, layer_idx=layer_idx: oracle.save(\n"
        "                    f\"layer.{layer_idx:02d}\",\n"
        "                    layer.mlp_hyper_connection.combine(hidden_states, block_output, injection),\n"
        "                ),\n"
        "            )\n"
        "            if deepstack_input_embeds is not None and layer_idx < len(\n",
        "layer",
    )
    source = replace_once(
        source,
        "        return sample_hidden_states\n",
        "        _rocket_k0_guard(\"final_norm\", lambda oracle: oracle.save(\"final_norm\", sample_hidden_states))\n"
        "        return sample_hidden_states\n",
        "final norm",
    )
    source = replace_once(
        source,
        "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n        return self.logits_processor(self.lm_head, hidden_states)\n",
        "    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:\n"
        "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
        "        if logits is not None:\n"
        "            _rocket_k0_guard(\"logits\", lambda oracle: oracle.finish(logits))\n"
        "        return logits\n",
        "logits",
    )
    path.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    patch(args.model)


if __name__ == "__main__":
    main()
