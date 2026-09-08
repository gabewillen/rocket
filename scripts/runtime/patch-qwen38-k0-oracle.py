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
        arm = json.loads((self.output.parent / "ARMED").read_text())
        self.decision_forwards = int(arm.get("decision_forwards", 1))
        self.decode_mode = self.decision_forwards == 8
        expected_arm = {"schema", "request_sha256", "generation_index"}
        if getattr(self, "decode_mode", False):
            expected_arm.add("decision_forwards")
        if set(arm) != expected_arm or self.decision_forwards not in (1, 8):
            raise RuntimeError("oracle arm schema differs")
        if arm["schema"] != "rocket.qwen38.k0-target-oracle-arm.v1":
            raise RuntimeError("oracle arm schema differs")
        if arm["request_sha256"] != self.identity.get("request_sha256"):
            raise RuntimeError("oracle request identity differs")
        if arm["generation_index"] != 0:
            raise RuntimeError("oracle generation identity differs")
        self.request_sha256 = arm["request_sha256"]
        self.generation_index = arm["generation_index"]
        self.schema = (
            "rocket.qwen38.k0-target-decode-oracle.v1"
            if self.decode_mode else type(self).schema
        )
        self.eos_token_ids = set(json.loads(os.environ.get("ROCKET_QWEN38_K0_EOS_IDS", "[]")))
        self.artifacts = []
        self.artifact_by_name = {}
        self.expected_forward_names = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm"]
        self.consumed_tokens = 0
        self.active_forward = False
        self.forward_names = []
        self.complete = False
        self.phase = 0
        self.generations = []

    def phase_name(self):
        return "prefill" if self.phase == 0 else f"decode.{self.phase:02d}"

    def artifact_name(self, name):
        return f"{self.phase_name()}.{name}" if getattr(self, "decode_mode", False) else name

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

    def append(self, name, tensor):
        value = tensor.detach().contiguous()
        payload = value.view(torch.uint8).cpu().numpy().tobytes()
        filename = name.replace(".", "-") + ".bin"
        path = self.output / filename
        previous = self.artifact_by_name.get(name)
        if previous is None:
            temporary = self.output / ("." + filename + ".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            item = {
                "name": name,
                "file": filename,
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
                "strides": list(value.stride()),
                "numel": value.numel(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            self.artifacts.append(item)
            self.artifact_by_name[name] = item
            return
        if previous["dtype"] != str(value.dtype).removeprefix("torch.") or previous["shape"][1:] != list(value.shape)[1:] or list(value.stride())[-1:] != [1]:
            raise RuntimeError(f"oracle chunk layout differs: {name}")
        with path.open("ab") as stream:
            stream.write(payload)
        previous["shape"][0] += value.shape[0]
        previous["strides"] = [previous["shape"][1], 1]
        previous["numel"] += value.numel()
        previous["bytes"] += len(payload)
        previous["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def save(self, name, tensor):
        if not self.active_forward:
            raise RuntimeError(f"oracle {name} arrived before authenticated request begin")
        if self.complete:
            raise RuntimeError(f"oracle {name} arrived after exactly-once consumption")
        index = len(self.forward_names)
        expected = self.expected_forward_names[index] if index < len(self.expected_forward_names) else None
        if name != expected:
            raise RuntimeError(f"oracle forward artifact order differs: expected {expected}, got {name}")
        self.append(self.artifact_name(name), tensor)
        self.forward_names.append(name)

    def begin(self, input_ids, embedding):
        if self.complete:
            raise RuntimeError("oracle received a second generation after exactly-once consumption")
        if self.active_forward:
            raise RuntimeError("oracle previous authenticated forward is missing logits")
        actual = input_ids.detach().cpu().tolist()
        if self.consumed_tokens < len(self.expected_ids):
            expected = self.expected_ids[self.consumed_tokens:self.consumed_tokens + len(actual)]
        elif self.decode_mode and self.generations and self.phase < self.decision_forwards:
            expected = [self.generations[-1]["token_id"]]
            if len(actual) != 1:
                raise RuntimeError("decode oracle requires one input token per post-prefill forward")
        else:
            raise RuntimeError("oracle received a second generation after exactly-once consumption")
        if not actual or actual != expected:
            raise RuntimeError(f"input token IDs differ at offset {self.consumed_tokens}: expected {expected}, got {actual}")
        if embedding.ndim != 2 or embedding.shape[0] != len(actual):
            raise RuntimeError("oracle embedding extent differs from authenticated request")
        self.active_forward = True
        self.forward_names = []
        if self.consumed_tokens < len(self.expected_ids):
            self.consumed_tokens += len(actual)
        self.save("embedding", embedding)

    def finish(self, logits):
        if not self.active_forward or self.complete:
            raise RuntimeError("oracle logits arrived outside the one target forward")
        if self.forward_names != self.expected_forward_names:
            raise RuntimeError(f"oracle forward ended before all boundaries: {self.forward_names}")
        self.active_forward = False
        if self.consumed_tokens < len(self.expected_ids):
            return
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("oracle logits generation extent differs")
        self.append(self.artifact_name("logits"), logits)
        last = logits[-1].detach().float()
        values, indices = torch.topk(last, min(20, last.numel()), sorted=True)
        top_k = [
            {"token_id": int(index), "logit": float(value)}
            for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]
        token_id = int(indices[0])
        if getattr(self, "decode_mode", False):
            if token_id in self.eos_token_ids:
                raise RuntimeError(f"decode oracle encountered EOS at decision {self.phase}")
            self.generations.append({
                "request_sha256": self.request_sha256,
                "generation_index": self.generation_index,
                "decision_index": self.phase,
                "kind": "prefill" if self.phase == 0 else "decode",
                "input_token_ids": (
                    self.expected_ids if self.phase == 0 else [self.generations[-1]["token_id"]]
                ),
                "token_id": token_id,
                "top_k": top_k,
            })
            if len(self.generations) < self.decision_forwards:
                self.phase += 1
                return
        manifest = {
            "schema": self.schema,
            "valid": True,
            "complete": True,
            "identity": self.identity,
            "request_sha256": self.request_sha256,
            "generation_index": self.generation_index,
            "input_token_ids": self.expected_ids,
            "artifacts": self.artifacts,
            "top_k": top_k,
            "greedy_token_id": token_id,
        }
        boundary_names = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm", "logits"]
        expected = boundary_names
        if getattr(self, "decode_mode", False):
            manifest.update({
                "decision_forwards": self.decision_forwards,
                "post_prefill_decode_forwards": self.decision_forwards - 1,
                "generations": self.generations,
            })
            expected = [
                f"{'prefill' if phase == 0 else f'decode.{phase:02d}'}.{name}"
                for phase in range(self.decision_forwards) for name in boundary_names
            ]
        names = [item["name"] for item in self.artifacts]
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
        "completed": sorted(path.stem for path in output.glob("*.bin"))[:408],
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
        "        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:\n            return inputs_embeds\n",
        "        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:\n"
        "            _rocket_k0_guard(\"embedding\", lambda oracle: oracle.begin(input_ids, inputs_embeds))\n"
        "            return inputs_embeds\n",
        "external embedding lifecycle",
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
