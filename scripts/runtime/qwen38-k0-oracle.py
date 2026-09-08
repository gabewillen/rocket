#!/usr/bin/env python3
"""Prepare, invoke, and validate the fixed Qwen3.8 K0 oracle request."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


PROMPT = "Write a Python function `is_prime(n: int) -> bool` that handles integers below 2 and uses trial division up to the square root. Return only the function."
SCHEMA = "rocket.qwen38.k0-target-oracle-request.v1"
DECODE_SCHEMA = "rocket.qwen38.k0-target-decode-oracle-request.v1"
DECODE_DECISIONS = 8
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(model_dir: Path, output: Path, decode_decisions: int = 1) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    messages = [{"role": "user", "content": PROMPT}]
    if decode_decisions == 1:
        token_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
        if not token_ids or tokenizer.decode(token_ids) != PROMPT:
            raise SystemExit("fixed prompt tokenizer round trip failed")
    elif decode_decisions == DECODE_DECISIONS:
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(token_ids, list):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise SystemExit("fixed chat template produced a batched prompt")
            token_ids = token_ids[0]
        if not token_ids:
            raise SystemExit("fixed chat template produced no prompt tokens")
    else:
        raise SystemExit("decode oracle requires exactly eight decision forwards")
    files = {}
    for name in TOKENIZER_FILES:
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"tokenizer identity file missing: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    record = {
        "schema": SCHEMA if decode_decisions == 1 else DECODE_SCHEMA,
        "prompt": PROMPT,
        "input_token_ids": token_ids,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files": files,
    }
    if decode_decisions > 1:
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise SystemExit("tokenizer chat template is absent")
        record.update({
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "decision_forwards": decode_decisions,
            "post_prefill_decode_forwards": decode_decisions - 1,
            "eos_token_ids": sorted({
                int(value) for value in (
                    tokenizer.eos_token_id,
                    *(getattr(tokenizer, "additional_special_tokens_ids", []) or []),
                ) if value is not None and tokenizer.convert_ids_to_tokens(value) == "<|im_end|>"
            }),
        })
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def invoke(endpoint: str, request_path: Path, output: Path, arm: Path) -> None:
    contract = json.loads(request_path.read_text())
    if contract.get("schema") not in (SCHEMA, DECODE_SCHEMA) or contract.get("prompt") != PROMPT:
        raise SystemExit("oracle request contract mismatch")
    decode = contract["schema"] == DECODE_SCHEMA
    body_record = {
        "model": "qwen3.8-flash-next", "temperature": 0, "seed": 0,
        "max_tokens": DECODE_DECISIONS if decode else 1,
    }
    body_record.update(
        {"messages": contract["messages"], "return_token_ids": True}
        if decode else {"prompt": PROMPT}
    )
    body = json.dumps(body_record).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + ("/v1/chat/completions" if decode else "/v1/completions"),
        data=body,
        headers={"Content-Type": "application/json"},
    )
    arm_record = {
        "schema": "rocket.qwen38.k0-target-oracle-arm.v1",
        "request_sha256": sha256(request_path),
        "generation_index": 0,
    }
    if decode:
        arm_record["decision_forwards"] = DECODE_DECISIONS
    arm.write_text(json.dumps(arm_record, sort_keys=True) + "\n")
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate(request_path: Path, capture_dir: Path, response_path: Path, output: Path,
             model_dir: Path | None = None) -> None:
    request = json.loads(request_path.read_text())
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("oracle manifest missing")
    manifest = json.loads(manifest_path.read_text())
    decode = request.get("schema") == DECODE_SCHEMA
    expected_manifest_schema = (
        "rocket.qwen38.k0-target-decode-oracle.v1" if decode
        else "rocket.qwen38.k0-target-oracle.v1"
    )
    if manifest.get("schema") != expected_manifest_schema:
        raise SystemExit("oracle manifest schema mismatch")
    if manifest.get("valid") is not True or manifest.get("complete") is not True:
        raise SystemExit("oracle manifest is invalid or incomplete")
    if manifest.get("input_token_ids") != request.get("input_token_ids"):
        raise SystemExit("oracle input token IDs mismatch")
    request_sha256 = sha256(request_path)
    if manifest.get("request_sha256") != request_sha256:
        raise SystemExit("oracle request identity mismatch")
    if manifest.get("generation_index") != 0:
        raise SystemExit("oracle generation identity mismatch")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("request_sha256") != request_sha256 or identity.get("generation_index") != 0:
        raise SystemExit("oracle manifest identity mismatch")
    boundary_names = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm", "logits"]
    expected_names = boundary_names if not decode else [
        f"{'prefill' if phase == 0 else f'decode.{phase:02d}'}.{name}"
        for phase in range(DECODE_DECISIONS) for name in boundary_names
    ]
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or [item.get("name") for item in artifacts] != expected_names:
        raise SystemExit("oracle artifact sequence mismatch")
    artifact_keys = {"name", "file", "dtype", "shape", "strides", "numel", "bytes", "sha256"}
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}
    for item in artifacts:
        if set(item) != artifact_keys:
            raise SystemExit(f"oracle artifact schema mismatch: {item.get('name')}")
        shape = item["shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(not isinstance(value, int) or value <= 0 for value in shape):
            raise SystemExit(f"oracle artifact shape mismatch: {item['name']}")
        strides = [shape[1], 1]
        if item["strides"] != strides or item["numel"] != shape[0] * shape[1]:
            raise SystemExit(f"oracle artifact layout mismatch: {item['name']}")
        if item["dtype"] not in dtype_bytes or item["bytes"] != item["numel"] * dtype_bytes[item["dtype"]]:
            raise SystemExit(f"oracle artifact dtype/byte mismatch: {item['name']}")
    token_count = len(request["input_token_ids"])
    hidden_size = artifacts[0]["shape"][1]
    if artifacts[0]["shape"] != [token_count, hidden_size]:
        raise SystemExit("oracle embedding extent mismatch")
    layer_shape = artifacts[1]["shape"]
    if layer_shape[0] != token_count or layer_shape[1] % hidden_size != 0:
        raise SystemExit("oracle post-layer extent mismatch")
    if any(item["shape"] != layer_shape for item in artifacts[1:49]):
        raise SystemExit("oracle post-layer extents differ")
    if artifacts[49]["shape"] != [token_count, hidden_size]:
        raise SystemExit("oracle final norm extent mismatch")
    if artifacts[50]["shape"][0] != 1:
        raise SystemExit("oracle logits must contain exactly one sampled position")
    if decode:
        for phase in range(1, DECODE_DECISIONS):
            base = phase * len(boundary_names)
            if artifacts[base]["shape"] != [1, hidden_size]:
                raise SystemExit("decode embedding extent mismatch")
            if any(item["shape"] != [1, layer_shape[1]] for item in artifacts[base + 1:base + 49]):
                raise SystemExit("decode post-layer extent mismatch")
            if artifacts[base + 49]["shape"] != [1, hidden_size] or artifacts[base + 50]["shape"][0] != 1:
                raise SystemExit("decode final boundary extent mismatch")
    expected_files = {"manifest.json"}
    for item in artifacts:
        path = capture_dir / item["file"]
        expected_files.add(item["file"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise SystemExit(f"oracle artifact identity mismatch: {item['name']}")
    actual_files = {path.name for path in capture_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise SystemExit(f"oracle capture has missing or extra files: {sorted(actual_files ^ expected_files)}")
    response = json.loads(response_path.read_text())
    choices = response.get("choices")
    text_key = "content" if decode else "text"
    choice_text = choices[0].get("message", {}).get(text_key) if decode and isinstance(choices, list) and len(choices) == 1 else (choices[0].get(text_key) if isinstance(choices, list) and len(choices) == 1 else None)
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choice_text, str):
        raise SystemExit("oracle response does not contain one detokenized choice")
    if decode:
        token_ids = choices[0].get("token_ids")
        generations = manifest.get("generations")
        if not isinstance(token_ids, list) or len(token_ids) != DECODE_DECISIONS:
            raise SystemExit("decode oracle returned fewer than eight tokens")
        if not isinstance(generations, list) or len(generations) != DECODE_DECISIONS:
            raise SystemExit("decode oracle captured fewer than eight decision forwards")
        if token_ids != [item.get("token_id") for item in generations]:
            raise SystemExit("decode oracle response/capture token IDs differ")
        for index, generation in enumerate(generations):
            if (
                generation.get("request_sha256") != request_sha256
                or generation.get("generation_index") != 0
                or generation.get("decision_index") != index
                or generation.get("kind") != ("prefill" if index == 0 else "decode")
            ):
                raise SystemExit("decode oracle phase identity differs")
        if any(token in request.get("eos_token_ids", []) for token in token_ids):
            raise SystemExit("decode oracle encountered EOS before eight decisions")
        if model_dir is None:
            raise SystemExit("decode oracle validation requires the pinned tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != request["chat_template_sha256"]:
            raise SystemExit("decode oracle chat template identity changed")
        cumulative = []
        for index, generation in enumerate(generations):
            generation["token_text"] = tokenizer.decode(
                [token_ids[index]], skip_special_tokens=False
            )
            cumulative.append(token_ids[index])
            generation["cumulative_text"] = tokenizer.decode(
                cumulative, skip_special_tokens=False
            )
        if tokenizer.decode(token_ids, skip_special_tokens=True) != choice_text:
            raise SystemExit("decode oracle token/text detokenization differs")
    result = {
        "schema": (
            "rocket.qwen38.k0-target-decode-oracle-result.v1" if decode
            else "rocket.qwen38.k0-target-oracle-result.v1"
        ),
        "valid": True,
        "complete": True,
        "prompt": request["prompt"],
        "input_token_ids": request["input_token_ids"],
        "tokenizer": {key: request[key] for key in ("tokenizer_class", "tokenizer_files")},
        "greedy_token_id": manifest["greedy_token_id"],
        "detokenized_output": choice_text,
        "top_k": manifest["top_k"],
        "capture_manifest_sha256": sha256(manifest_path),
        "response_sha256": sha256(response_path),
        "identity": identity,
    }
    if decode:
        result.update({
            "decision_forwards": DECODE_DECISIONS,
            "post_prefill_decode_forwards": DECODE_DECISIONS - 1,
            "generated_token_ids": token_ids,
            "generations": generations,
            "chat_template_sha256": request["chat_template_sha256"],
        })
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--model-dir", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--decode-decisions", type=int, default=1)
    invoke_parser = subparsers.add_parser("invoke")
    invoke_parser.add_argument("--endpoint", required=True)
    invoke_parser.add_argument("--request", type=Path, required=True)
    invoke_parser.add_argument("--output", type=Path, required=True)
    invoke_parser.add_argument("--arm", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--request", type=Path, required=True)
    validate_parser.add_argument("--capture-dir", type=Path, required=True)
    validate_parser.add_argument("--response", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model_dir, args.output, args.decode_decisions)
    elif args.command == "invoke":
        invoke(args.endpoint, args.request, args.output, args.arm)
    else:
        validate(args.request, args.capture_dir, args.response, args.output, args.model_dir)


if __name__ == "__main__":
    main()
