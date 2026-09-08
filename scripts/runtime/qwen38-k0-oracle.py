#!/usr/bin/env python3
"""Prepare, invoke, and validate the fixed Qwen3.8 K0 oracle request."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


PROMPT = "Write a Python function `is_prime(n: int) -> bool` that handles integers below 2 and uses trial division up to the square root. Return only the function."
SCHEMA = "rocket.qwen38.k0-target-oracle-request.v1"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(model_dir: Path, output: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    token_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    if not token_ids or tokenizer.decode(token_ids) != PROMPT:
        raise SystemExit("fixed prompt tokenizer round trip failed")
    files = {}
    for name in TOKENIZER_FILES:
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"tokenizer identity file missing: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    record = {
        "schema": SCHEMA,
        "prompt": PROMPT,
        "input_token_ids": token_ids,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files": files,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def invoke(endpoint: str, request_path: Path, output: Path, arm: Path) -> None:
    contract = json.loads(request_path.read_text())
    if contract.get("schema") != SCHEMA or contract.get("prompt") != PROMPT:
        raise SystemExit("oracle request contract mismatch")
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "prompt": PROMPT,
        "max_tokens": 1,
        "temperature": 0,
        "seed": 0,
    }).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    arm.write_text("one target request\n")
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate(request_path: Path, capture_dir: Path, response_path: Path, output: Path) -> None:
    request = json.loads(request_path.read_text())
    manifest_path = capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("oracle manifest missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rocket.qwen38.k0-target-oracle.v1":
        raise SystemExit("oracle manifest schema mismatch")
    if manifest.get("valid") is not True or manifest.get("complete") is not True:
        raise SystemExit("oracle manifest is invalid or incomplete")
    if manifest.get("input_token_ids") != request.get("input_token_ids"):
        raise SystemExit("oracle input token IDs mismatch")
    expected_names = ["embedding", *[f"layer.{i:02d}" for i in range(48)], "final_norm", "logits"]
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
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0].get("text"), str):
        raise SystemExit("oracle response does not contain one detokenized choice")
    result = {
        "schema": "rocket.qwen38.k0-target-oracle-result.v1",
        "valid": True,
        "complete": True,
        "prompt": request["prompt"],
        "input_token_ids": request["input_token_ids"],
        "tokenizer": {key: request[key] for key in ("tokenizer_class", "tokenizer_files")},
        "greedy_token_id": manifest["greedy_token_id"],
        "detokenized_output": choices[0]["text"],
        "top_k": manifest["top_k"],
        "capture_manifest_sha256": sha256(manifest_path),
        "response_sha256": sha256(response_path),
        "identity": manifest["identity"],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--model-dir", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
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
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.model_dir, args.output)
    elif args.command == "invoke":
        invoke(args.endpoint, args.request, args.output, args.arm)
    else:
        validate(args.request, args.capture_dir, args.response, args.output)


if __name__ == "__main__":
    main()
