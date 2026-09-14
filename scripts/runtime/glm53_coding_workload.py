#!/usr/bin/env python3
"""Validation and prompt materialization for the GLM-5.3 coding benchmark."""

import hashlib
import json
import pathlib
import re


def load_workload(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def validate(sessions):
    if len(sessions) < 16:
        raise ValueError("workload needs at least sixteen sessions")
    ids = [s["id"] for s in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate session id")
    hashes = [hashlib.sha256(s["turns"][0]["content"].encode()).hexdigest() for s in sessions]
    if len(hashes) != len(set(hashes)):
        raise ValueError("replicated prompt")
    for s in sessions:
        if len(s["turns"]) < 3 or not any(t["role"] == "tool" for t in s["turns"]):
            raise ValueError(f"{s['id']} lacks deterministic multi-turn tool history")
        files = re.findall(r"^FILE ([^ ]+) SHA256 ", s["turns"][0]["content"], re.MULTILINE)
        if len(files) != len(set(files)):
            raise ValueError(f"{s['id']} repeats a source block")
    return {"sessions": len(sessions), "distinct_prompts": len(set(hashes))}


def useful_tokens(result):
    generated = result["generated_token_ids"]
    count = sum(len(stream) for stream in generated)
    if count != result["useful_output_tokens"]:
        raise ValueError("useful token count includes padding or rejected drafts")
    return count


def materialize_phase(sessions, phase, batch, out_dir):
    selected = sessions[phase * 16:phase * 16 + batch]
    if len(selected) != batch:
        raise ValueError("phase does not contain requested batch")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in selected:
        text = "\n".join(f"<{t['role']}>\n{t['content']}" for t in s["turns"])
        path = out_dir / f"{s['id']}.txt"
        path.write_text(text)
        paths.append(path)
    list_path = out_dir / "prompts.list"
    list_path.write_text("".join(str(p.resolve()) + "\n" for p in paths))
    return selected, list_path
