#!/usr/bin/env python3
"""Build the deterministic concurrent coding-session workload."""

import argparse
import hashlib
import json
import pathlib
import subprocess

LANGUAGES = ["python", "c++", "cuda", "rust", "javascript", "shell"]
TASKS = ["comprehension", "bug-diagnosis", "patch-generation", "test-repair", "code-review", "refactoring"]
CONTEXT_TOKENS = [2048, 8192, 32768, 65536]
RESPONSE_TOKENS = [128, 256, 512, 1024]


TEMPLATES = [
    "Inspect the implementation and identify the first correctness risk before proposing a patch.",
    "Diagnose the failing behavior from the code and produce the smallest tested fix.",
    "Implement the requested change, preserve existing contracts, and include focused tests.",
    "Repair the failing tests without weakening assertions or bypassing production behavior.",
    "Review the change for correctness, security, numerical drift, and missing tests.",
    "Refactor the hot path to remove duplicated work while preserving output behavior.",
]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def source_excerpt(root: pathlib.Path, index: int) -> str:
    candidates = [
        root / "engines/glm5-moe-nvfp4-2b/src/model.cu",
        root / "engines/glm5-moe-nvfp4-2b/src/kernels.cu",
        root / "engines/glm5-moe-nvfp4-2b/src/dflash2_engine.cu",
        root / "engines/glm5-moe-nvfp4-2b/src/kv/nvme_prefix_store.cc",
        root / "scripts/runtime/glm53-nvfp4-spec-bench.sh",
        root / "scripts/numerics/glm53-teacher-score.py",
        root / "scripts/runtime/fixtures/coding/rust.rs",
        root / "scripts/runtime/fixtures/coding/javascript.js",
    ]
    path = candidates[index % len(candidates)]
    text = path.read_text(errors="replace")
    start = (index * 7919) % max(1, len(text) - 12000)
    return f"FILE {path.relative_to(root)}\n{text[start:start + 12000]}"


def expand_context(seed: str, target_tokens: int) -> str:
    target_chars = target_tokens * 3
    blocks = []
    counter = 0
    while sum(map(len, blocks)) < target_chars:
        digest = hashlib.sha256(f"{target_tokens}:{counter}:{seed}".encode()).hexdigest()
        blocks.append(f"\nCONTEXT BLOCK {counter} CHECKSUM {digest}\n{seed}\n")
        counter += 1
    return "".join(blocks)[:target_chars]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    commit = git("-C", str(root), "rev-parse", "HEAD")
    sessions = []
    for i in range(64):
        phase = i // 16
        language = LANGUAGES[i % len(LANGUAGES)]
        task = TASKS[i % len(TASKS)]
        context_tokens = CONTEXT_TOKENS[phase]
        response_tokens = RESPONSE_TOKENS[phase]
        excerpt = source_excerpt(root, phase)
        shared = (
            "You are working in the public Rocket repository. Return only a concise engineering response. "
            "Do not invent test results. Preserve numerical and memory-safety contracts.\n"
        )
        unique = f"SESSION {i:02d} LANGUAGE {language} TASK {task}. {TEMPLATES[i % len(TEMPLATES)]}\n"
        context = expand_context(excerpt, context_tokens)
        turns = [
            {"role": "user", "content": shared + context + "\n" + unique},
            {"role": "tool", "content": f"tool_result session={i} phase=inspect checksum={hashlib.sha256(excerpt.encode()).hexdigest()}"},
            {"role": "user", "content": f"Continue session {i:02d}. State the measured bottleneck and the next concrete edit."},
        ]
        sessions.append({
            "id": f"coding-{i:02d}",
            "language": language,
            "task_class": task,
            "context_tokens_target": context_tokens,
            "response_tokens": response_tokens,
            "arrival_ms": (i % 8) * 125,
            "turns": turns,
        })
    hashes = [hashlib.sha256(x["turns"][0]["content"].encode()).hexdigest() for x in sessions]
    phase_c8 = [x for phase in range(4) for x in sessions[phase * 16:phase * 16 + 8]]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("workload prompts are not distinct")
    if sum(x["response_tokens"] for x in phase_c8) < 4096:
        raise RuntimeError("c8 useful-token budget is below 4096")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in sessions))
    manifest = {
        "schema": "rocket.glm53.coding-workload.v1",
        "source": "https://github.com/gabewillen/rocket",
        "source_commit": commit,
        "session_count": len(sessions),
        "distinct_prompt_sha256": hashes,
        "phase_size": 16,
        "c8_useful_output_tokens": sum(x["response_tokens"] for x in phase_c8),
        "context_token_targets": CONTEXT_TOKENS,
        "response_token_budgets": RESPONSE_TOKENS,
        "workload_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
