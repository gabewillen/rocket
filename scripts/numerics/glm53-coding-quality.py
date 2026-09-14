#!/usr/bin/env python3
"""Deterministic format and code-signal checks for coding benchmark output."""

import argparse
import collections
import json
import pathlib
import re


def score_one(text, task_class):
    checks = {
        "nonempty": len(text.strip()) >= 32,
        "finite_text": "nan" not in text.lower() and "inf" not in text.lower(),
        "requested_format": not text.lstrip().startswith(("As an AI", "I cannot")),
    }
    if task_class in {"patch-generation", "test-repair", "refactoring"}:
        checks["code_signal"] = bool(re.search(r"```|\b(def|fn|void|class|if|for|return)\b", text))
    elif task_class == "bug-diagnosis":
        checks["diagnosis_signal"] = bool(re.search(r"\b(fail|bug|risk|cause|overflow|race|error)\b", text, re.I))
    elif task_class == "code-review":
        checks["review_signal"] = bool(re.search(r"\b(test|correct|safety|regress|issue|risk)\b", text, re.I))
    else:
        checks["comprehension_signal"] = len(text.split()) >= 8
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path)
    a = p.parse_args()
    run = json.loads(a.run.read_text())
    classes = run.get("task_classes", ["comprehension"] * len(run["generated_text"]))
    rows = []
    by_class = collections.defaultdict(list)
    for i, (text, cls) in enumerate(zip(run["generated_text"], classes)):
        checks = score_one(text, cls)
        passed = all(checks.values())
        rows.append({"stream": i, "task_class": cls, "passed": passed, "checks": checks})
        by_class[cls].append(passed)
    result = {
        "schema": "rocket.glm53.coding-quality.v1",
        "pass_rate": sum(x["passed"] for x in rows) / len(rows),
        "pass_rate_by_class": {k: sum(v) / len(v) for k, v in sorted(by_class.items())},
        "rows": rows,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.output: a.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
