#!/usr/bin/env python3
"""Run deterministic workload phases through rocket-coding-bench."""

import argparse
import json
import os
import pathlib
import subprocess
import time
from glm53_coding_workload import load_workload, materialize_phase, useful_tokens, validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workload", type=pathlib.Path, required=True)
    p.add_argument("--batch", type=int, choices=(4, 8, 16), default=8)
    p.add_argument("--spec", type=int, default=7)
    p.add_argument("--phase", type=int, choices=range(4), action="append")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    sessions = load_workload(a.workload)
    summary = validate(sessions)
    phases = a.phase or list(range(4))
    a.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for phase in phases:
        selected, prompt_list = materialize_phase(sessions, phase, a.batch, a.out / f"phase-{phase}")
        replacement = None
        if a.batch < 16:
            replacement_session = sessions[phase * 16 + a.batch]
            replacement = a.out / f"phase-{phase}" / f"replacement-{replacement_session['id']}.txt"
            replacement.write_text("\n".join(
                f"<{t['role']}>\n{t['content']}" for t in replacement_session["turns"]))
        token_limit = selected[0]["context_tokens_target"]
        tokens = selected[0]["response_tokens"]
        result = a.out / f"phase-{phase}.json"
        cmd = [str(pathlib.Path(__file__).with_name("glm53-coding-bench-pair.sh"))]
        env = os.environ.copy()
        env.update(BATCH=str(a.batch), SPEC=str(a.spec), TOKENS=str(tokens),
                   PROMPT_TOKEN_LIMIT=str(token_limit), MAX_TOKENS=str(token_limit + tokens + 128),
                   PROMPT_LIST=str(prompt_list.resolve()), RESULT_JSON=str(result.resolve()),
                   PORT=str(18782 + phase))
        if replacement:
            env["REPLACEMENT_PROMPT"] = str(replacement.resolve())
        if not a.dry_run:
            start = time.monotonic()
            subprocess.run(cmd, env=env, check=True)
            data = json.loads(result.read_text())
            useful_tokens(data)
            data["phase"] = phase
            data["task_classes"] = [x["task_class"] for x in selected]
            data["languages"] = [x["language"] for x in selected]
            data["wall_ms"] = (time.monotonic() - start) * 1000
            result.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            runs.append(data)
        else:
            runs.append({"phase": phase, "batch": a.batch, "spec_k": a.spec,
                         "prompt_count": len(selected), "prompt_token_limit": token_limit,
                         "useful_output_tokens": a.batch * tokens})
    total = sum(x["useful_output_tokens"] for x in runs)
    final = {**summary, "batch": a.batch, "spec_k": a.spec, "runs": runs,
             "useful_output_tokens": total}
    (a.out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
