#!/usr/bin/env python3
"""Summarize or compare Rocket GLM-5.3 teacher-forced score traces."""

import argparse
import json
import math
import pathlib
import struct
from array import array

MAGIC = b"RKTLOG1\0"


def read_trace(path: pathlib.Path):
    metadata = None
    rows = []
    summary = None
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            kind = item.get("type")
            if kind == "metadata":
                metadata = item
            elif kind == "token":
                rows.append(item)
            elif kind == "summary":
                summary = item
            else:
                raise ValueError(f"{path}:{line_no}: unknown record type {kind!r}")
    if metadata is None or summary is None:
        raise ValueError(f"{path}: incomplete trace")
    if summary["tokens"] != len(rows):
        raise ValueError(f"{path}: summary token count does not match rows")
    return metadata, rows, summary


def read_logits(path: pathlib.Path):
    with path.open("rb") as handle:
        header = handle.read(20)
        if len(header) != 20:
            raise ValueError(f"{path}: truncated header")
        magic, vocab, rows = struct.unpack("<8sIQ", header)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic")
        values = array("f")
        values.fromfile(handle, rows * vocab)
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes")
    return vocab, rows, values


def trace_summary(rows, run_summary=None):
    nll = -sum(row["target_logprob"] for row in rows) / len(rows)
    result = {
        "tokens": len(rows),
        "mean_nll": nll,
        "perplexity": math.exp(nll),
        "target_top1_rate": sum(row["top5"][0]["id"] == row["target_id"] for row in rows) / len(rows),
        "target_top5_rate": sum(any(x["id"] == row["target_id"] for x in row["top5"]) for row in rows) / len(rows),
    }
    if run_summary:
        for key in ("model_ms", "model_tokens_per_second", "median_token_ms",
                    "wall_ms", "wall_tokens_per_second"):
            if key in run_summary:
                result[key] = run_summary[key]
    return result


def compare(base_rows, cand_rows, base_logits=None, cand_logits=None):
    if len(base_rows) != len(cand_rows):
        raise ValueError("trace token counts differ")
    top1 = top5 = first_divergence = 0
    target_drift = []
    for index, (base, cand) in enumerate(zip(base_rows, cand_rows)):
        key = (base["sequence"], base["position"], base["input_id"], base["target_id"])
        other = (cand["sequence"], cand["position"], cand["input_id"], cand["target_id"])
        if key != other:
            raise ValueError(f"trace alignment differs at row {index}")
        same_top1 = base["top5"][0]["id"] == cand["top5"][0]["id"]
        top1 += same_top1
        if not same_top1 and first_divergence == 0:
            first_divergence = index + 1
        top5 += len({x["id"] for x in base["top5"]} & {x["id"] for x in cand["top5"]}) / 5.0
        target_drift.append(abs(base["target_logprob"] - cand["target_logprob"]))
    result = {
        "tokens": len(base_rows),
        "baseline": trace_summary(base_rows),
        "candidate": trace_summary(cand_rows),
        "perplexity_ratio": trace_summary(cand_rows)["perplexity"] / trace_summary(base_rows)["perplexity"],
        "top1_agreement": top1 / len(base_rows),
        "mean_top5_overlap": top5 / len(base_rows),
        "mean_abs_target_logprob_drift": sum(target_drift) / len(target_drift),
        "max_abs_target_logprob_drift": max(target_drift),
        "first_top1_divergence_row": first_divergence or None,
    }
    if base_logits and cand_logits:
        bvocab, brows, bvals = base_logits
        cvocab, crows, cvals = cand_logits
        if (bvocab, brows) != (cvocab, crows) or brows != len(base_rows):
            raise ValueError("logit tensor shapes do not match traces")
        kl_sum = mse_sum = centered_mse_sum = 0.0
        count = brows * bvocab
        for row in range(brows):
            lo, hi = row * bvocab, (row + 1) * bvocab
            b = bvals[lo:hi]
            c = cvals[lo:hi]
            bm, cm = max(b), max(c)
            bz = sum(math.exp(x - bm) for x in b)
            cz = sum(math.exp(x - cm) for x in c)
            blse, clse = bm + math.log(bz), cm + math.log(cz)
            bmean = sum(b) / bvocab
            cmean = sum(c) / bvocab
            for bx, cx in zip(b, c):
                bp = math.exp(bx - blse)
                kl_sum += bp * ((bx - blse) - (cx - clse))
                mse_sum += (cx - bx) ** 2
                centered_mse_sum += ((cx - cmean) - (bx - bmean)) ** 2
        result.update({
            "mean_kl_divergence_nats": kl_sum / brows,
            "logit_rms_error": math.sqrt(mse_sum / count),
            "centered_logit_rms_error": math.sqrt(centered_mse_sum / count),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=pathlib.Path)
    parser.add_argument("--logits", type=pathlib.Path)
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--baseline-logits", type=pathlib.Path)
    parser.add_argument("--candidate-logits", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    if args.trace:
        _, rows, summary = read_trace(args.trace)
        result = trace_summary(rows, summary)
    elif args.baseline and args.candidate:
        _, base, base_summary = read_trace(args.baseline)
        _, cand, cand_summary = read_trace(args.candidate)
        blogits = read_logits(args.baseline_logits) if args.baseline_logits else None
        clogits = read_logits(args.candidate_logits) if args.candidate_logits else None
        if bool(blogits) != bool(clogits):
            parser.error("both baseline logit files are required for KL/RMS comparison")
        result = compare(base, cand, blogits, clogits)
        result["baseline"].update({k: base_summary[k] for k in
            ("model_ms", "model_tokens_per_second", "median_token_ms", "wall_ms", "wall_tokens_per_second")
            if k in base_summary})
        result["candidate"].update({k: cand_summary[k] for k in
            ("model_ms", "model_tokens_per_second", "median_token_ms", "wall_ms", "wall_tokens_per_second")
            if k in cand_summary})
    else:
        parser.error("use --trace or --baseline and --candidate")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
