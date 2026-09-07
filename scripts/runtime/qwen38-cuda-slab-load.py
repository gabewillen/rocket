#!/usr/bin/env python3
"""Measure owner-local Qwen rank-slab load into its final CUDA byte layout."""

from __future__ import annotations

import argparse
import json
import shlex
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qwen38_slab.cuda_slab_loader import CudaRankSlabLoader
from qwen38_slab.loader import DirectSlabLoader

BASELINE_MODEL_READY_SECONDS = 193.803748
CHECKPOINT_BYTES = 132_639_846_394
VLLM_READ_BYTES = 242_288_619_520
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
CONTAINER_REPO = "/rocket"
SCRIPT = "scripts/runtime/qwen38-cuda-slab-load.py"
TOTAL_READY_STAGES = (
    "weights_loaded",
    "allocations_complete",
    "kernels_initialized",
    "graphs_captured",
    "scheduler_ready",
    "api_ready",
    "first_token_generated",
)


def total_cold_start_status(
    process_exec_started_ns: int,
    stage_completed_ns: dict[str, int],
    first_token_id: int | None,
) -> dict[str, object]:
    """Validate the total cold-start terminal boundary without inventing readiness."""

    if (
        isinstance(process_exec_started_ns, bool)
        or not isinstance(process_exec_started_ns, int)
        or process_exec_started_ns <= 0
        or set(stage_completed_ns) - set(TOTAL_READY_STAGES)
    ):
        raise ValueError("invalid total cold-start evidence")
    for stage, completed_ns in stage_completed_ns.items():
        if (
            isinstance(completed_ns, bool)
            or not isinstance(completed_ns, int)
            or completed_ns < process_exec_started_ns
        ):
            raise ValueError(f"invalid monotonic timestamp for {stage}")
    if first_token_id is not None and (
        isinstance(first_token_id, bool)
        or not isinstance(first_token_id, int)
        or first_token_id < 0
    ):
        raise ValueError("invalid first generated token")
    missing_values = [
        stage for stage in TOTAL_READY_STAGES if stage not in stage_completed_ns
    ]
    if first_token_id is None:
        missing_values.append("first_token_id")
    missing = tuple(missing_values)
    first_token_ns = stage_completed_ns.get("first_token_generated")
    complete = not missing
    if complete and first_token_ns != max(stage_completed_ns.values()):
        raise ValueError("first token is not the terminal cold-start event")
    return {
        "status": "complete" if complete else "incomplete_engine",
        "process_exec_started_ns": process_exec_started_ns,
        "first_token_completed_ns": first_token_ns if complete else None,
        "first_token_id": first_token_id if complete else None,
        "total_cold_load_ns": first_token_ns - process_exec_started_ns if complete else None,
        "missing_evidence": missing,
        "comparison_scope_match": complete,
    }


class Owner:
    def __init__(self):
        self.slabs = None
        self.publish_ns = None

    def publish_rank_slabs(self, rank, slabs):
        expected = {f"rank{rank}-target", f"rank{rank}-mtp"}
        if set(slabs) != expected or self.slabs is not None:
            raise RuntimeError("rank slab publication contract changed")
        self.slabs = slabs
        self.publish_ns = time.perf_counter_ns()


def stage_summary(receipt) -> dict[str, int]:
    chunks = receipt.chunks
    return {
        "chunks": len(chunks),
        "direct_read_ns_sum": sum(chunk.direct_read_ns for chunk in chunks),
        "direct_read_ns_max": max(chunk.direct_read_ns for chunk in chunks),
        "sha256_ns_sum": sum(chunk.sha256_ns for chunk in chunks),
        "sha256_ns_max": max(chunk.sha256_ns for chunk in chunks),
        "h2d_fence_ns_sum": sum(chunk.h2d_fence_ns for chunk in chunks),
        "h2d_fence_ns_max": max(chunk.h2d_fence_ns for chunk in chunks),
    }


def verify(rank: int, artifact: Path) -> dict[str, object]:
    """Authenticate both owner-local files without allocating CUDA memory."""

    from opentelemetry import trace

    loader = DirectSlabLoader(
        artifact, trace.get_tracer("rocket.qwen38.cuda_slab_loader.verify")
    )
    started_ns = time.perf_counter_ns()
    target_bytes = loader.read(f"rank{rank}-target", lambda _offset, _view: None)
    mtp_bytes = loader.read(f"rank{rank}-mtp", lambda _offset, _view: None)
    completed_ns = time.perf_counter_ns()
    return {
        "result": "qwen38_owner_local_slab_verify",
        "rank": rank,
        "node": socket.gethostname(),
        "target_bytes": target_bytes,
        "mtp_bytes": mtp_bytes,
        "bytes_verified": target_bytes + mtp_bytes,
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "verify_ns": completed_ns - started_ns,
    }


def load(rank: int, artifact: Path, device: str) -> dict[str, object]:
    import torch
    from opentelemetry import metrics, trace

    tracer = trace.get_tracer("rocket.qwen38.cuda_slab_loader")
    owner = Owner()
    target = artifact / f"rank{rank}-target.slab"
    mtp = artifact / f"rank{rank}-mtp.slab"
    if not target.is_file() or not mtp.is_file():
        raise RuntimeError("owner-local target or MTP slab is absent")
    payload_bytes = target.stat().st_size + mtp.stat().st_size
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    staging_bytes = 6 * 256 * 1024**2
    required_bytes = payload_bytes + staging_bytes + 2 * 1024**3
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"CUDA preflight requires {required_bytes} bytes, only {free_bytes} free"
        )
    loader = CudaRankSlabLoader(
        artifact,
        rank=rank,
        owner=owner,
        tracer=tracer,
        meter=metrics.get_meter("rocket.qwen38.cuda_slab_loader"),
        torch_api=torch,
        device=device,
    )
    result = loader.load()
    torch.cuda.synchronize(device)
    receipt = result.receipt
    if owner.publish_ns is None or receipt.bytes_read != payload_bytes:
        raise RuntimeError("rank load receipt or publication changed")
    if receipt.h2d_bytes != receipt.bytes_read:
        raise RuntimeError("rank load performed incomplete H2D placement")
    return {
        "result": "qwen38_owner_local_cuda_slab_load",
        "rank": rank,
        "node": socket.gethostname(),
        "device": torch.cuda.get_device_name(device),
        "clock": receipt.clock,
        "target_bytes": receipt.target.bytes_read,
        "mtp_bytes": receipt.mtp.bytes_read,
        "bytes_read": receipt.bytes_read,
        "h2d_bytes": receipt.h2d_bytes,
        "direct_reads": receipt.target.direct_reads + receipt.mtp.direct_reads,
        "h2d_copies": receipt.target.h2d_copies + receipt.mtp.h2d_copies,
        "target_started_ns": receipt.target.started_ns,
        "target_completed_ns": receipt.target.completed_ns,
        "target_stages": stage_summary(receipt.target),
        "mtp_started_ns": receipt.mtp.started_ns,
        "mtp_completed_ns": receipt.mtp.completed_ns,
        "mtp_stages": stage_summary(receipt.mtp),
        "reader_overlap_ns": receipt.reader_overlap_ns,
        "allocation_ns": receipt.allocation_ns,
        "publish_ns": receipt.publish_ns,
        "load_to_publish_ns": receipt.load_to_publish_ns,
        "effective_gbps": receipt.bytes_read / receipt.load_to_publish_ns,
        "preflight_free_bytes": free_bytes,
        "preflight_required_bytes": required_bytes,
        "published_once": True,
    }


def worker_command(
    repo: Path, artifact: Path, rank: int, image: str, remote: str | None
) -> list[str]:
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--entrypoint", "python3",
        "-v", f"{repo}:{CONTAINER_REPO}:ro",
        "-v", f"{artifact}:{artifact}:ro",
        "-e", f"PYTHONPATH={CONTAINER_REPO}/engines/qwen38-flash-next-nvfp4-2b/src",
        image, f"{CONTAINER_REPO}/{SCRIPT}", "--worker", "--rank", str(rank),
        "--artifact", str(artifact),
    ]
    if remote is None:
        return command
    return ["ssh", "-o", "BatchMode=yes", remote, shlex.join(command)]


def coordinate(repo: Path, artifact: Path, image: str, remote: str) -> dict[str, object]:
    commands = (
        worker_command(repo, artifact, 0, image, None),
        worker_command(repo, artifact, 1, image, remote),
    )
    coordinator_started_ns = time.perf_counter_ns()
    processes = tuple(
        subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for command in commands
    )

    def collect(index: int):
        stdout, stderr = processes[index].communicate()
        completed_ns = time.perf_counter_ns()
        if processes[index].returncode:
            raise RuntimeError(
                f"rank {index} loader failed ({processes[index].returncode}): {stderr.strip()}"
            )
        lines = [line for line in stdout.splitlines() if line.startswith("{")]
        if len(lines) != 1:
            raise RuntimeError(f"rank {index} returned an invalid receipt")
        return json.loads(lines[0]), completed_ns

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(collect, rank) for rank in (0, 1))
        collected = tuple(future.result() for future in futures)
    completed = tuple(item[1] for item in collected)
    ranks = tuple(item[0] for item in collected)
    aggregate_bytes = sum(int(item["bytes_read"]) for item in ranks)
    elapsed_ns = max(completed) - coordinator_started_ns
    total_status = total_cold_start_status(coordinator_started_ns, {}, None)
    return {
        "result": "qwen38_two_node_cuda_slab_load",
        "clock": "time.perf_counter_ns on coordinator",
        "ranks": ranks,
        "aggregate_bytes_read": aggregate_bytes,
        "aggregate_h2d_bytes": sum(int(item["h2d_bytes"]) for item in ranks),
        "rank_process_overlap_ns": min(completed) - coordinator_started_ns,
        "subphase_load_to_both_published_ns": elapsed_ns,
        "aggregate_effective_gbps": aggregate_bytes / elapsed_ns,
        "total_cold_start": total_status,
        "comparison": {
            "vllm_model_ready_seconds": BASELINE_MODEL_READY_SECONDS,
            "vllm_process_read_bytes": VLLM_READ_BYTES,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "vllm_read_amplification": VLLM_READ_BYTES / CHECKPOINT_BYTES,
            "speedup_claimed": False,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--remote")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_only:
        if args.rank is None:
            raise SystemExit("--verify-only requires --rank")
        print(json.dumps(verify(args.rank, args.artifact), sort_keys=True))
        return
    if args.worker:
        if args.rank is None:
            raise SystemExit("--worker requires --rank")
        print(json.dumps(load(args.rank, args.artifact, args.device), sort_keys=True))
        return
    if not args.remote:
        raise SystemExit("two-node coordinator requires --remote")
    print(json.dumps(coordinate(
        args.repo.resolve(), args.artifact.resolve(), args.image, args.remote
    ), sort_keys=True))


if __name__ == "__main__":
    main()
