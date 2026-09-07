#!/usr/bin/env python3
"""Measure pinned two-node vLLM service exec through first generated token.

Artifact construction and deployment are intentionally outside this timer. Run
the expanded-calibration launcher in ``--two-node-preflight`` mode first, then
pass that immutable directory here. Every replicate cleans the named service,
drops both nodes' page caches, samples both GPUs, starts the worker and head,
waits for model readiness, and terminates only after a streamed response carries
a non-empty generated token.
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

SCHEMA = "rocket.qwen38.vllm-cold-first-token.v1"
HEAD_CONTAINER = "rocket-qwen38-calibration-head"
WORKER_CONTAINER = "rocket-qwen38-calibration-worker"
MODEL = "qwen3.8-flash-next"


def first_generated_token(lines: Iterable[bytes]) -> tuple[str, str]:
    """Return the first non-empty generated field from an OpenAI SSE stream."""

    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
            if not isinstance(delta, dict):
                continue
            for field in ("content", "reasoning_content"):
                value = delta.get(field)
                if isinstance(value, str) and value:
                    return field, value
    raise RuntimeError("stream completed without a generated token")


def hardware_summary(
    samples: list[dict[str, object]], started_unix_ns: int, finished_unix_ns: int
) -> dict[str, object]:
    """Reduce only paired GPU samples overlapping the timed cold-start window."""

    selected = [
        row
        for row in samples
        if int(row["finished_unix_ns"]) >= started_unix_ns
        and int(row["started_unix_ns"]) <= finished_unix_ns
    ]
    ranks = []
    for rank in (0, 1):
        rows = [row for row in selected if row.get("rank") == rank]
        if not rows:
            raise RuntimeError(f"no rank {rank} hardware samples in cold-start window")
        rank_summary: dict[str, object] = {"rank": rank, "samples": len(rows)}
        for source, destination in (
            ("clock_mhz", "clock_mhz"),
            ("power_w", "power_w"),
            ("utilization_percent", "utilization_percent"),
        ):
            values = [float(row[source]) for row in rows]
            rank_summary[destination] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
        ranks.append(rank_summary)
    spans = {
        (int(row["pair_started_unix_ns"]), int(row["pair_finished_unix_ns"]))
        for row in selected
    }
    return {
        "query": "clocks.sm,power.draw,utilization.gpu",
        "max_pair_span_ms": max((end - start) / 1e6 for start, end in spans),
        "ranks": ranks,
    }


def variance_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    """Return explicit sample variance for every required cold-start duration."""

    if len(runs) < 2:
        raise ValueError("at least two cold-start runs are required for variance")
    output: dict[str, object] = {"runs": len(runs)}
    for field in (
        "exec_to_model_ready_seconds",
        "model_ready_to_first_token_seconds",
        "exec_to_first_token_seconds",
    ):
        values = [float(run[field]) for run in runs]
        output[field] = {
            "mean": statistics.mean(values),
            "sample_stdev": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
    return output


def validate_production_prepared(
    prepared: Path, required_worker_cache_kind: str | None = None
) -> dict[str, object]:
    """Reject calibration/eager launch scripts before any cold timer starts."""

    run_record = json.loads(prepared.joinpath("run.json").read_text())
    if run_record.get("mtp_depth") != 1:
        raise ValueError("cold c16 control requires the measured K1 ceiling")
    if required_worker_cache_kind is not None:
        actual = run_record.get("worker_cache_kind")
        if actual != required_worker_cache_kind:
            raise ValueError(
                "prepared worker cache kind mismatch: "
                f"expected {required_worker_cache_kind}, got {actual or 'missing'}"
            )
        if required_worker_cache_kind == "host_ext4" and not run_record.get(
            "checkpoint_manifest_sha256"
        ):
            raise ValueError("local ext4 preparation lacks checkpoint manifest proof")
        if required_worker_cache_kind == "host_ext4":
            if (
                run_record.get("head_cache_filesystem") != "ext4"
                or run_record.get("worker_cache_filesystem") != "ext4"
            ):
                raise ValueError("local cache preparation requires ext4 on both nodes")
            if run_record.get("checkpoint_safetensor_shards") != 11:
                raise ValueError("local cache preparation requires 11 checkpoint shards")
            if not run_record.get("head_snapshot_path") or not run_record.get(
                "worker_snapshot_path"
            ):
                raise ValueError("local cache preparation lacks exact snapshot paths")
            if not run_record.get("head_runtime_cache_path") or not run_record.get(
                "worker_runtime_cache_path"
            ):
                raise ValueError("local cache preparation lacks exact runtime cache paths")
    for name in ("launch-head.sh", "launch-worker.sh"):
        source = prepared.joinpath(name).read_text()
        forbidden = (
            "--enforce-eager",
            "ROCKET_NVFP4_CALIBRATE",
            "ROCKET_NVFP4_SAMPLE_ELEMENTS",
            "ROCKET_NVFP4_MAX_EMISSIONS",
            "ROCKET_QWEN38_LOAD_TRACE",
            "model_telemetry.py",
        )
        present = [item for item in forbidden if item in source]
        if present:
            raise ValueError(f"{name} is not a production launch: {present}")
    return run_record


def run_checked(command: list[str], *, timeout: float | None = None) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def run_combined(command: list[str], *, timeout: float | None = None) -> str:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stdout.strip()}"
        )
    return result.stdout


def remote(worker: str, command: str, *, timeout: float | None = None) -> str:
    return run_checked(
        ["ssh", "-o", "BatchMode=yes", worker, command],
        timeout=timeout,
    )


def remote_combined(worker: str, command: str) -> str:
    return run_combined(["ssh", "-o", "BatchMode=yes", worker, command])


def clean_and_drop_caches(worker: str) -> None:
    """Remove only owned containers, reject other CUDA owners, then go cold."""

    if _container_exists(HEAD_CONTAINER):
        run_checked(["docker", "rm", "-f", HEAD_CONTAINER])
    remote(worker, f"docker rm -f {WORKER_CONTAINER} >/dev/null 2>&1 || true")
    local_pids = run_checked(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    ).strip()
    worker_pids = remote(
        worker,
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
    ).strip()
    if local_pids or worker_pids:
        raise RuntimeError("cold-start control requires both GPUs to have no compute owners")
    run_checked(
        ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    )
    remote(worker, "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")


def _container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", name], capture_output=True, text=True, check=False
    )
    return result.returncode == 0


def wait_for_health(endpoint: str, deadline_ns: int) -> None:
    while time.monotonic_ns() < deadline_ns:
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("vLLM did not become model-ready before the cold-start deadline")


def request_first_token(
    endpoint: str, timeout_seconds: float, stream_path: Path
) -> tuple[str, str, int, int]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with one digit."}],
        "max_tokens": 8,
        "min_tokens": 8,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        stream_path.with_name("first-token-request.json").write_text(
            json.dumps(
                {
                    "endpoint_path": "/v1/chat/completions",
                    "request": payload,
                    "response_content_type": response.headers.get("Content-Type"),
                    "response_status": response.status,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        with stream_path.open("wb") as stream:

            def recorded_lines() -> Iterable[bytes]:
                for line in response:
                    stream.write(line)
                    stream.flush()
                    yield line

            field, text = first_generated_token(recorded_lines())
        return field, text, time.monotonic_ns(), time.time_ns()


def stop_monitor(process: subprocess.Popen[bytes]) -> None:
    process.send_signal(signal.SIGINT)
    if process.wait(timeout=30) != 0:
        raise RuntimeError("two-node hardware monitor failed")


def collect_logs(worker: str, run_dir: Path) -> None:
    run_dir.joinpath("head.log").write_text(
        run_combined(["docker", "logs", "--timestamps", HEAD_CONTAINER])
    )
    run_dir.joinpath("worker.log").write_text(
        remote_combined(worker, f"docker logs --timestamps {WORKER_CONTAINER}")
    )


def cleanup_services(worker: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", HEAD_CONTAINER],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            worker,
            "docker",
            "rm",
            "-f",
            WORKER_CONTAINER,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def one_run(
    prepared: Path,
    output: Path,
    worker: str,
    endpoint: str,
    timeout_seconds: int,
    monitor_script: Path,
) -> dict[str, object]:
    clean_and_drop_caches(worker)
    samples_path = output / "hardware-samples.jsonl"
    monitor = subprocess.Popen(
        [
            "python3",
            str(monitor_script),
            "collect",
            "--worker",
            worker,
            "--output",
            str(samples_path),
        ]
    )
    service_exec_started_ns = time.monotonic_ns()
    service_exec_started_unix_ns = time.time_ns()
    try:
        remote(
            worker,
            f"bash {shlex.quote(str(prepared / 'launch-worker.sh'))}",
            timeout=30,
        )
        time.sleep(15)
        run_checked(["bash", str(prepared / "launch-head.sh")], timeout=30)
        deadline_ns = service_exec_started_ns + timeout_seconds * 1_000_000_000
        wait_for_health(endpoint, deadline_ns)
        model_ready_ns = time.monotonic_ns()
        model_ready_unix_ns = time.time_ns()
        remaining = max(1.0, (deadline_ns - model_ready_ns) / 1e9)
        token_field, token_text, first_token_ns, first_token_unix_ns = (
            request_first_token(endpoint, remaining, output / "first-token-stream.sse")
        )
        collect_logs(worker, output)
    finally:
        stop_monitor(monitor)
        cleanup_services(worker)
    samples = [
        json.loads(line)
        for line in samples_path.read_text().splitlines()
        if line.strip()
    ]
    return {
        "service_exec_started_monotonic_ns": service_exec_started_ns,
        "model_ready_monotonic_ns": model_ready_ns,
        "first_token_monotonic_ns": first_token_ns,
        "service_exec_started_unix_ns": service_exec_started_unix_ns,
        "model_ready_unix_ns": model_ready_unix_ns,
        "first_token_unix_ns": first_token_unix_ns,
        "exec_to_model_ready_seconds": (model_ready_ns - service_exec_started_ns)
        / 1e9,
        "model_ready_to_first_token_seconds": (first_token_ns - model_ready_ns)
        / 1e9,
        "exec_to_first_token_seconds": (first_token_ns - service_exec_started_ns)
        / 1e9,
        "first_token_field": token_field,
        "first_token_text": token_text,
        "hardware": hardware_summary(
            samples, service_exec_started_unix_ns, first_token_unix_ns
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker", default="glwillen@192.168.100.11")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8888")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--required-worker-cache-kind", choices=("docker_volume", "host_ext4")
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 to report variance")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.prepared_dir.is_absolute() or not args.output_dir.is_absolute():
        parser.error("--prepared-dir and --output-dir must be absolute")
    for name in ("launch-head.sh", "launch-worker.sh", "run.json"):
        if not args.prepared_dir.joinpath(name).is_file():
            parser.error(f"prepared directory is missing {name}")
    try:
        prepared_run = validate_production_prepared(
            args.prepared_dir, args.required_worker_cache_kind
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    args.output_dir.mkdir(parents=True)
    monitor = Path(__file__).resolve().parents[1] / "hardware/qwen38-two-node-monitor.py"
    runs = []
    for index in range(1, args.runs + 1):
        run_dir = args.output_dir / f"run-{index:02d}"
        run_dir.mkdir()
        result = one_run(
            args.prepared_dir,
            run_dir,
            args.worker,
            args.endpoint,
            args.timeout_seconds,
            monitor,
        )
        result["run"] = index
        run_dir.joinpath("result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        runs.append(result)
    payload = {
        "schema": SCHEMA,
        "boundary": {
            "start": "monotonic timestamp immediately before worker service exec",
            "model_ready": "first successful HTTP /health after engine initialization",
            "terminal": "first non-empty generated token in a real streamed request",
            "excluded": "artifact construction and two-node deployment",
            "model_ready_observation_resolution_seconds": 1,
        },
        "prepared_run": prepared_run,
        "runs": runs,
        "variance": variance_summary(runs),
    }
    args.output_dir.joinpath("cold-first-token.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
