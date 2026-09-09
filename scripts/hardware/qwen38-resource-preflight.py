#!/usr/bin/env python3
"""Read-only resource checks before a Qwen3.8 capture or native launch.

The guard samples host state only. It never removes containers, changes GPU
state, drops caches, or inspects arbitrary process metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Sequence


SCHEMA = "rocket.qwen38.resource-preflight.v1"
BYTES_PER_GIB = 1024**3
DEFAULT_MIN_MEM_AVAILABLE_GIB = 16
DEFAULT_MIN_SWAP_FREE_GIB = 4
DEFAULT_EXPECTED_GPUS = 1
COMMAND_TIMEOUT_SECONDS = 10
_INTEGER = re.compile(r"[0-9]+")
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CONTAINER_MISSING = re.compile(
    r"(?:Error:\s*No such object:|Error(?: response from daemon)?:\s*"
    r"No such container:)\s*([^\n]+)\Z"
)
_CONTAINER_STATES = frozenset(
    {"created", "running", "paused", "restarting", "removing", "exited", "dead"}
)


class ResourcePreflightError(RuntimeError):
    """A bounded, non-sensitive reason for a failed host check."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _command_output(
    command: Sequence[str], runner: Callable[..., subprocess.CompletedProcess[str]]
) -> str:
    """Run one read-only command without returning stderr or command details."""

    try:
        result = runner(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ResourcePreflightError("command_timeout") from error
    except (FileNotFoundError, OSError) as error:
        raise ResourcePreflightError("command_unavailable") from error
    if result.returncode != 0:
        raise ResourcePreflightError("command_failed")
    if not isinstance(result.stdout, str):
        raise ResourcePreflightError("malformed_reading")
    return result.stdout


def parse_meminfo(payload: str) -> dict[str, int]:
    """Parse the two required kB counters from ``/proc/meminfo``."""

    values: dict[str, int] = {}
    for line in payload.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        if name not in {"MemAvailable", "SwapFree"}:
            continue
        fields = raw.split()
        if len(fields) != 2 or fields[1] != "kB" or not _INTEGER.fullmatch(fields[0]):
            raise ResourcePreflightError("malformed_meminfo")
        value = int(fields[0])
        if name in values or value < 0:
            raise ResourcePreflightError("malformed_meminfo")
        values[name] = value * 1024
    if set(values) != {"MemAvailable", "SwapFree"}:
        raise ResourcePreflightError("missing_meminfo")
    return values


def read_gpu_count(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Return the number of uniquely reported local GPUs."""

    output = _command_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        runner,
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if not rows or any(not _INTEGER.fullmatch(row) for row in rows):
        raise ResourcePreflightError("malformed_gpu_inventory")
    indices = [int(row) for row in rows]
    if len(set(indices)) != len(indices):
        raise ResourcePreflightError("duplicate_gpu_inventory")
    return len(indices)


def read_compute_owner_count(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    """Return the count of compute-owner rows without exposing PIDs."""

    output = _command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        runner,
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if any(not _INTEGER.fullmatch(row) for row in rows):
        raise ResourcePreflightError("malformed_compute_inventory")
    return len(rows)


def read_container_state(
    name: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str | None:
    """Return a named container state, or ``None`` when it is absent."""

    if not _CONTAINER_NAME.fullmatch(name):
        raise ResourcePreflightError("invalid_container_name")
    try:
        result = runner(
            ["docker", "container", "inspect", "--format", "{{.State.Status}}", name],
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ResourcePreflightError("command_timeout") from error
    except (FileNotFoundError, OSError) as error:
        raise ResourcePreflightError("command_unavailable") from error
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else ""
        missing = _CONTAINER_MISSING.fullmatch(stderr.strip())
        if missing is not None and missing.group(1) == name:
            return None
        raise ResourcePreflightError("command_failed")
    if not isinstance(result.stdout, str):
        raise ResourcePreflightError("malformed_container_state")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0] not in _CONTAINER_STATES:
        raise ResourcePreflightError("malformed_container_state")
    return rows[0]


def _failure(check: str, reason: str) -> dict[str, str]:
    return {"check": check, "reason": reason}


def run_preflight(
    *,
    min_mem_available_bytes: int,
    min_swap_free_bytes: int,
    expected_gpu_count: int,
    container_names: Sequence[str],
    meminfo_reader: Callable[[], str],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Run all checks and return structured, bounded telemetry."""

    failures: list[dict[str, str]] = []
    observations: dict[str, object] = {}

    try:
        memory = parse_meminfo(meminfo_reader())
        observations["memory"] = {
            "mem_available_bytes": memory["MemAvailable"],
            "swap_free_bytes": memory["SwapFree"],
        }
        if memory["MemAvailable"] < min_mem_available_bytes:
            failures.append(_failure("memory.mem_available", "below_minimum"))
        if memory["SwapFree"] < min_swap_free_bytes:
            failures.append(_failure("memory.swap_free", "below_minimum"))
    except (ResourcePreflightError, OSError) as error:
        reason = error.reason if isinstance(error, ResourcePreflightError) else "read_failed"
        failures.append(_failure("memory", reason))

    try:
        gpu_count = read_gpu_count(runner)
        observations["gpu_inventory"] = {"observed_gpu_count": gpu_count}
        if gpu_count != expected_gpu_count:
            failures.append(_failure("gpu_inventory", "count_mismatch"))
    except ResourcePreflightError as error:
        failures.append(_failure("gpu_inventory", error.reason))

    try:
        owner_count = read_compute_owner_count(runner)
        observations["compute_owners"] = {"owner_count": owner_count}
        if owner_count != 0:
            failures.append(_failure("compute_owners", "owners_present"))
    except ResourcePreflightError as error:
        failures.append(_failure("compute_owners", error.reason))

    container_observations: list[dict[str, object]] = []
    seen_names: set[str] = set()
    if not container_names:
        failures.append(_failure("containers", "no_names"))
    for name in container_names:
        if name in seen_names:
            failures.append(_failure("containers", "duplicate_name"))
            continue
        seen_names.add(name)
        try:
            state = read_container_state(name, runner)
            container_observations.append(
                {"name": name, "state": state if state is not None else "absent"}
            )
            if state is not None:
                failures.append(_failure(f"container:{name}", "present"))
        except ResourcePreflightError as error:
            failures.append(_failure(f"container:{name}", error.reason))
    observations["containers"] = container_observations

    return {
        "schema": SCHEMA,
        "status": "passed" if not failures else "failed",
        "thresholds": {
            "min_mem_available_bytes": min_mem_available_bytes,
            "min_swap_free_bytes": min_swap_free_bytes,
            "expected_gpu_count": expected_gpu_count,
        },
        "observations": observations,
        "failures": failures,
    }


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(
    argv: Sequence[str] | None = None,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-mem-available-gib",
        type=_nonnegative_int,
        default=DEFAULT_MIN_MEM_AVAILABLE_GIB,
    )
    parser.add_argument(
        "--min-swap-free-gib",
        type=_nonnegative_int,
        default=DEFAULT_MIN_SWAP_FREE_GIB,
    )
    parser.add_argument(
        "--expected-gpus", type=_positive_int, default=DEFAULT_EXPECTED_GPUS
    )
    parser.add_argument(
        "--container-name", action="append", required=True, dest="container_names"
    )
    args = parser.parse_args(argv)

    def read_meminfo() -> str:
        return meminfo_path.read_text(encoding="ascii")

    payload = run_preflight(
        min_mem_available_bytes=args.min_mem_available_gib * BYTES_PER_GIB,
        min_swap_free_bytes=args.min_swap_free_gib * BYTES_PER_GIB,
        expected_gpu_count=args.expected_gpus,
        container_names=args.container_names,
        meminfo_reader=read_meminfo,
        runner=runner,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
