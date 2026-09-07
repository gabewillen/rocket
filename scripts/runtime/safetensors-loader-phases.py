#!/usr/bin/env python3
"""Measure bounded safetensors loader phases without launching a model.

The CLI reads one safetensors payload window through three independent paths:

* ``preadv_aligned``: positional reads into one reusable anonymous buffer.
* ``mmap_fault_copy``: a fresh regular-file mapping per iteration, copied into
  that same anonymous buffer so mapping faults and the copy are timed together.
* ``hot_reused_memcpy``: repeated copies between two prefaulted anonymous buffers.

Anonymous buffers are 65,536-byte aligned. The process opens the checkpoint
read-only, does not drop global page cache, and allocates at most two
``chunk_bytes + 65,536`` anonymous mappings plus one bounded file mapping.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import resource
import statistics
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

ALIGNMENT_BYTES = 65_536
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_SAMPLE_BYTES = 1024 * 1024 * 1024
DEFAULT_ITERATIONS = 3
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 1024 * 1024 * 1024
MAX_ITERATIONS = 1000
SCHEMA = "rocket.safetensors-loader-phases.v1"


class BenchmarkError(RuntimeError):
    """Expected input, platform, or I/O failure at the benchmark boundary."""


@dataclass(frozen=True)
class FaultCounts:
    """Process page-fault counters sampled around one iteration."""

    minor: int
    major: int


@dataclass(frozen=True)
class SafetensorsRegion:
    """Validated payload boundaries for one safetensors file."""

    path: Path
    file_size_bytes: int
    header_bytes: int
    payload_offset_bytes: int
    payload_bytes: int
    tensor_count: int
    device: int
    inode: int


@dataclass(frozen=True)
class BenchmarkConfig:
    """Finite benchmark dimensions, all measured in bytes or iterations."""

    file: Path
    sample_bytes_requested: int
    iterations: int
    chunk_bytes: int


class AlignedAnonymousBuffer:
    """Own one reusable anonymous mapping and expose an aligned writable view.

    The owner must be closed after all borrowed views are released. Instances
    are single-threaded and are not safe for concurrent mutation.
    """

    def __init__(self, size: int, alignment: int = ALIGNMENT_BYTES) -> None:
        if size <= 0:
            raise ValueError("buffer size must be positive")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        self.size = size
        self.alignment = alignment
        self._mapping = mmap.mmap(-1, size + alignment)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        offset = (-base) % alignment
        self.address = base + offset
        self.view = memoryview(self._mapping)[offset : offset + size]
        self._closed = False
        if self.address % alignment != 0:
            self.close()
            raise BenchmarkError("anonymous buffer alignment invariant failed")

    def close(self) -> None:
        if self._closed:
            return
        self.view.release()
        self._mapping.close()
        self._closed = True

    def __enter__(self) -> AlignedAnonymousBuffer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


Clock = Callable[[], float]
FaultReader = Callable[[], FaultCounts]


def read_fault_counts() -> FaultCounts:
    """Return this process's cumulative minor and major page faults."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return FaultCounts(minor=usage.ru_minflt, major=usage.ru_majflt)


def _read_exact(file: BinaryIO, count: int, what: str) -> bytes:
    data = file.read(count)
    if len(data) != count:
        raise BenchmarkError(f"short read while reading {what}: {len(data)} of {count} bytes")
    return data


def inspect_safetensors(path: Path) -> SafetensorsRegion:
    """Validate a safetensors header and return its bounded payload region.

    The file is borrowed and opened read-only. Headers larger than 64 MiB are
    rejected before allocation. Tensor offsets must be integer pairs contained
    in the file payload; malformed JSON or metadata raises ``BenchmarkError``.
    """

    display_path = path.absolute()
    try:
        with display_path.open("rb") as file:
            identity = os.fstat(file.fileno())
            file_size = identity.st_size
            if file_size < 9:
                raise BenchmarkError("safetensors file is too small")
            header_bytes = struct.unpack("<Q", _read_exact(file, 8, "header length"))[0]
            if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
                raise BenchmarkError(
                    f"safetensors header length must be in 1..{MAX_HEADER_BYTES} bytes"
                )
            if header_bytes > file_size - 8:
                raise BenchmarkError("safetensors header extends beyond the file")
            raw_header = _read_exact(file, header_bytes, "safetensors header")
    except OSError as exc:
        raise BenchmarkError(f"cannot read safetensors file: {exc}") from exc

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise BenchmarkError("safetensors header must be a JSON object")

    payload_offset = 8 + header_bytes
    payload_capacity = file_size - payload_offset
    tensor_count = 0
    payload_bytes = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise BenchmarkError("safetensors tensor metadata must be keyed objects")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise BenchmarkError(f"tensor {name!r} has invalid data_offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_capacity:
            raise BenchmarkError(f"tensor {name!r} data_offsets exceed the file payload")
        tensor_count += 1
        payload_bytes = max(payload_bytes, end)
    if tensor_count == 0 or payload_bytes == 0:
        raise BenchmarkError("safetensors file has no non-empty tensor payload")

    return SafetensorsRegion(
        path=display_path,
        file_size_bytes=file_size,
        header_bytes=header_bytes,
        payload_offset_bytes=payload_offset,
        payload_bytes=payload_bytes,
        tensor_count=tensor_count,
        device=identity.st_dev,
        inode=identity.st_ino,
    )


def _read_bounded_json(path: Path, byte_limit: int, what: str) -> object:
    try:
        size = path.stat().st_size
        if size > byte_limit:
            raise BenchmarkError(f"{what} exceeds {byte_limit} bytes")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot parse {what}: {exc}") from exc


def resolve_checkpoint_file(checkpoint: Path) -> Path:
    """Resolve one deterministic safetensors shard from a checkpoint directory."""

    root = checkpoint.absolute()
    if not root.is_dir():
        raise BenchmarkError(f"checkpoint is not a directory: {root}")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_bounded_json(index_path, MAX_INDEX_BYTES, "safetensors index")
        if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
            raise BenchmarkError("safetensors index has no weight_map object")
        shard_values = list(index["weight_map"].values())
        if not shard_values or any(not isinstance(shard, str) for shard in shard_values):
            raise BenchmarkError("safetensors index has no valid shard names")
        shards = sorted(set(shard_values))
        relative = Path(shards[0])
        if relative.is_absolute() or relative.name != str(relative):
            raise BenchmarkError("safetensors index shard must be a plain file name")
        selected = root / relative
    else:
        candidates = sorted(root.glob("*.safetensors"))
        if not candidates:
            raise BenchmarkError(f"checkpoint has no safetensors files: {root}")
        selected = candidates[0]
    if not selected.is_file():
        raise BenchmarkError(f"selected safetensors shard does not exist: {selected}")
    return selected


def validate_config(config: BenchmarkConfig) -> SafetensorsRegion:
    """Check all caller preconditions before timing any phase."""

    if not hasattr(os, "preadv"):
        raise BenchmarkError("this platform does not provide os.preadv")
    if config.sample_bytes_requested <= 0:
        raise BenchmarkError("sample bytes must be positive")
    if not 1 <= config.iterations <= MAX_ITERATIONS:
        raise BenchmarkError(f"iterations must be in 1..{MAX_ITERATIONS}")
    if not ALIGNMENT_BYTES <= config.chunk_bytes <= MAX_CHUNK_BYTES:
        raise BenchmarkError(
            f"chunk bytes must be in {ALIGNMENT_BYTES}..{MAX_CHUNK_BYTES}"
        )
    if config.chunk_bytes % ALIGNMENT_BYTES:
        raise BenchmarkError(f"chunk bytes must be a multiple of {ALIGNMENT_BYTES}")
    return inspect_safetensors(config.file)


def _pread_exact_into(fd: int, target: memoryview, offset: int, count: int) -> None:
    completed = 0
    while completed < count:
        read = os.preadv(fd, [target[completed:count]], offset + completed)
        if read <= 0:
            raise BenchmarkError(
                f"short preadv at file offset {offset + completed}: {completed} of {count} bytes"
            )
        completed += read


def _run_pread_iteration(
    fd: int, destination: AlignedAnonymousBuffer, offset: int, byte_count: int
) -> int:
    completed = 0
    while completed < byte_count:
        count = min(destination.size, byte_count - completed)
        _pread_exact_into(fd, destination.view, offset + completed, count)
        completed += count
    return completed


def _run_mmap_iteration(
    fd: int, destination: AlignedAnonymousBuffer, offset: int, byte_count: int
) -> int:
    mapping_bytes = offset + byte_count
    try:
        with mmap.mmap(fd, mapping_bytes, access=mmap.ACCESS_COPY) as source:
            source_address = ctypes.addressof(ctypes.c_char.from_buffer(source))
            completed = 0
            while completed < byte_count:
                count = min(destination.size, byte_count - completed)
                ctypes.memmove(
                    destination.address,
                    source_address + offset + completed,
                    count,
                )
                completed += count
            return completed
    except (BufferError, OSError, ValueError) as exc:
        raise BenchmarkError(f"mmap fault+copy failed: {exc}") from exc


def _run_hot_reused_copy_iteration(
    source: AlignedAnonymousBuffer,
    destination: AlignedAnonymousBuffer,
    byte_count: int,
) -> int:
    completed = 0
    while completed < byte_count:
        count = min(source.size, byte_count - completed)
        ctypes.memmove(destination.address, source.address, count)
        completed += count
    return completed


def _round_float(value: float) -> float:
    return round(value, 9)


def _measure_iterations(
    name: str,
    operation_detail: str,
    byte_count: int,
    iterations: int,
    operation: Callable[[], int],
    clock: Clock,
    fault_reader: FaultReader,
) -> dict[str, object]:
    measurements: list[dict[str, object]] = []
    for iteration in range(1, iterations + 1):
        faults_before = fault_reader()
        started = clock()
        transferred = operation()
        elapsed = clock() - started
        faults_after = fault_reader()
        if transferred != byte_count:
            raise BenchmarkError(
                f"{name} transferred {transferred} bytes, expected {byte_count}"
            )
        if elapsed <= 0:
            raise BenchmarkError(f"{name} timer did not advance")
        measurements.append(
            {
                "iteration": iteration,
                "bytes": transferred,
                "elapsed_seconds": _round_float(elapsed),
                "gib_per_second": _round_float(transferred / elapsed / (1 << 30)),
                "minor_faults": faults_after.minor - faults_before.minor,
                "major_faults": faults_after.major - faults_before.major,
            }
        )
    rates = [float(item["gib_per_second"]) for item in measurements]
    elapsed_values = [float(item["elapsed_seconds"]) for item in measurements]
    return {
        "name": name,
        "operation": operation_detail,
        "iterations": measurements,
        "summary": {
            "median_elapsed_seconds": _round_float(statistics.median(elapsed_values)),
            "median_gib_per_second": _round_float(statistics.median(rates)),
            "min_gib_per_second": _round_float(min(rates)),
            "max_gib_per_second": _round_float(max(rates)),
        },
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    clock: Clock = time.perf_counter,
    fault_reader: FaultReader = read_fault_counts,
) -> dict[str, object]:
    """Run all phases and return a stable, JSON-serializable report.

    The selected file is read-only. Work is bounded by ``iterations`` times the
    clamped payload sample for each phase. The warm phase copies the same byte
    count from a prefaulted synthetic source and does not read the checkpoint.
    """

    region = validate_config(config)
    sample_bytes = min(config.sample_bytes_requested, region.payload_bytes)
    if sample_bytes <= 0:
        raise BenchmarkError("selected safetensors payload is empty")

    try:
        fd = os.open(region.path, os.O_RDONLY)
    except OSError as exc:
        raise BenchmarkError(f"cannot open safetensors file read-only: {exc}") from exc

    try:
        identity = os.fstat(fd)
        if (
            identity.st_dev != region.device
            or identity.st_ino != region.inode
            or identity.st_size != region.file_size_bytes
        ):
            raise BenchmarkError("safetensors file identity changed after validation")
        with (
            AlignedAnonymousBuffer(config.chunk_bytes) as source,
            AlignedAnonymousBuffer(config.chunk_bytes) as destination,
        ):
            ctypes.memset(source.address, 0xA5, source.size)
            ctypes.memset(destination.address, 0, destination.size)
            phases = [
                _measure_iterations(
                    "preadv_aligned",
                    "sequential os.preadv into a reusable 64 KiB-aligned anonymous buffer",
                    sample_bytes,
                    config.iterations,
                    lambda: _run_pread_iteration(
                        fd, destination, region.payload_offset_bytes, sample_bytes
                    ),
                    clock,
                    fault_reader,
                ),
                _measure_iterations(
                    "mmap_fault_copy",
                    "fresh regular-file ACCESS_COPY mmap plus copy into aligned anonymous storage",
                    sample_bytes,
                    config.iterations,
                    lambda: _run_mmap_iteration(
                        fd, destination, region.payload_offset_bytes, sample_bytes
                    ),
                    clock,
                    fault_reader,
                ),
                _measure_iterations(
                    "hot_reused_memcpy",
                    "ctypes.memmove between two prefaulted aligned anonymous buffers",
                    sample_bytes,
                    config.iterations,
                    lambda: _run_hot_reused_copy_iteration(
                        source, destination, sample_bytes
                    ),
                    clock,
                    fault_reader,
                ),
            ]
    finally:
        os.close(fd)

    return {
        "schema": SCHEMA,
        "file": str(region.path),
        "file_size_bytes": region.file_size_bytes,
        "safetensors_header_bytes": region.header_bytes,
        "payload_offset_bytes": region.payload_offset_bytes,
        "payload_bytes": region.payload_bytes,
        "tensor_count": region.tensor_count,
        "sample_bytes_requested": config.sample_bytes_requested,
        "sample_bytes": sample_bytes,
        "iterations": config.iterations,
        "chunk_bytes": config.chunk_bytes,
        "phase_order": [phase["name"] for phase in phases],
        "hot_copy_working_set_bytes": config.chunk_bytes,
        "anonymous_alignment_bytes": ALIGNMENT_BYTES,
        "host_page_bytes": mmap.PAGESIZE,
        "anonymous_scratch_upper_bound_bytes": 2
        * (config.chunk_bytes + ALIGNMENT_BYTES),
        "serialized_header_limit_bytes": MAX_HEADER_BYTES,
        "file_mapping_bytes_per_mmap_iteration": region.payload_offset_bytes
        + sample_bytes,
        "cache_control": "none; filesystem cache state is ambient and reported iterations stay separate",
        "cache_note": "preadv runs before mmap, so mmap sees pages warmed by this process under ambient cache",
        "phases": phases,
    }


def render_json(report: dict[str, object]) -> str:
    """Render the stable report schema as deterministic-key-order JSON."""

    return json.dumps(report, indent=2, ensure_ascii=True) + "\n"


def render_table(report: dict[str, object]) -> str:
    """Render one stable row per iteration followed by phase summaries."""

    lines = ["metric\tvalue"]
    for key in (
        "schema",
        "file",
        "file_size_bytes",
        "payload_offset_bytes",
        "payload_bytes",
        "sample_bytes_requested",
        "sample_bytes",
        "iterations",
        "chunk_bytes",
        "phase_order",
        "hot_copy_working_set_bytes",
        "anonymous_alignment_bytes",
        "host_page_bytes",
        "anonymous_scratch_upper_bound_bytes",
        "serialized_header_limit_bytes",
        "file_mapping_bytes_per_mmap_iteration",
        "cache_control",
        "cache_note",
    ):
        value = report[key]
        rendered = (
            json.dumps(value, ensure_ascii=True)
            if isinstance(value, (str, list))
            else str(value)
        )
        lines.append(f"{key}\t{rendered}")
    lines.extend(
        [
            "",
        "phase\titeration\tbytes\telapsed_s\tGiB_s\tminor_faults\tmajor_faults"
        ]
    )
    phases = report["phases"]
    assert isinstance(phases, list)
    for phase in phases:
        assert isinstance(phase, dict)
        name = phase["name"]
        iterations = phase["iterations"]
        assert isinstance(iterations, list)
        for item in iterations:
            assert isinstance(item, dict)
            lines.append(
                f"{name}\t{item['iteration']}\t{item['bytes']}\t"
                f"{float(item['elapsed_seconds']):.9f}\t"
                f"{float(item['gib_per_second']):.9f}\t"
                f"{item['minor_faults']}\t{item['major_faults']}"
            )
    lines.append("")
    lines.append("phase\tmedian_elapsed_s\tmedian_GiB_s\tmin_GiB_s\tmax_GiB_s")
    for phase in phases:
        assert isinstance(phase, dict)
        summary = phase["summary"]
        assert isinstance(summary, dict)
        lines.append(
            f"{phase['name']}\t{float(summary['median_elapsed_seconds']):.9f}\t"
            f"{float(summary['median_gib_per_second']):.9f}\t"
            f"{float(summary['min_gib_per_second']):.9f}\t"
            f"{float(summary['max_gib_per_second']):.9f}"
        )
    return "\n".join(lines) + "\n"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded safetensors loader phases without launching a model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="checkpoint directory")
    source.add_argument("--file", type=Path, help="exact .safetensors file")
    parser.add_argument(
        "--sample-bytes",
        type=_positive_int,
        default=DEFAULT_SAMPLE_BYTES,
        help="maximum payload bytes per iteration (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=DEFAULT_ITERATIONS,
        help="iterations per phase (default: %(default)s, maximum: 1000)",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=_positive_int,
        default=DEFAULT_CHUNK_BYTES,
        help="reusable buffer bytes, a multiple of 65536 (default: %(default)s)",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected = (
            resolve_checkpoint_file(args.checkpoint)
            if args.checkpoint is not None
            else args.file
        )
        if selected is None:
            raise BenchmarkError("one checkpoint or file source is required")
        report = run_benchmark(
            BenchmarkConfig(
                file=selected,
                sample_bytes_requested=args.sample_bytes,
                iterations=args.iterations,
                chunk_bytes=args.chunk_bytes,
            )
        )
    except BenchmarkError as exc:
        print(f"error: {exc}", file=stderr)
        return 2
    output = render_json(report) if args.format == "json" else render_table(report)
    stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
