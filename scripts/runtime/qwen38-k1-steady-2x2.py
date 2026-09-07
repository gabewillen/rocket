#!/usr/bin/env python3
"""Run matched K1 single/dual-HCA and CPU-pin steady decode controls."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "rocket.qwen38.k1-steady-2x2.v1"
WORKER = "glwillen@192.168.100.11"
FIXTURE = Path(
    "/home/glwillen/calibration/"
    "qwen38-vllm-cold-k1-local-ext4-production-prepared-20260907-03"
)
RUNNER = Path("/home/glwillen/Development/inference-references/myllmbox-runner")
RECIPE = Path(
    "/home/glwillen/Development/inference-references/"
    "qwen38-flash-next-cluster-recipe"
)
RUNNER_COMMIT = "5bc4c2b6f483f3a8f422ca8a8d8aac58c1a7c061"
RECIPE_COMMIT = "c7f69055e9ca572d0c562708a2cd68ea9a1af42b"
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
IMAGE_ID = "sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a"
MODEL_REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
FIXTURE_SHA256 = {
    "run.json": "be69fb77bb68a02c39df89783732789f43004e014c22547be3cc86d87bb57fe8",
    "launch-head.sh": "7f103aa014bb9c86453d32647451d5dbf43f9e7591c4543590ef151edddbb51b",
    "launch-worker.sh": "cf9e9afd3aa61bc3bbd9c04d2541baed7b546ee98000495a669bfdf45d897d07",
}
RUNNER_SHA256 = {
    "bench/test.py": "c98f15bf14c62e1f1040aaff406a182b76bb3c982e8e430b9d1d39c8ad855b8d",
    "bench/pasture-text.txt": "4d502a3633efd6f81f74c1db102793481eb7976bed90c99d8f409b4743c8fb68",
}
RUNNER_REL = "bench/test.py"
HEAD_CONTAINER = "rocket-qwen38-calibration-head"
WORKER_CONTAINER = "rocket-qwen38-calibration-worker"
HCA = ("rocep1s0f1", "roceP2p1s0f1")
NETDEV = ("enp1s0f1np1", "enP2p1s0f1np1")
CPUSET = "5-9,15-19"
CELLS = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Cell:
    name: str
    dual_hca: bool
    cpuset: bool


def _run(command: list[str], *, timeout: int = 30) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode:
        raise ContractError(
            f"command failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sources() -> dict[str, object]:
    for name, expected in FIXTURE_SHA256.items():
        if _sha256(FIXTURE / name) != expected:
            raise ContractError(f"K1 fixture identity changed: {name}")
    for name, expected in RUNNER_SHA256.items():
        if _sha256(RUNNER / name) != expected:
            raise ContractError(f"benchmark source identity changed: {name}")
    if _run(["git", "-C", str(RUNNER), "rev-parse", "HEAD"]) != RUNNER_COMMIT:
        raise ContractError("myllmbox runner commit changed")
    if _run(["git", "-C", str(RECIPE), "rev-parse", "HEAD"]) != RECIPE_COMMIT:
        raise ContractError("reference recipe commit changed")
    record = json.loads((FIXTURE / "run.json").read_text())
    expected = {
        "image_id": IMAGE_ID,
        "model_revision": MODEL_REVISION,
        "mtp_depth": 1,
        "worker_cache_kind": "host_ext4",
        "head_cache_filesystem": "ext4",
        "worker_cache_filesystem": "ext4",
        "checkpoint_safetensor_shards": 11,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ContractError("K1 fixture metadata changed")
    return record


def render_launch(source: str, cell: Cell) -> str:
    if source.count("NCCL_IB_HCA=rocep1s0f1") != 1:
        raise ContractError("launch HCA anchor changed")
    if "--cpuset-cpus" in source or "NCCL_IB_QPS_PER_CONNECTION" in source:
        raise ContractError("launch control is already modified")
    hca = ",".join(HCA) if cell.dual_hca else HCA[0]
    output = source.replace("NCCL_IB_HCA=rocep1s0f1", f"NCCL_IB_HCA={hca}")
    if cell.dual_hca:
        anchor = "  -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN \\\n"
        if output.count(anchor) != 1:
            raise ContractError("launch dual-HCA environment anchor changed")
        output = output.replace(
            anchor,
            anchor
            + "  -e NCCL_IB_QPS_PER_CONNECTION=4 "
            "-e NCCL_IB_SPLIT_DATA_ON_QPS=1 -e NCCL_CROSS_NIC=1 \\\n",
        )
    if cell.cpuset:
        anchor = "  --gpus all --network host --ipc host \\\n"
        if output.count(anchor) != 1:
            raise ContractError("launch cpuset anchor changed")
        output = output.replace(
            anchor, anchor + f"  --cpuset-cpus {CPUSET} \\\n"
        )
    return output


def render_runner(source: str) -> str:
    """Keep a fixed c16 cohort alive after EOS so three windows remain valid."""

    anchor = 'body = {"model": self.a.model, "temperature": self.a.temperature, "top_p": self.a.top_p,\n'
    replacement = (
        'body = {"model": self.a.model, "temperature": self.a.temperature, '
        '"top_p": self.a.top_p, "ignore_eos": True,\n'
    )
    if source.count(anchor) != 1 or '"ignore_eos": True' in source:
        raise ContractError("benchmark request anchor changed")
    return source.replace(anchor, replacement)


def prepare_cell(root: Path, cell: Cell) -> Path:
    cell_dir = root / f"cell-{cell.name}"
    if cell_dir.exists():
        raise ContractError(f"cell artifact already exists: {cell_dir}")
    cell_dir.mkdir(parents=True)
    for side in ("head", "worker"):
        rendered = render_launch(
            (FIXTURE / f"launch-{side}.sh").read_text(), cell
        )
        target = cell_dir / f"launch-{side}.sh"
        target.write_text(rendered)
        target.chmod(0o755)
    runner = cell_dir / "bench-test.py"
    runner.write_text(render_runner((RUNNER / RUNNER_REL).read_text()))
    runner.chmod(0o755)
    (cell_dir / "contract.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "cell": cell.name,
                "dual_hca": cell.dual_hca,
                "cpuset": CPUSET if cell.cpuset else None,
                "fixture": str(FIXTURE),
                "image_id": IMAGE_ID,
                "model_revision": MODEL_REVISION,
                "mtp_depth": 1,
                "runner_commit": RUNNER_COMMIT,
                "recipe_commit": RECIPE_COMMIT,
                "prompt_sha256": RUNNER_SHA256["bench/pasture-text.txt"],
                "windows": 3,
                "window_seconds": 10,
                "otel_cardinality": {"cell": 4, "rank": 2, "hca": 2},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return cell_dir


def _node(command: str, remote: bool = False) -> str:
    if remote:
        return _run(["ssh", "-o", "BatchMode=yes", WORKER, command])
    return _run(["bash", "-lc", command])


def preflight(cell: Cell) -> None:
    for remote in (False, True):
        if _node("cat /proc/sys/vm/compaction_proactiveness", remote) != "20":
            raise ContractError("vm.compaction_proactiveness must equal 20")
        if _node(
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d'",
            remote,
        ):
            raise ContractError("GPU already has a compute tenant")
        if _node(
            f"docker ps -aq --filter name=^{HEAD_CONTAINER if not remote else WORKER_CONTAINER}$",
            remote,
        ):
            raise ContractError("fixed benchmark container already exists")
        if _node(f"docker image inspect {IMAGE} --format '{{{{.Id}}}}'", remote) != IMAGE_ID:
            raise ContractError("container image identity changed")
        fixture_name = "launch-worker.sh" if remote else "run.json"
        if _node(f"sha256sum {FIXTURE}/{fixture_name} | cut -d' ' -f1", remote) != FIXTURE_SHA256[fixture_name]:
            raise ContractError("node-local K1 fixture identity changed")
        if remote and _node(f"cd {FIXTURE}/artifacts && sha256sum --check SHA256SUMS >/dev/null && echo verified", True) != "verified":
            raise ContractError("worker runtime artifact identity changed")
        required = HCA if cell.dual_hca else HCA[:1]
        for hca, netdev in zip(required, NETDEV):
            if not _node(
                f"test \"$(cat /sys/class/infiniband/{hca}/ports/1/state)\" "
                "= '4: ACTIVE' && echo active",
                remote,
            ) == "active":
                raise ContractError(f"inactive HCA: {hca}")
            if not _node(f"ip -4 -o address show dev {netdev}", remote):
                raise ContractError(f"HCA netdev has no IPv4 address: {netdev}")
    if _node("ss -H -ltn | awk '{print $4}' | grep -E '(:|])8888$' || true"):
        raise ContractError("head API port 8888 is already listening")
    if _node("ss -H -ltn | awk '{print $4}' | grep -E '(:|])50000$' || true"):
        raise ContractError("head rendezvous port 50000 is already listening")


def telemetry_snapshot() -> dict[str, object]:
    shell = """
set -e
printf 'gpu '
nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits
for h in rocep1s0f1 roceP2p1s0f1; do printf 'rdma %s ' "$h"; cat "/sys/class/infiniband/$h/ports/1/counters/port_xmit_data"; done
for n in enp1s0f1np1 enP2p1s0f1np1; do printf 'tcp %s ' "$n"; cat "/sys/class/net/$n/statistics/tx_bytes"; done
"""
    def one(rank: int) -> dict[str, object]:
        text = _node(shell, rank == 1)
        row: dict[str, object] = {"rank": rank}
        rdma, tcp = {}, {}
        for line in text.splitlines():
            fields = line.split()
            if fields[0] == "gpu":
                values = [float(value.rstrip(",")) for value in fields[1:]]
                row.update(clock_mhz=values[0], power_w=values[1], utilization_percent=values[2])
            elif fields[0] == "rdma":
                rdma[fields[1]] = int(fields[2]) * 4
            elif fields[0] == "tcp":
                tcp[fields[1]] = int(fields[2])
        if len(rdma) != 2 or len(tcp) != 2 or "clock_mhz" not in row:
            raise ContractError(f"rank {rank} telemetry is incomplete")
        row["rdma_xmit_bytes"] = rdma
        row["tcp_xmit_bytes"] = tcp
        return row

    started = time.time_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        nodes = list(pool.map(one, (0, 1)))
    return {"observed_unix_ns": time.time_ns(), "sample_span_ns": time.time_ns() - started, "nodes": nodes}


def collect_telemetry(path: Path, stop: threading.Event) -> None:
    with path.open("w") as stream:
        while not stop.is_set():
            stream.write(json.dumps(telemetry_snapshot(), sort_keys=True) + "\n")
            stream.flush()
            stop.wait(1.0)


def reduce_cell(cell: Cell, cell_dir: Path) -> dict[str, object]:
    benchmark = json.loads((cell_dir / "steady.json").read_text())
    steady = [sample for sample in benchmark["samples"] if sample.get("steady")]
    if len(steady) != 3:
        raise ContractError(f"cell {cell.name} requires exactly three steady windows")
    if any(
        sample.get("running") != 16
        or sample.get("prompt_tok") != 0
        or not 8.0 <= sample.get("dt", 0) <= 12.0
        for sample in steady
    ):
        raise ContractError(f"cell {cell.name} has a non-steady window")
    requests = benchmark.get("requests", [])
    if len(requests) != 16:
        raise ContractError(f"cell {cell.name} did not start 16 streams")
    origin = min(float(request["t0"]) for request in requests)
    telemetry = [json.loads(line) for line in (cell_dir / "telemetry.jsonl").read_text().splitlines()]
    windows = []
    for sample in steady:
        end_ns = int((origin + float(sample["t"])) * 1e9)
        start_ns = end_ns - int(float(sample["dt"]) * 1e9)
        selected = [row for row in telemetry if start_ns <= row["observed_unix_ns"] <= end_ns]
        if len(selected) < 2:
            raise ContractError(f"cell {cell.name} hardware window has fewer than two samples")
        ranks = []
        for rank in (0, 1):
            node_rows = [row["nodes"][rank] for row in selected]
            first, last = node_rows[0], node_rows[-1]
            rdma = {h: last["rdma_xmit_bytes"][h] - first["rdma_xmit_bytes"][h] for h in HCA}
            tcp = {n: last["tcp_xmit_bytes"][n] - first["tcp_xmit_bytes"][n] for n in NETDEV}
            if min((*rdma.values(), *tcp.values())) < 0:
                raise ContractError("transport counter reset during a window")
            ranks.append(
                {
                    "rank": rank,
                    "clock_mhz_median": statistics.median(row["clock_mhz"] for row in node_rows),
                    "power_w_median": statistics.median(row["power_w"] for row in node_rows),
                    "utilization_percent_median": statistics.median(row["utilization_percent"] for row in node_rows),
                    "rdma_xmit_bytes": rdma,
                    "tcp_xmit_bytes": tcp,
                }
            )
        windows.append(
            {
                "seconds": sample["dt"],
                "engine_steps_per_second": sample["steps_ps"],
                "mean_accepted_length": sample["acc_len"],
                "draft_acceptance_rate": sample["acc_len"] - 1.0,
                "acceptance_by_position": sample.get("acc_pos"),
                "aggregate_tokens_per_second": sample["gen_tps"],
                "per_stream_tokens_per_second": sample["per_stream"],
                "ranks": ranks,
            }
        )
    required_hcas = HCA if cell.dual_hca else HCA[:1]
    if any(rank["rdma_xmit_bytes"][hca] <= 0 for window in windows for rank in window["ranks"] for hca in required_hcas):
        raise ContractError(f"cell {cell.name} has a flat required RDMA HCA")
    for window in windows:
        for rank in window["ranks"]:
            rdma_total = sum(rank["rdma_xmit_bytes"].values())
            tcp_total = sum(rank["tcp_xmit_bytes"].values())
            rank["tcp_flat"] = tcp_total <= max(1_048_576, rdma_total // 50)
            if not rank["tcp_flat"]:
                raise ContractError(f"cell {cell.name} TCP transport is not flat")
    result = {
        "schema": SCHEMA,
        "cell": cell.name,
        "dual_hca": cell.dual_hca,
        "cpuset": CPUSET if cell.cpuset else None,
        "windows": windows,
    }
    (cell_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def wait_healthy(timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        if _node(f"docker inspect -f '{{{{.State.Running}}}}' {HEAD_CONTAINER}") != "true":
            raise ContractError("head container exited during startup")
        if _node(f"docker inspect -f '{{{{.State.Running}}}}' {WORKER_CONTAINER}", True) != "true":
            raise ContractError("worker container exited during startup")
        time.sleep(5)
    raise ContractError("server health timeout")


def cleanup() -> None:
    subprocess.run(["docker", "rm", "-f", HEAD_CONTAINER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER, "docker", "rm", "-f", WORKER_CONTAINER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_cleanup(timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not any(
            _node(
                "nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                "| sed '/^$/d'",
                remote,
            )
            for remote in (False, True)
        ):
            return
        time.sleep(2)
    raise ContractError("CUDA tenants remained after cell cleanup")


def run_cell(root: Path, cell: Cell, timeout_seconds: int) -> dict[str, object]:
    preflight(cell)
    cell_dir = prepare_cell(root, cell)
    _run(
        [
            "ssh", "-o", "BatchMode=yes", WORKER,
            f"test ! -e {shlex.quote(str(cell_dir))} && "
            f"mkdir -p {shlex.quote(str(cell_dir))}",
        ]
    )
    _run(["scp", "-q", str(cell_dir / "launch-worker.sh"), f"{WORKER}:{cell_dir}/launch-worker.sh"])
    log_files = []
    followers = []
    stop = threading.Event()
    telemetry_thread = None
    try:
        _run(["ssh", "-o", "BatchMode=yes", WORKER, "bash", str(cell_dir / "launch-worker.sh")])
        time.sleep(5)
        _run(["bash", str(cell_dir / "launch-head.sh")])
        pid = int(_node(f"docker inspect -f '{{{{.State.Pid}}}}' {HEAD_CONTAINER}"))
        (root / "active.json").write_text(json.dumps({"cell": cell.name, "pid": pid, "artifact": str(cell_dir)}) + "\n")
        for name, command in (
            ("head.log", ["docker", "logs", "--timestamps", "-f", HEAD_CONTAINER]),
            ("worker.log", ["ssh", "-o", "BatchMode=yes", WORKER, "docker", "logs", "--timestamps", "-f", WORKER_CONTAINER]),
        ):
            stream = (cell_dir / name).open("w")
            log_files.append(stream)
            followers.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True))
        wait_healthy(timeout_seconds)
        telemetry_thread = threading.Thread(target=collect_telemetry, args=(cell_dir / "telemetry.jsonl", stop), daemon=True)
        telemetry_thread.start()
        command = [
            "python3", str(cell_dir / "bench-test.py"), "--url", "http://127.0.0.1:8888",
            "--model", "qwen3.8-flash-next", "--c", "16", "--seconds", "30",
            "--warmup", "60", "--sample", "10", "--max-tokens", "32768",
            "--thinking", "off", "--prompt", str(RUNNER / "bench/pasture-text.txt"),
            "--temperature", "0.6", "--top-p", "0.95", "--k", "1",
            "--tag", f"rocket-2x2-{cell.name}", "--lane", f"rocket-2x2-{cell.name}",
            "--json", str(cell_dir / "steady.json"),
        ]
        benchmark_stdout = _run(command, timeout=900)
        (cell_dir / "benchmark.stdout").write_text(benchmark_stdout + "\n")
        stop.set()
        telemetry_thread.join(timeout=30)
        if telemetry_thread.is_alive():
            raise ContractError("telemetry collector did not stop")
        return reduce_cell(cell, cell_dir)
    finally:
        stop.set()
        if telemetry_thread is not None:
            telemetry_thread.join(timeout=30)
        cleanup()
        wait_cleanup()
        for follower in followers:
            follower.terminate()
            try:
                follower.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follower.kill()
        for stream in log_files:
            stream.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cells", nargs="+", choices=tuple(CELLS), default=list(CELLS))
    parser.add_argument("--startup-timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists() or args.startup_timeout_seconds < 1:
        parser.error("--output must be a new absolute path and timeout must be positive")
    validate_sources()
    args.output.mkdir(parents=True)
    results = []
    for name in args.cells:
        dual, pinned = CELLS[name]
        results.append(run_cell(args.output, Cell(name, dual, pinned), args.startup_timeout_seconds))
    (args.output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (args.output / "active.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
