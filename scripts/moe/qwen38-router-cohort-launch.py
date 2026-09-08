#!/usr/bin/env python3
"""Build and optionally execute the authenticated two-node c16 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "qwen38-router-cohort-launch-v4.json"
SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    "fc694b54fb0174e0913e6adf86691ef85a4ead47"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind(source: str, destination: str, read_only: bool = True) -> dict:
    return {
        "source": source,
        "destination": destination,
        "read_only": read_only,
    }


def _paths(manifest: dict, home: Path, calibration: Path) -> dict[str, str]:
    common = manifest["common_artifacts"]
    return {
        "driver": str(calibration / manifest["driver"]["path"]),
        "model": str(calibration / manifest["model_telemetry"]["path"]),
        "config": str(calibration / common["config"]["path"]),
        "modelopt": str(calibration / common["modelopt"]["path"]),
        "ple": str(calibration / common["ple"]["path"]),
        "qsa_nvidia": str(calibration / common["qsa_nvidia"]["path"]),
        "qsa_ops": str(calibration / common["qsa_ops"]["path"]),
        "weight_utils": str(calibration / common["weight_utils"]["path"]),
        "vllm_cache": str(home / ".cache/vllm"),
    }


def build_binds(node: str, manifest: dict, home: Path, calibration: Path) -> list[dict]:
    paths = _paths(manifest, home, calibration)
    node_config = manifest["nodes"][node]
    overlay = str(calibration / node_config["overlay"]["path"])
    hf_cache = node_config["hf_cache"].replace("${HOME}", str(home))
    config_destination = f"{SNAPSHOT}/config.json"
    quant_destination = f"{SNAPSHOT}/hf_quant_config.json"
    if node == "head":
        return [
            _bind(paths["weight_utils"], "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"),
            _bind(paths["model"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py"),
            _bind(paths["qsa_ops"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"),
            _bind(paths["qsa_nvidia"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/qsa.py"),
            _bind(paths["vllm_cache"], "/root/.cache/vllm", False),
            _bind(paths["driver"], "/work/qwen38-router-cohort-live.py"),
            _bind(paths["ple"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"),
            _bind(paths["modelopt"], "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"),
            _bind(hf_cache, "/root/.cache/huggingface"),
            _bind(paths["config"], config_destination),
            _bind(f"{overlay}/hf_quant_config.json", quant_destination),
            _bind(overlay, "/rocket/qwen38-linear-nvfp4"),
        ]
    return [
        _bind(overlay, "/rocket/qwen38-linear-nvfp4"),
        _bind(paths["ple"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"),
        _bind(paths["modelopt"], "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"),
        _bind(paths["weight_utils"], "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py"),
        _bind(paths["qsa_ops"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"),
        _bind(hf_cache, "/root/.cache/huggingface"),
        _bind(paths["config"], config_destination),
        _bind(f"{overlay}/hf_quant_config.json", quant_destination),
        _bind(paths["model"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py"),
        _bind(paths["qsa_nvidia"], "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/qsa.py"),
        _bind(paths["vllm_cache"], "/root/.cache/vllm", False),
        _bind(paths["driver"], "/work/qwen38-router-cohort-live.py"),
    ]


def validate_node_plan(node: str, plan: dict, manifest: dict) -> None:
    argv = plan["docker_argv"]
    entrypoint_index = argv.index("--entrypoint")
    if argv[entrypoint_index + 1] != manifest["entrypoint"]:
        raise ValueError(f"{node}: explicit python3 entrypoint is required")
    if entrypoint_index >= argv.index(manifest["image"]):
        raise ValueError(f"{node}: entrypoint must be a Docker option")
    destinations = [item["destination"] for item in plan["binds"]]
    argv_destinations = [
        argv[index + 1].rsplit(":", 2)[1]
        for index, item in enumerate(argv)
        if item == "-v"
    ]
    if argv_destinations != destinations:
        raise ValueError(f"{node}: Docker argv bind ordering differs from plan")
    expected = manifest["nodes"][node]["bind_destination_order"]
    if destinations != expected:
        raise ValueError(f"{node}: bind ordering differs from successful contract")
    parent = destinations.index("/root/.cache/huggingface")
    for child in (f"{SNAPSHOT}/config.json", f"{SNAPSHOT}/hf_quant_config.json"):
        if parent >= destinations.index(child):
            raise ValueError(f"{node}: HF cache parent must precede child bind {child}")


def build_plan(manifest: dict, home: Path, calibration: Path, port: int, suffix: str) -> dict:
    workload = manifest["workload"]
    expected_workload = {
        "concurrency": 16,
        "decode": 24,
        "prefix_tokens": 6304,
        "divergence_tokens": 128,
        "expected_cache_block_size": 3216,
        "expected_prompt_tokens": 6433,
        "expected_cached_tokens": 6432,
        "cache_geometry": "two_cache_pages",
        "pool_with_c1_c8_prompt_distribution": False,
    }
    if workload != expected_workload:
        raise ValueError("manifest workload differs from the c16 geometry contract")
    common_env = {
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_OFFLINE": "1",
        "NCCL_SOCKET_IFNAME": "enp1s0f1np1",
        "TP_SOCKET_IFNAME": "enp1s0f1np1",
        "GLOO_SOCKET_IFNAME": "enp1s0f1np1",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_HCA": "rocep1s0f1",
        "NCCL_IB_AUTO_DETECT": "0",
        "NCCL_DEBUG": "WARN",
        "ROCKET_NVFP4_CALIBRATE": "1",
        "ROCKET_NVFP4_SAMPLE_ELEMENTS": "2048",
        "ROCKET_NVFP4_MAX_EMISSIONS": "12",
        "ROCKET_QWEN38_LOAD_TRACE": "1",
        "ROCKET_QWEN38_NVFP4_OVERLAY_MANIFEST": "/rocket/qwen38-linear-nvfp4/manifest.json",
        "ROCKET_QWEN38_NVFP4_QUANT_CONFIG": "/rocket/qwen38-linear-nvfp4/hf_quant_config.json",
    }
    nodes = {}
    for rank, node in enumerate(("head", "worker")):
        binds = build_binds(node, manifest, home, calibration)
        name = f"rocket-qwen38-router-v4-c16-{node}-{suffix}"
        env = {**common_env, "VLLM_HOST_IP": manifest["nodes"][node]["host_ip"]}
        argv = [
            "docker", "run", "-d", "--name", name,
            "--gpus", "all", "--network", "host", "--ipc", "host",
            "--cap-add", "SYS_NICE", "--security-opt", "label=disable",
            "--ulimit", "memlock=-1:-1", "--ulimit", "stack=67108864:67108864",
            "--device", "/dev/infiniband:/dev/infiniband",
            "--entrypoint", manifest["entrypoint"],
        ]
        for key, value in env.items():
            argv.extend(("-e", f"{key}={value}"))
        for item in binds:
            mode = "ro" if item["read_only"] else "rw"
            argv.extend(("-v", f'{item["source"]}:{item["destination"]}:{mode}'))
        argv.extend(
            (
                manifest["image"], "-m", "torch.distributed.run", "--nnodes=2",
                "--nproc-per-node=1", f"--node-rank={rank}",
                "--master-addr=192.168.100.10", f"--master-port={port}",
                "/work/qwen38-router-cohort-live.py", "--concurrency",
                str(workload["concurrency"]),
                "--decode", str(workload["decode"]),
                "--prefix-tokens", str(workload["prefix_tokens"]),
                "--divergence-tokens", str(workload["divergence_tokens"]),
                "--expected-cache-block-size",
                str(workload["expected_cache_block_size"]),
            )
        )
        nodes[node] = {"name": name, "rank": rank, "env": env, "binds": binds, "docker_argv": argv}
    plan = {
        "schema": "rocket.qwen38.router-cohort-launch-dry-run.v1",
        "port": port,
        "suffix": suffix,
        "workload": workload,
        "image": manifest["image"],
        "driver_sha256": manifest["driver"]["sha256"],
        "model_telemetry_sha256": manifest["model_telemetry"]["sha256"],
        "common_artifact_sha256": {
            key: value["sha256"]
            for key, value in manifest["common_artifacts"].items()
        },
        "overlay_sha256": {
            node: {
                "manifest": manifest["nodes"][node]["overlay"]["manifest_sha256"],
                "quant_config": manifest["nodes"][node]["overlay"]["quant_sha256"],
            }
            for node in ("head", "worker")
        },
        "nodes": nodes,
    }
    for node in nodes:
        validate_node_plan(node, nodes[node], manifest)
    return plan


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def preflight(plan: dict, manifest: dict, home: Path, calibration: Path, remote: str) -> None:
    paths = _paths(manifest, home, calibration)
    for key, manifest_key in (("driver", "driver"), ("model", "model_telemetry")):
        expected = manifest[manifest_key]["sha256"]
        if _sha256(Path(paths[key])) != expected:
            raise RuntimeError(f"head {key} SHA-256 mismatch")
        result = _run(["ssh", remote, "sha256sum", paths[key]])
        if result.returncode or result.stdout.split()[0] != expected:
            raise RuntimeError(f"worker {key} SHA-256 mismatch")
    for key, artifact in manifest["common_artifacts"].items():
        expected = artifact["sha256"]
        if _sha256(Path(paths[key])) != expected:
            raise RuntimeError(f"head {key} SHA-256 mismatch")
        result = _run(["ssh", remote, "sha256sum", paths[key]])
        if result.returncode or result.stdout.split()[0] != expected:
            raise RuntimeError(f"worker {key} SHA-256 mismatch")
    for node, prefix in (("head", []), ("worker", ["ssh", remote])):
        overlay = calibration / manifest["nodes"][node]["overlay"]["path"]
        for filename, field in (("manifest.json", "manifest_sha256"), ("hf_quant_config.json", "quant_sha256")):
            expected = manifest["nodes"][node]["overlay"][field]
            result = _run(prefix + ["sha256sum", str(overlay / filename)])
            if result.returncode or result.stdout.split()[0] != expected:
                raise RuntimeError(f"{node}: overlay {filename} SHA-256 mismatch")
    for node, prefix in (("head", []), ("worker", ["ssh", remote])):
        gpu = _run(prefix + ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
        if gpu.returncode or gpu.stdout.strip():
            raise RuntimeError(f"{node}: GPU process table is not empty")
        names = _run(prefix + ["docker", "ps", "-a", "--format", "{{.Names}}"])
        if names.returncode or plan["nodes"][node]["name"] in names.stdout.splitlines():
            raise RuntimeError(f"{node}: container name is unavailable")
        sockets = _run(prefix + ["ss", "-ltn"])
        if sockets.returncode or f":{plan['port']} " in sockets.stdout:
            raise RuntimeError(f"{node}: rendezvous port is unavailable")


def execute(plan: dict, remote: str) -> dict[str, str]:
    worker_command = shlex.join(plan["nodes"]["worker"]["docker_argv"])
    worker = _run(["ssh", remote, worker_command])
    if worker.returncode:
        raise RuntimeError(f"worker launch failed: {worker.stderr.strip()}")
    head = _run(plan["nodes"]["head"]["docker_argv"])
    if head.returncode:
        raise RuntimeError(
            f"head launch failed after worker {worker.stdout.strip()}: {head.stderr.strip()}"
        )
    return {"worker": worker.stdout.strip(), "head": head.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-root", type=Path, default=Path.home() / "calibration")
    parser.add_argument("--remote", default="192.168.100.11")
    parser.add_argument("--port", type=int, default=50187)
    parser.add_argument("--suffix", default="geometry-r7")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = build_plan(manifest, Path.home(), args.calibration_root, args.port, args.suffix)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    preflight(plan, manifest, Path.home(), args.calibration_root, args.remote)
    print(json.dumps({"plan": plan, "handles": execute(plan, args.remote)}, sort_keys=True))


if __name__ == "__main__":
    main()
