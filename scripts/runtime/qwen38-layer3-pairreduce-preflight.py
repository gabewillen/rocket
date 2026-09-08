#!/usr/bin/env python3
"""No-launch validation of one rank's exact layer-3 PairReduce endpoint."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path

SESSION = "05ea3af1c4694a9c035ce2fe9ce006acc58881df0fe86771b1846f4bd8e5f48b"
RAILS = ("rocep1s0f1", "roceP2p1s0f1")
SCHEMA = "rocket.qwen38.layer3-pairreduce-preflight.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    if args.rank not in (0, 1) or args.peer_rank != 1 - args.rank:
        raise ValueError("rank/peer identity changed")
    ipaddress.IPv4Address(args.bootstrap_host)
    if not 1 <= args.bootstrap_port <= 65_535:
        raise ValueError("bootstrap port changed")
    if not 100 <= args.timeout_ms <= 120_000:
        raise ValueError("operation timeout changed")
    session_source = "embedded"
    if args.oracle_manifest is not None:
        if sha256(args.oracle_manifest) != SESSION:
            raise ValueError("oracle session identity changed")
        session_source = "authenticated_manifest"
    rail_records = []
    for rail in RAILS:
        root = args.infiniband_sysfs / rail / "ports" / "1"
        state = (root / "state").read_text().strip()
        gid = (root / "gids" / "3").read_text().strip().lower()
        if not state.startswith("4: ACTIVE"):
            raise ValueError(f"RDMA rail inactive: {rail}")
        if not gid or gid in {"::", "0:0:0:0:0:0:0:0"}:
            raise ValueError(f"RDMA GID 3 unavailable: {rail}")
        rail_records.append({"device": rail, "port": 1, "gid_index": 3,
                             "state": "ACTIVE"})
    return {
        "schema": SCHEMA, "valid": True, "complete": True,
        "phase": "no_launch_preflight", "rank": args.rank,
        "peer_rank": args.peer_rank, "bootstrap_host": args.bootstrap_host,
        "bootstrap_port": args.bootstrap_port, "timeout_ms": args.timeout_ms,
        "session_sha256": SESSION, "rails": rail_records,
        "session_source": session_source,
        "calls": 70, "schedule": "35x(attention_m1,moe_m1)",
        "cuda_launches": 0, "rdma_connections": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--peer-rank", type=int, required=True)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--bootstrap-port", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--oracle-manifest", type=Path)
    parser.add_argument("--infiniband-sysfs", type=Path,
                        default=Path("/sys/class/infiniband"))
    args = parser.parse_args()
    try:
        record = preflight(args)
    except Exception as exc:
        record = {
            "schema": SCHEMA, "valid": False, "complete": False,
            "phase": "no_launch_preflight", "rank": args.rank,
            "failure_class": "io" if isinstance(exc, OSError) else "contract",
            "reason": str(exc)[:512], "cuda_launches": 0,
            "rdma_connections": 0,
        }
        print(json.dumps(record, sort_keys=True))
        return 1
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
