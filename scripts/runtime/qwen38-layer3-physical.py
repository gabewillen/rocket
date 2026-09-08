#!/usr/bin/env python3
"""Authenticate the exact two-rank layer-3 physical launch inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines/qwen38-flash-next-nvfp4-2b/src"))

from qwen38_slab.layer3_factory import (  # noqa: E402
    Layer3FactoryError, native_rank_descriptor, prepare_layer3_physical_plan,
    public_plan,
)


class _Span:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): return None
    def set_attribute(self, key, value): pass
    def record_exception(self, exception): pass


class _Tracer:
    def start_as_current_span(self, name): return _Span()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--indexer-sidecar", type=Path, required=True)
    parser.add_argument("--oracle-capture", type=Path, required=True)
    parser.add_argument("--bootstrap-host", required=True)
    parser.add_argument("--bootstrap-port", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--rank", type=int, choices=(0, 1))
    parser.add_argument("--native-plan", type=Path)
    parser.add_argument("--preflight-only", action="store_true", required=True)
    args = parser.parse_args()
    try:
        plan = prepare_layer3_physical_plan(
            artifact=args.artifact, indexer_sidecar=args.indexer_sidecar,
            oracle_capture=args.oracle_capture, tracer=_Tracer(),
            bootstrap_host=args.bootstrap_host,
            bootstrap_port=args.bootstrap_port, timeout_ms=args.timeout_ms,
        )
    except Exception as exc:
        print(json.dumps({
            "schema": "rocket.qwen38.layer3-physical-plan.v1",
            "valid": False, "complete": False, "phase": "preflight",
            "failure_class": (
                "io" if isinstance(exc, OSError)
                else "contract" if isinstance(exc, Layer3FactoryError)
                else "dependency"
            ),
            "reason": str(exc)[:512],
        }, sort_keys=True))
        return 1
    if args.native_plan is not None or args.rank is not None:
        if args.native_plan is None or args.rank is None:
            print(json.dumps({
                "schema": "rocket.qwen38.layer3-physical-plan.v1",
                "valid": False, "complete": False, "phase": "handoff",
                "failure_class": "contract",
                "reason": "--rank and --native-plan must be supplied together",
            }, sort_keys=True))
            return 1
        descriptor = dict(native_rank_descriptor(plan, args.rank))
        args.native_plan.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{args.native_plan.name}.", dir=args.native_plan.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, args.native_plan)
        except Exception:
            try: os.unlink(temporary)
            except FileNotFoundError: pass
            raise
    print(json.dumps(dict(public_plan(plan)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
