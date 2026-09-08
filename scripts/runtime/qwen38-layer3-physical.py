#!/usr/bin/env python3
"""Authenticate the exact two-rank layer-3 physical launch inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines/qwen38-flash-next-nvfp4-2b/src"))

from qwen38_slab.layer3_factory import (  # noqa: E402
    Layer3FactoryError, prepare_layer3_physical_plan, public_plan,
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
    print(json.dumps(dict(public_plan(plan)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
