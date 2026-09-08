#!/usr/bin/env python3
"""Fail-closed two-rank layer-3 production composition entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engines/qwen38-flash-next-nvfp4-2b"
sys.path.insert(0, str(ENGINE / "src"))

from qwen38_slab.layer3_runtime import (  # noqa: E402
    REQUIRED_LAYER3_DEPENDENCIES,
    Layer3RuntimeError,
    TwoRankLayer3Factory,
)

SCHEMA = "rocket.qwen38.k0-layer3-two-rank-preflight.v1"


def load(spec: str):
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise Layer3RuntimeError("dependency factory must use module:callable")
    factory = getattr(importlib.import_module(module_name), factory_name, None)
    if not callable(factory):
        raise Layer3RuntimeError("dependency factory is not callable")
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dependency", action="append", default=[], metavar="NAME=MODULE:FACTORY"
    )
    args = parser.parse_args()
    dependencies = {}
    try:
        for raw in args.dependency:
            name, separator, spec = raw.partition("=")
            if not separator or name in dependencies:
                raise Layer3RuntimeError("duplicate or malformed dependency argument")
            dependencies[name] = load(spec)
        binding = TwoRankLayer3Factory().bind(dependencies)
    except Layer3RuntimeError as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "valid": False,
            "complete": False,
            "phase": "bind",
            "missing": list(exc.missing),
            "reason": str(exc)[:512],
        }, sort_keys=True))
        return 1
    print(json.dumps({
        "schema": SCHEMA,
        "valid": True,
        "complete": True,
        "phase": "bind",
        "dependencies": list(binding.dependencies),
        "launch": "factory_bound",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
