#!/usr/bin/env python3
"""Patch vLLM to stage one safetensors tensor off a 64 KiB mmap."""

import argparse
from pathlib import Path

OLD = """                    param = f.get_tensor(name)\n                    yield name, param\n"""
NEW = """                    # Bound staging to one tensor and avoid CUDA copying directly\n                    # from a 64 KiB-page safetensors mmap.\n                    param = f.get_tensor(name).clone()\n                    yield name, param\n"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_utils", type=Path)
    args = parser.parse_args()
    source = args.weight_utils.read_text()
    if NEW in source:
        raise SystemExit("already patched")
    if source.count(OLD) != 1:
        raise SystemExit("vLLM source drift: expected one lazy-loader assignment")
    args.weight_utils.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    main()
