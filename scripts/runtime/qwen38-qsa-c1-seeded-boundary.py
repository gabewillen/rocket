#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Authenticate row-35 QSA input and seed its host-native c1 state."""

import argparse
import json
from pathlib import Path

from qwen38_slab.qsa_c1_composite import execute_qsa_c1_composite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--seed-library", required=True, type=Path)
    parser.add_argument("--graph-library", required=True, type=Path)
    parser.add_argument("--rope-library", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        execute_qsa_c1_composite(
            args.bundle, args.seed_library, args.graph_library,
            args.rope_library, args.artifact, args.sidecar,
        ),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
