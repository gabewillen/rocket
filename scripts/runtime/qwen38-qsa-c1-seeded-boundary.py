#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Authenticate row-35 QSA input and seed its host-native c1 state."""

import argparse
import json
from pathlib import Path

from qwen38_slab.qsa_m35_state_seed import execute_qsa_c1_seeded_boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        execute_qsa_c1_seeded_boundary(args.bundle, args.library),
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
