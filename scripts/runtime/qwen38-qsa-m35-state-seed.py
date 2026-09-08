#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json

from qwen38_slab.qsa_m35_state_seed import execute_qsa_m35_state_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=__import__("pathlib").Path)
    parser.add_argument("--library", required=True, type=__import__("pathlib").Path)
    args = parser.parse_args()
    print(json.dumps(execute_qsa_m35_state_seed(args.bundle, args.library), sort_keys=True))


if __name__ == "__main__":
    main()
