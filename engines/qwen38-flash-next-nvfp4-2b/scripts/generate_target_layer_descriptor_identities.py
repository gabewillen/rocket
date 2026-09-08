#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the checked-in 96-entry target-layer descriptor allowlist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen38_slab.target_layer_descriptor import descriptor_identity, target_layer_descriptor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identities = [
        descriptor_identity(target_layer_descriptor(args.artifact, args.sidecar, rank, layer))
        for rank in (0, 1) for layer in range(48)
    ]
    args.output.write_text(json.dumps(identities, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
