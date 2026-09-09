#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extend the authenticated K0 oracle through one stateful c1 continuation."""

import argparse
import hashlib
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "4f981e92e18003feb4a075060cff076434178484f4d5c8f1c29c2942da282f06"
)


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("pinned K0 oracle c1 anchor changed")
    return source.replace(before, after, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PINNED_SOURCE_SHA256:
        raise RuntimeError("pinned K0 oracle identity changed")
    source = payload.decode()
    source = replace_once(
        source,
        "        self.complete = False\n\n    def fail(self, phase, error):",
        "        self.complete = False\n"
        "        self.c1_token = None\n"
        "        self.c1_active = False\n\n"
        "    def fail(self, phase, error):",
    )
    source = replace_once(
        source,
        "    def save(self, name, tensor):\n"
        "        if not self.active_forward:\n",
        "    def save(self, name, tensor):\n"
        "        if self.c1_active:\n"
        "            return\n"
        "        if not self.active_forward:\n",
    )
    source = replace_once(
        source,
        "    def begin(self, input_ids, embedding):\n"
        "        if self.complete:\n",
        "    def begin(self, input_ids, embedding):\n"
        "        if self.c1_token is not None and not self.c1_active:\n"
        "            actual = input_ids.detach().cpu().tolist()\n"
        "            if actual != [self.c1_token]:\n"
        "                raise RuntimeError(\"authenticated c1 token differs\")\n"
        "            if embedding.ndim != 2 or tuple(embedding.shape) != (1, 2560):\n"
        "                raise RuntimeError(\"authenticated c1 embedding extent differs\")\n"
        "            self.active_forward = True\n"
        "            self.c1_active = True\n"
        "            self.forward_names = []\n"
        "            return\n"
        "        if self.complete:\n",
    )
    source = replace_once(
        source,
        "    def finish(self, logits):\n"
        "        if not self.active_forward or self.complete:\n",
        "    def finish(self, logits):\n"
        "        if self.c1_active:\n"
        "            if not self.active_forward or self.complete:\n"
        "                raise RuntimeError(\"oracle c1 logits arrived outside continuation\")\n"
        "            self.active_forward = False\n"
        "            self.c1_active = False\n"
        "            self.complete = True\n"
        "            return\n"
        "        if not self.active_forward or self.complete:\n",
    )
    source = replace_once(
        source,
        "        self.complete = True\n"
        "        print(\"ROCKET_QWEN38_K0_ORACLE_COMPLETE\\t\" + json.dumps({",
        "        self.c1_token = int(indices[0])\n"
        "        print(\"ROCKET_QWEN38_K0_ORACLE_PREFILL_COMPLETE\\t\" + json.dumps({",
    )
    args.output.write_text(source)


if __name__ == "__main__":
    main()
