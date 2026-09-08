#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only provenance and full-payload contract for fixed layer-3 RoPE."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/attention/layer3_rope_owner.h"
SOURCE = ROOT / "src/attention/layer3_rope_owner.cc"
BITS = ROOT / "src/attention/layer3_rope_bits.inc"
CONFIG = Path(
    "/home/glwillen/.cache/huggingface/hub/"
    "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/"
    "fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json"
)
CONFIG_SHA256 = "deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80"
PAYLOAD_SHA256 = "f22ad8a42a36ec8078d7f43a89bdb98cbb05ef7139c34b09a7cd5af62b98a516"


def main() -> None:
    header = HEADER.read_text()
    source = SOURCE.read_text()
    words = [int(item, 16) for item in re.findall(r"0x[0-9a-f]{4}", BITS.read_text())]
    assert len(words) == 35 * 64
    payload = struct.pack(f"<{len(words)}H", *words)
    assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA256
    assert PAYLOAD_SHA256 in header
    assert "58f4d1ed59074a181c9db6e4c5a950db14e7ad0b" in header
    assert "fc694b54fb0174e0913e6adf86691ef85a4ead47" in header
    assert "8e685d198" in header
    assert "cudaStreamNonBlocking" in source
    assert "cudaEventRecord(ready_, initialization_stream_)" in source
    assert "cudaStreamWaitEvent(consumer_stream, ready_, 0)" in source
    assert "cudaEventSynchronize" not in source
    assert "cudaDeviceSynchronize" not in source
    assert "torch" not in source.lower()
    assert "python" not in source.lower()
    if CONFIG.exists():
        assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
        config = json.loads(CONFIG.read_text())["text_config"]
        assert config["layer_types"][3] == "full_attention"
        assert config["head_dim"] == 256
        assert config["partial_rotary_factor"] == 0.25
        assert config["rope_parameters"] == {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000_000,
            "rope_type": "default",
        }
    print(
        "layer3_rope valid=1 complete=1 phase=source_contract "
        f"words={len(words)} sha256={PAYLOAD_SHA256}"
    )


if __name__ == "__main__":
    main()
