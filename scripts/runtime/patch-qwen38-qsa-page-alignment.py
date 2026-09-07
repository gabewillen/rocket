#!/usr/bin/env python3
"""Make pinned vLLM include Qwen3.8's QSA ring in page alignment."""

import argparse
import ast
from pathlib import Path

MARKER = "# ROCKET_QWEN38_QSA_PAGE_ALIGNMENT_V1\n"
ANCHOR = """            if model_config.use_mla:
                # TRTLLM/FlashInfer MLA decode kernels require the physical
                # number of kernel blocks to be aligned to 128 / kernel_block_size.
                # For hybrid MLA/Mamba models, make the manager block size a
                # multiple of 128 so split kernel blocks keep that invariant.
                kernel_block_alignment_size = max(kernel_block_alignment_size, 128)
"""
PATCH = ANCHOR + MARKER + """            if model_config.hf_text_config.model_type == "qwen4_exp_text":
                # QSA keeps an open compression group plus every speculative
                # row until acceptance. Include that whole-group ring in the
                # same LCM used to size the hybrid attention/Mamba page.
                compress_ratio = model_config.hf_text_config.indexer_compress_ratio
                if not isinstance(compress_ratio, int) or compress_ratio <= 0:
                    raise ValueError("Qwen3.8 QSA compression ratio must be positive")
                qsa_span = compress_ratio + vllm_config.num_speculative_tokens
                qsa_capacity = compress_ratio * cdiv(qsa_span, compress_ratio)
                kernel_block_alignment_size = lcm(
                    kernel_block_alignment_size, qsa_capacity
                )
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("platform_py", type=Path)
    args = parser.parse_args()
    source = args.platform_py.read_text()
    if MARKER.strip() in source:
        raise SystemExit("already patched")
    if source.count(ANCHOR) != 1:
        raise SystemExit("vLLM source drift: expected one hybrid alignment anchor")
    source = source.replace(ANCHOR, PATCH)
    ast.parse(source, filename=str(args.platform_py))
    args.platform_py.write_text(source)


if __name__ == "__main__":
    main()
