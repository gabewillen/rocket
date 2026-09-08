#!/usr/bin/env python3
"""Add a bounded layer-0 boundary capture to the pinned K0 oracle overlay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SHA256 = "77ab4fdcb2ded4bca03e329b1a542b88924f64bf924cf4cf8d094d77f313b1c0"


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise RuntimeError("pinned model oracle patch anchor differs")
    return source.replace(before, after, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = args.input.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("pinned model oracle source identity differs")
    source = payload.decode()

    helper = r'''
_ROCKET_K0_BOUNDARY_NAMES = (
    "attention_output",
    "hc_combine_mix",
    "moe_output",
)
_ROCKET_K0_BOUNDARY_SEEN = set()


def _rocket_k0_boundary_save(layer_idx, name, tensor):
    if layer_idx != 0 or os.getenv("ROCKET_QWEN38_K0_BOUNDARY_CAPTURE") != "1":
        return
    # Startup profiling executes layer 0 with dummy zero inputs before the
    # authenticated request. The oracle is active only across the target
    # forward, so this gate excludes every warmup invocation.
    if _ROCKET_K0_ORACLE is None or not _ROCKET_K0_ORACLE.active_forward:
        return
    from vllm.distributed import get_tensor_model_parallel_rank
    if get_tensor_model_parallel_rank() != 0:
        return
    if name not in _ROCKET_K0_BOUNDARY_NAMES:
        raise RuntimeError("layer0 boundary name differs")
    if len(_ROCKET_K0_BOUNDARY_SEEN) >= len(_ROCKET_K0_BOUNDARY_NAMES):
        return
    expected = _ROCKET_K0_BOUNDARY_NAMES[len(_ROCKET_K0_BOUNDARY_SEEN)]
    if name != expected or name in _ROCKET_K0_BOUNDARY_SEEN:
        raise RuntimeError("layer0 boundary order differs")
    value = tensor[:1].detach().contiguous()
    if value.dtype != torch.bfloat16 or value.ndim != 2:
        raise RuntimeError("layer0 boundary layout differs")
    output = Path(os.environ["ROCKET_QWEN38_K0_BOUNDARY_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    payload = value.view(torch.uint8).cpu().numpy().tobytes()
    destination = output / (name + ".bin")
    temporary = output / ("." + name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    record = {
        "schema": "rocket.qwen38.k0-layer0-boundaries.v1",
        "name": name,
        "dtype": "bfloat16",
        "shape": list(value.shape),
        "strides": list(value.stride()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    metadata = output / (name + ".json")
    temporary = output / ("." + name + ".json.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, metadata)
    _ROCKET_K0_BOUNDARY_SEEN.add(name)
'''
    source = replace_once(
        source,
        "_ROCKET_K0_ORACLE = None\n\n\nclass _RocketK0Oracle:",
        "_ROCKET_K0_ORACLE = None\n" + helper + "\n\nclass _RocketK0Oracle:",
    )
    source = replace_once(
        source,
        '''        if not actual or actual != expected:
            raise RuntimeError(f"input token IDs differ at offset {self.consumed_tokens}: expected {expected}, got {actual}")
''',
        '''        if not actual or actual != expected:
            # vLLM startup profiling runs dummy token IDs before the service is
            # reachable. Leave the oracle disarmed until the exact request.
            if self.consumed_tokens == 0 and not self.active_forward:
                return
            raise RuntimeError(f"input token IDs differ at offset {self.consumed_tokens}: expected {expected}, got {actual}")
''',
    )
    source = replace_once(
        source,
        '''    if oracle is None:
        return
    try:
        callback(oracle)
''',
        '''    if oracle is None:
        return
    if phase != "embedding" and not oracle.active_forward:
        return
    try:
        callback(oracle)
''',
    )
    source = replace_once(
        source,
        """        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection
""",
        """        _rocket_k0_boundary_save(self.layer_idx, "attention_output", attn_out)
        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection
        )
        _rocket_k0_boundary_save(self.layer_idx, "hc_combine_mix", hidden_states)
        mlp_out = self.mlp(block_input)
        _rocket_k0_boundary_save(self.layer_idx, "moe_output", mlp_out)
        return hidden_states, mlp_out, injection
""",
    )
    args.output.write_text(source)


if __name__ == "__main__":
    main()
