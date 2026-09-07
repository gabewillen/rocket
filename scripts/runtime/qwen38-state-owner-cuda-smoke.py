#!/usr/bin/env python3
"""Exercise Qwen accepted-state ownership on a real Torch CUDA device."""

from __future__ import annotations

import hashlib

import torch

from qwen38_slab.device_decode import DevicePhase, DevicePublication
from qwen38_slab.runtime_state import CudaStateBinding, DeviceState, RuntimeBoundary
from qwen38_slab.state_owner import DecoderStateOwner, OwnerPhase
from qwen38_slab.state_txn import AuthenticatedState, FamilyPayload, STATE_FAMILIES
from qwen38_slab.torch_cuda import TorchCudaRuntime


class _Span:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def set_attribute(self, key, value):
        del key, value

    def record_exception(self, exception):
        del exception


class _Tracer:
    def start_as_current_span(self, name):
        del name
        return _Span()


class _Decoder:
    def __init__(self):
        self.phase = DevicePhase.IDLE
        self.publication = None

    def upload_and_launch(self, generation):
        self.publication = DevicePublication(generation, 1, generation % 2)
        return self.publication


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = "cuda:0"
    tracer = _Tracer()
    decoder = _Decoder()
    owner = DecoderStateOwner(rank=0, decoder=decoder, tracer=tracer)
    token_hash = hashlib.sha256(b"qwen38-state-owner-smoke").hexdigest()
    boundary = RuntimeBoundary(1, token_hash, 1)
    owner.accept_boundary(owner.upload_and_launch(1), boundary)
    runtime = TorchCudaRuntime(
        owner=owner,
        compute_streams=(torch.cuda.Stream(device=device),),
        torch_api=torch,
        device=device,
    )
    binding = CudaStateBinding(0, runtime, tracer)
    payloads = {
        family: FamilyPayload(f"cuda:{family}".encode())
        for family in STATE_FAMILIES
    }
    authenticated = AuthenticatedState._from_verified(
        token_count=1,
        token_hash=token_hash,
        rank_payloads={0: payloads, 1: payloads},
    )

    binding.restore(authenticated, generation_epoch=1)
    if owner.phase is not OwnerPhase.OPEN or tuple(owner.active_state) != STATE_FAMILIES:
        raise RuntimeError("restore did not publish the complete owner table")
    sources = {
        family: DeviceState(
            family=family,
            handle=owner.active_state[family],
            accepted_bytes=len(payloads[family].accepted),
            allocated_bytes=owner.active_state[family].numel(),
        )
        for family in STATE_FAMILIES
    }
    accepted, captured = binding.capture(boundary, sources)
    torch.cuda.synchronize()
    if accepted != authenticated.boundary:
        raise RuntimeError("capture boundary changed")
    if any(captured[family] != payloads[family] for family in STATE_FAMILIES):
        raise RuntimeError("CUDA round trip changed a family payload")
    print(
        f"device={torch.cuda.get_device_name(0)} families={len(captured)} "
        f"bytes={sum(len(payload.accepted) for payload in captured.values())}"
    )


if __name__ == "__main__":
    main()
