#!/usr/bin/env python3
"""Falsifiable ROSA episodic-memory adapter experiment for GLM-5.3 mHC.

This module does not train GLM. It trains and exports only the additive memory
projection and gates from hidden-state capture files. The adapter is exactly
zero at initialization and when disabled.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Retrieval:
    tokens: tuple[int, ...]
    source_end: int
    recency: int
    confidence: float
    matches: int


class ExactSuffixMemory:
    """Bounded online exact suffix index. Updates are O(max_ngram), a constant."""

    def __init__(self, max_ngram: int = 8, max_history: int = 262_144):
        self.max_ngram = max_ngram
        self.max_history = max_history
        self.tokens: list[int] = []
        self.index: list[dict[tuple[int, ...], list[int]]] = [dict() for _ in range(max_ngram + 1)]

    def append(self, token: int) -> None:
        self.tokens.append(int(token))
        end = len(self.tokens)
        for n in range(1, min(self.max_ngram, end) + 1):
            key = tuple(self.tokens[end - n:end])
            self.index[n].setdefault(key, []).append(end)
        if len(self.tokens) > self.max_history:
            # Rebuild at a bounded checkpoint. This avoids stale positions and
            # makes the maximum CPU memory part of the experiment contract.
            keep = self.tokens[-self.max_history:]
            self.tokens = []
            self.index = [dict() for _ in range(self.max_ngram + 1)]
            for value in keep:
                self.append(value)

    def extend(self, tokens: list[int] | tuple[int, ...]) -> None:
        for token in tokens:
            self.append(token)

    def retrieve(self, width: int, min_ngram: int = 3) -> Retrieval | None:
        end = len(self.tokens)
        for n in range(min(self.max_ngram, end), min_ngram - 1, -1):
            key = tuple(self.tokens[end - n:end])
            candidates = [p for p in self.index[n].get(key, []) if p + width <= end - n]
            if not candidates:
                continue
            # Preserve the measured first-match policy. Ambiguity lowers the
            # confidence supplied to the learned gate rather than changing the
            # continuation silently.
            source_end = candidates[0]
            continuation = tuple(self.tokens[source_end:source_end + width])
            same = sum(tuple(self.tokens[p:p + width]) == continuation for p in candidates)
            confidence = (n / self.max_ngram) * (same / len(candidates))
            return Retrieval(continuation, source_end, end - source_end, confidence, len(candidates))
        return None


class RosaMemoryLane(nn.Module):
    """Low-rank projection plus signed zero-initialized gates for mHC stream 4."""

    def __init__(self, hidden: int, layers: int = 45, rank: int = 64, stream: int = 3):
        super().__init__()
        self.hidden, self.layers, self.rank, self.stream = hidden, layers, rank, stream
        self.down = nn.Linear(hidden + 3, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        self.layer_gate = nn.Parameter(torch.zeros(layers))
        self.token_gate = nn.Linear(hidden + 3, 1, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.token_gate.weight)

    def projected(self, retrieval_hidden: Tensor, features: Tensor) -> Tensor:
        x = torch.cat((retrieval_hidden, features), dim=-1)
        # Identity makes the untrained injection experiment meaningful and
        # gives a zero-initialized gate a nonzero gradient. The learned path
        # is an additive low-rank correction, so export remains small.
        return retrieval_hidden + self.up(torch.nn.functional.silu(self.down(x)))

    def forward(self, streams: Tensor, retrieval_hidden: Tensor, features: Tensor,
                layer: int, enabled: bool = True, fixed_gate: float | None = None) -> tuple[Tensor, Tensor]:
        if not enabled:
            return streams, torch.zeros(streams.shape[0], device=streams.device, dtype=streams.dtype)
        x = torch.cat((retrieval_hidden, features), dim=-1)
        raw = self.layer_gate[layer] + self.token_gate(x).squeeze(-1)
        # confidence is feature 0. tanh(0) is exactly zero, so the initialized
        # trained adapter reproduces the input stream tensor bit for bit.
        gate = features[:, 0] * (torch.full_like(raw, fixed_gate) if fixed_gate is not None else torch.tanh(raw))
        delta = gate[:, None] * self.projected(retrieval_hidden, features)
        out = streams.clone()
        out[:, self.stream, :] = out[:, self.stream, :] + delta.to(out.dtype)
        return out, gate

    def export_npz(self, path: pathlib.Path) -> None:
        arrays = {name: value.detach().cpu().numpy() for name, value in self.state_dict().items()}
        arrays["format_version"] = np.array([1], dtype=np.int32)
        arrays["hidden"] = np.array([self.hidden], dtype=np.int32)
        arrays["layers"] = np.array([self.layers], dtype=np.int32)
        arrays["rank"] = np.array([self.rank], dtype=np.int32)
        arrays["stream"] = np.array([self.stream], dtype=np.int32)
        np.savez(path, **arrays)


def load_capture(path: pathlib.Path) -> dict[str, Tensor]:
    with np.load(path) as data:
        required = {"streams", "retrieval_hidden", "features", "target_delta", "layer", "trust"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"capture missing arrays: {sorted(missing)}")
        return {key: torch.from_numpy(data[key]) for key in required}


def train(capture: pathlib.Path, output: pathlib.Path, rank: int, epochs: int, lr: float) -> dict:
    data = load_capture(capture)
    hidden = int(data["streams"].shape[-1])
    model = RosaMemoryLane(hidden, rank=rank)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        total = torch.zeros((), dtype=torch.float32)
        for layer in torch.unique(data["layer"]).tolist():
            mask = data["layer"] == layer
            out, gate = model(data["streams"][mask].float(), data["retrieval_hidden"][mask].float(),
                              data["features"][mask].float(), int(layer))
            delta = out[:, 3] - data["streams"][mask, 3].float()
            useful = data["trust"][mask].float()
            target = data["target_delta"][mask].float()
            fit = ((delta - target) ** 2).mean(dim=-1)
            suppress = delta.square().mean(dim=-1)
            total = total + (useful * fit + (1.0 - useful) * suppress).mean()
            total = total + 0.01 * ((gate - useful) ** 2).mean()
        total.backward()
        opt.step()
        losses.append(float(total.detach()))
    model.export_npz(output)
    report = {"capture": str(capture), "output": str(output), "epochs": epochs,
              "loss_first": losses[0], "loss_last": losses[-1], "rank": rank}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def self_test() -> None:
    memory = ExactSuffixMemory(max_ngram=4)
    memory.extend([1, 2, 3, 9, 1, 2, 3])
    got = memory.retrieve(1, min_ngram=3)
    assert got and got.tokens == (9,) and got.source_end == 3
    model = RosaMemoryLane(hidden=16, layers=45, rank=4)
    streams = torch.randn(3, 4, 16)
    retrieval = torch.randn(3, 16)
    features = torch.tensor([[1.0, 0.1, 0.0], [0.5, 0.4, 1.0], [0.0, 1.0, 0.2]])
    out, gate = model(streams, retrieval, features, layer=7)
    assert torch.equal(out, streams)
    assert torch.count_nonzero(gate) == 0
    off, _ = model(streams, retrieval, features, layer=7, enabled=False)
    assert torch.equal(off, streams)
    injected, fixed = model(streams, retrieval, features, layer=7, fixed_gate=0.25)
    assert not torch.equal(injected, streams)
    assert float(fixed[2]) == 0.0
    # A future token is unavailable until append, which is the causal contract.
    before = ExactSuffixMemory(max_ngram=3)
    before.extend([4, 5, 6, 4, 5, 6])
    assert before.retrieve(1) is None
    before.append(7)
    assert before.retrieve(1) is None
    print("ROSA memory-lane self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    p = sub.add_parser("train")
    p.add_argument("--capture", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        print(json.dumps(train(args.capture, args.output, args.rank, args.epochs, args.lr), indent=2))


if __name__ == "__main__":
    main()
