"""Fail-closed production composition contract for one TP2 K0 token.

This module is the composition boundary, not a kernel substitute. A binding is
published only when both authenticated target slabs and every layer-local
target participant are present. MTP participants are outside this K0 domain.

OpenTelemetry cardinality is bounded to phase (validate), outcome (success or
failure), and missing bucket (0, 1, 2-16, 17-64, or 65-394). Participant names,
prompts, tokens, paths, hashes, pointers, and exception text are excluded from
attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from .whole_decoder import AttentionKind, DecoderSlabs, LAYER_TOPOLOGY

K0_DOMAIN = "target_k0"


class K0CompositionError(RuntimeError):
    """A complete TP2 K0 production binding could not be published."""

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing = missing


class K0Participant(Protocol):
    """One production adapter bound to the target-only K0 execution domain."""

    @property
    def execution_domain(self) -> str: ...


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


def _rank_participant_names(rank: int) -> tuple[str, ...]:
    prefix = f"rank{rank}"
    result = [
        f"{prefix}.target_slab",
        f"{prefix}.graph_factory",
        f"{prefix}.embedder",
    ]
    for layer, kind in LAYER_TOPOLOGY.items():
        result.append(f"{prefix}.{kind.value}.layer{layer}")
    result.extend(f"{prefix}.moe.layer{layer}" for layer in range(48))
    for layer in range(48):
        result.append(f"{prefix}.pair_reduce.attention.layer{layer}")
        result.append(f"{prefix}.pair_reduce.moe.layer{layer}")
    result.append(f"{prefix}.output_sampler")
    return tuple(result)


REQUIRED_K0_PARTICIPANTS = (
    "tokenizer",
    "saved_vllm_comparator",
    *_rank_participant_names(0),
    *_rank_participant_names(1),
)


def _missing_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 16:
        return "2-16"
    if count <= 64:
        return "17-64"
    return "65-394"


@dataclass(frozen=True)
class K0CompositionBinding:
    """Immutable, complete participant table for a later TP2 launch root."""

    participants: Mapping[str, object]


class K0CompositionRoot:
    """Validate and atomically publish the first-token dependency set.

    This first vertical-slice gate does not launch CUDA. Concrete adapters are
    added behind these fixed names in blocker order. Until all names bind the
    target K0 domain, ``binding`` remains absent.
    """

    def __init__(self, participants: Mapping[str, object], tracer: OtelTracer):
        if tracer is None or not callable(
            getattr(tracer, "start_as_current_span", None)
        ):
            raise K0CompositionError("K0 composition requires OpenTelemetry")
        self._binding: K0CompositionBinding | None = None
        with tracer.start_as_current_span(
            "rocket.qwen38.k0_composition.validate"
        ) as span:
            span.set_attribute("phase", "validate")
            try:
                owned = self._validate(participants)
            except K0CompositionError as exc:
                span.set_attribute("outcome", "failure")
                span.set_attribute("missing.bucket", _missing_bucket(len(exc.missing)))
                span.record_exception(exc)
                raise
            span.set_attribute("outcome", "success")
            span.set_attribute("missing.bucket", "0")
        self._binding = K0CompositionBinding(MappingProxyType(owned))

    @property
    def binding(self) -> K0CompositionBinding:
        if self._binding is None:
            raise K0CompositionError("K0 composition was not published")
        return self._binding

    @staticmethod
    def _validate(participants: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(participants, Mapping):
            raise K0CompositionError(
                "K0 participants must be an ordered-name mapping",
                missing=REQUIRED_K0_PARTICIPANTS,
            )
        unknown = tuple(sorted(set(participants) - set(REQUIRED_K0_PARTICIPANTS)))
        if unknown:
            raise K0CompositionError(f"unknown K0 participant: {unknown[0]}")
        missing = tuple(
            name
            for name in REQUIRED_K0_PARTICIPANTS
            if participants.get(name) is None
        )
        if missing:
            raise K0CompositionError(
                f"missing K0 participant: {missing[0]}", missing=missing
            )

        owned = {name: participants[name] for name in REQUIRED_K0_PARTICIPANTS}
        slabs = []
        for rank in (0, 1):
            name = f"rank{rank}.target_slab"
            slab = owned[name]
            if not isinstance(slab, DecoderSlabs) or slab.rank != rank:
                raise K0CompositionError(f"K0 participant identity changed: {name}")
            slabs.append(slab)
        if slabs[0].target is slabs[1].target:
            raise K0CompositionError("rank target slab ownership is aliased")

        for name, participant in owned.items():
            if name.endswith("target_slab"):
                continue
            if getattr(participant, "execution_domain", None) != K0_DOMAIN:
                raise K0CompositionError(
                    f"K0 participant domain changed: {name}"
                )
            if ".gdn.layer" in name and getattr(
                participant, "kind", None
            ) is not AttentionKind.GDN:
                raise K0CompositionError(f"GDN participant kind changed: {name}")
            if ".qsa.layer" in name and getattr(
                participant, "kind", None
            ) is not AttentionKind.QSA:
                raise K0CompositionError(f"QSA participant kind changed: {name}")
            required_method = _required_method(name)
            if not callable(getattr(participant, required_method, None)):
                raise K0CompositionError(
                    f"K0 participant interface changed: {name}.{required_method}"
                )
        return owned


def _required_method(name: str) -> str:
    if name == "tokenizer":
        return "encode_one"
    if name == "saved_vllm_comparator":
        return "compare"
    if name.endswith("graph_factory"):
        return "bind"
    if name.endswith("embedder"):
        return "embed"
    if ".gdn.layer" in name or ".qsa.layer" in name or ".moe.layer" in name:
        return "execute"
    if ".pair_reduce." in name:
        return "reduce"
    if name.endswith("output_sampler"):
        return "sample"
    raise K0CompositionError(f"unknown K0 participant interface: {name}")


__all__ = [
    "K0CompositionBinding",
    "K0CompositionError",
    "K0CompositionRoot",
    "K0_DOMAIN",
    "REQUIRED_K0_PARTICIPANTS",
]
