# SPDX-License-Identifier: Apache-2.0
"""Validation-only eager target MoE adapter.

This adapter runs the exact measured FlashInfer B12x target primitive with
authenticated target-slab weights. It intentionally retains eager Torch
allocation and wrapper-owned output. It is excluded from K0 composition and
cannot satisfy graph capture, fixed-buffer, latency, or throughput claims.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from .routed_moe import (
    FLASHINFER_COMMIT,
    HIDDEN,
    FlashInferRoutedMoeBackend,
    MoeShape,
    OwnerLocalMoeSlab,
    RoutedMoeGraphError,
    materialize_flashinfer_weights,
    selector_for,
)

EAGER_VALIDATION_IDENTITY = f"flashinfer-b12x:{FLASHINFER_COMMIT}:target-nvfp4"


@dataclass(frozen=True)
class EagerValidationParity:
    """Bounded numeric evidence from one two-rank correctness comparison."""

    rows: int
    max_abs: float
    max_rel: float
    atol: float
    rtol: float
    accepted: bool


def compare_pair_reduced_partial(
    rank0_partial,
    rank1_partial,
    reference,
    *,
    atol: float,
    rtol: float,
    torch_api=None,
) -> EagerValidationParity:
    """Compare the two eager rank partials with a saved vLLM BF16 oracle."""

    if torch_api is None:
        import torch as torch_api  # type: ignore[no-redef]
    torch = torch_api
    shape = tuple(getattr(reference, "shape", ()))
    if (
        len(shape) != 2
        or shape[1] != HIDDEN
        or tuple(getattr(rank0_partial, "shape", ())) != shape
        or tuple(getattr(rank1_partial, "shape", ())) != shape
        or atol < 0.0
        or rtol < 0.0
    ):
        raise RoutedMoeGraphError("eager target MoE parity inputs changed")
    observed = rank0_partial.float() + rank1_partial.float()
    expected = reference.float()
    absolute = (observed - expected).abs()
    denominator = expected.abs().clamp_min(1.0e-12)
    return EagerValidationParity(
        rows=shape[0],
        max_abs=float(absolute.max().item()),
        max_rel=float((absolute / denominator).max().item()),
        atol=float(atol),
        rtol=float(rtol),
        accepted=bool(torch.allclose(observed, expected, atol=atol, rtol=rtol)),
    )


class EagerValidationTargetMoeAdapter:
    """Real eager correctness path that is never a production fallback.

    Span dimensions are bounded to rank(2), layer(48), sequence.bucket(5),
    phase(1), and outcome(2). Routes, weights, pointers, hashes, paths, and
    error text are excluded from attributes.
    """

    validation_only = True
    production_eligible = False
    graph_safe = False
    _SPAN = "rocket.qwen38.validation.target_moe_eager"

    def __init__(self, slab, backend, tracer):
        if (
            not isinstance(slab, OwnerLocalMoeSlab)
            or getattr(backend, "implementation_identity", None)
            != EAGER_VALIDATION_IDENTITY
            or tracer is None
        ):
            raise RoutedMoeGraphError("eager target MoE validation identity changed")
        self._slab = slab
        self._backend = backend
        self._tracer = tracer

    @classmethod
    def from_authenticated_slab(
        cls, slab: OwnerLocalMoeSlab, tracer, *, torch_api=None, device="cuda",
    ) -> "EagerValidationTargetMoeAdapter":
        """Materialize the exact target slab into the measured B12x adapter."""

        weights = materialize_flashinfer_weights(
            slab, torch_api=torch_api, device=device
        )
        backend = FlashInferRoutedMoeBackend(weights, torch_api=torch_api)
        return cls(slab, backend, tracer)

    @property
    def identity(self):
        return (
            self._slab.revision,
            self._slab.artifact_key,
            self._slab.rank,
            self._slab.layer,
            self._slab.layout_sha256,
            EAGER_VALIDATION_IDENTITY,
        )

    def execute(self, hidden, global_ids, routing_weights, shape: MoeShape):
        if not isinstance(shape, MoeShape):
            raise RoutedMoeGraphError("eager target MoE validation shape changed")
        rows = shape.token_rows
        with self._observed(shape) as span:
            try:
                output = self._backend.launch(
                    hidden,
                    global_ids,
                    routing_weights,
                    rank=self._slab.rank,
                    shape=shape,
                    selector=selector_for(shape),
                )
                if (
                    tuple(getattr(output, "shape", ())) != (rows, HIDDEN)
                    or "bfloat16" not in str(getattr(output, "dtype", ""))
                    or str(getattr(output, "device", ""))
                    != str(getattr(hidden, "device", ""))
                ):
                    raise RoutedMoeGraphError(
                        "eager target MoE returned a foreign partial"
                    )
                span.set_attribute("outcome", "success")
                return output
            except BaseException as exc:
                span.set_attribute("outcome", "failure")
                span.record_exception(exc)
                raise

    @contextmanager
    def _observed(self, shape: MoeShape):
        with self._tracer.start_as_current_span(self._SPAN) as span:
            span.set_attribute("phase", "execute")
            span.set_attribute("rank", self._slab.rank)
            span.set_attribute("layer", self._slab.layer)
            span.set_attribute("sequence.bucket", shape.sequences)
            yield span


__all__ = [
    "EAGER_VALIDATION_IDENTITY",
    "EagerValidationParity",
    "EagerValidationTargetMoeAdapter",
    "compare_pair_reduced_partial",
]
