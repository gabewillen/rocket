"""Fixed decode scheduling and reusable QSA continuation metadata.

This module is the host contract for Qwen3.8 decode graph selection. The planner
borrows caller stream snapshots and returns an immutable schedule.
``QsaContinuationMetadata`` has one logical writer, retains all backing arrays,
and rewrites their contents in place for one depth bucket at a time. Its buffers
are borrowed read-only views valid until the next ``update``.

``DepthZeroDecodeExecutor`` is the K0 preparation boundary. It rejects loaded
MTP state and every speculative depth. The general planner still describes K0
through K3 so later adaptive executors can reuse the same graph ABI.

Expected validation failures raise :class:`DecodeContractError` before QSA
metadata changes. Callers must externally serialize ``prepare`` and ``update``.
The module does not access a filesystem or network and does not launch a model
kernel. Its required tracer is the explicit observability boundary.

OpenTelemetry cardinality: span attributes are ``phase`` (plan or metadata),
``depth`` (k0 through k3 or mixed), ``graph_batch`` (1, 2, 4, 8, 16, or mixed),
and ``outcome`` (success or failure). Stream and request identifiers are never
attributes.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator, Protocol, Sequence

MAX_STREAMS = 16
MAX_CONTEXT_TOKENS = 262_144
MAX_DEPTH = 3
MAX_QUERY_ROWS = MAX_STREAMS * (MAX_DEPTH + 1)
QSA_COMPRESS_RATIO = 4
QSA_RAW_RING_ROWS = 8
GRAPH_BATCHES = (1, 2, 4, 8, 16)
PAD = -1


class DecodeContractError(ValueError):
    """A caller violated the fixed Qwen3.8 decode contract."""


class Depth(IntEnum):
    """Number of draft tokens verified beside one target token."""

    K0 = 0
    K1 = 1
    K2 = 2
    K3 = 3


@dataclass(frozen=True)
class StreamStep:
    """Borrowed scheduler input at an accepted-token boundary.

    ``slot`` is the engine-owned stream slot. ``accepted_tokens`` excludes all
    unaccepted draft tokens. ``depth`` selects the graph for the next step.
    """

    slot: int
    accepted_tokens: int
    depth: Depth


@dataclass(frozen=True)
class DepthBucket:
    """Immutable stable-order input for one captured graph variant."""

    depth: Depth
    graph_batch: int
    streams: tuple[StreamStep, ...]

    @property
    def query_width(self) -> int:
        return int(self.depth) + 1

    @property
    def actual_batch(self) -> int:
        return len(self.streams)

    @property
    def actual_rows(self) -> int:
        return self.actual_batch * self.query_width

    @property
    def graph_rows(self) -> int:
        return self.graph_batch * self.query_width


@dataclass(frozen=True)
class DecodeSchedule:
    """Owned immutable snapshot of all nonempty depth buckets for one step."""

    buckets: tuple[DepthBucket, ...]

    @property
    def active_streams(self) -> int:
        return sum(bucket.actual_batch for bucket in self.buckets)


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str | int) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


class DepthBucketPlanner:
    """Deterministic bounded planner for K0 through K3 CUDA graph buckets."""

    def __init__(self, enabled_depths: Iterable[Depth], tracer: OtelTracer):
        try:
            raw_depths = tuple(enabled_depths)
            if any(isinstance(value, bool) for value in raw_depths):
                raise ValueError("boolean depth")
            depths = frozenset(Depth(value) for value in raw_depths)
        except (TypeError, ValueError) as exc:
            raise DecodeContractError("enabled depths must be K0 through K3") from exc
        if not depths:
            raise DecodeContractError("at least one decode depth must be enabled")
        if tracer is None:
            raise DecodeContractError("an OpenTelemetry tracer is required")
        self._enabled_depths = depths
        self._tracer = tracer

    @classmethod
    def depth_zero(cls, tracer: OtelTracer) -> "DepthBucketPlanner":
        """Build the K0-only planner used without resident MTP weights."""

        return cls((Depth.K0,), tracer)

    def plan(self, streams: Sequence[StreamStep]) -> DecodeSchedule:
        """Copy validated snapshots into stable depth and stream-slot order.

        The returned schedule owns immutable tuples and has no alias to the
        borrowed input sequence. Validation failure emits one failure span but
        does not mutate planner or QSA state. Runtime work is bounded by 16
        streams and four depth values.
        """

        depth_label = self._depth_label(streams)
        with _observed(self._tracer, "plan", depth_label, "mixed"):
            validated = self._validate(streams)
            buckets = []
            for depth in Depth:
                selected = tuple(step for step in validated if step.depth is depth)
                if selected:
                    buckets.append(
                        DepthBucket(depth, _graph_batch(len(selected)), selected)
                    )
            return DecodeSchedule(tuple(buckets))

    def _validate(self, streams: Sequence[StreamStep]) -> tuple[StreamStep, ...]:
        if not 0 < len(streams) <= MAX_STREAMS:
            raise DecodeContractError("decode step requires 1 through 16 streams")
        owned = []
        seen_slots = set()
        for value in streams:
            if not isinstance(value, StreamStep):
                raise DecodeContractError("decode inputs must be StreamStep values")
            try:
                if isinstance(value.depth, bool):
                    raise ValueError("boolean depth")
                depth = Depth(value.depth)
            except (TypeError, ValueError) as exc:
                raise DecodeContractError("stream depth must be K0 through K3") from exc
            if depth not in self._enabled_depths:
                raise DecodeContractError(f"decode depth k{int(depth)} is not resident")
            if isinstance(value.slot, bool) or not isinstance(value.slot, int):
                raise DecodeContractError("stream slot must be an integer")
            if not 0 <= value.slot < MAX_STREAMS or value.slot in seen_slots:
                raise DecodeContractError(
                    "stream slots must be unique values in 0 through 15"
                )
            if isinstance(value.accepted_tokens, bool) or not isinstance(
                value.accepted_tokens, int
            ):
                raise DecodeContractError("accepted token count must be an integer")
            if not 0 <= value.accepted_tokens <= MAX_CONTEXT_TOKENS - (
                int(depth) + 1
            ):
                raise DecodeContractError(
                    "decode query would exceed the 262144-token context"
                )
            seen_slots.add(value.slot)
            owned.append(StreamStep(value.slot, value.accepted_tokens, depth))
        return tuple(sorted(owned, key=lambda step: (int(step.depth), step.slot)))

    @staticmethod
    def _depth_label(streams: Sequence[StreamStep]) -> str:
        depths = {getattr(step, "depth", None) for step in streams}
        if len(depths) == 1:
            value = next(iter(depths))
            try:
                return f"k{int(Depth(value))}"
            except (TypeError, ValueError):
                pass
        return "mixed"


@dataclass(frozen=True)
class QsaBuffers:
    """Stable read-only views borrowed until the next metadata update."""

    query_start_loc: memoryview
    seq_lens: memoryview
    stream_slots: memoryview
    token_to_req: memoryview
    logical_positions: memoryview
    raw_ring_offsets: memoryview
    compressed_positions: memoryview


@dataclass(frozen=True)
class QsaMetadataLease:
    """Shape and generation for one completed in-place metadata rewrite."""

    generation: int
    depth: Depth
    actual_batch: int
    graph_batch: int
    actual_rows: int
    graph_rows: int


class QsaContinuationMetadata:
    """Single-writer, allocation-stable QSA decode metadata storage.

    All arrays are allocated at construction for c16 K3. ``update`` validates
    the complete bucket before changing them, clears unused graph rows, and
    returns a lease identifying the contents. The stored logical positions
    match vLLM's QSA formula ``seq_len - query_len + within_query``.
    """

    def __init__(self, tracer: OtelTracer):
        if tracer is None:
            raise DecodeContractError("an OpenTelemetry tracer is required")
        self._tracer = tracer
        self._query_start_loc = array("i", [0]) * (MAX_STREAMS + 1)
        self._seq_lens = array("i", [0]) * MAX_STREAMS
        self._stream_slots = array("i", [PAD]) * MAX_STREAMS
        self._token_to_req = array("i", [PAD]) * MAX_QUERY_ROWS
        self._logical_positions = array("q", [PAD]) * MAX_QUERY_ROWS
        self._raw_ring_offsets = array("i", [PAD]) * MAX_QUERY_ROWS
        self._compressed_positions = array("q", [PAD]) * MAX_QUERY_ROWS
        if array("i").itemsize != 4 or array("q").itemsize != 8:
            raise DecodeContractError(
                "host integer widths do not match the QSA metadata ABI"
            )
        self._buffers = QsaBuffers(
            memoryview(self._query_start_loc).toreadonly(),
            memoryview(self._seq_lens).toreadonly(),
            memoryview(self._stream_slots).toreadonly(),
            memoryview(self._token_to_req).toreadonly(),
            memoryview(self._logical_positions).toreadonly(),
            memoryview(self._raw_ring_offsets).toreadonly(),
            memoryview(self._compressed_positions).toreadonly(),
        )
        self._generation = 0

    @property
    def buffers(self) -> QsaBuffers:
        return self._buffers

    @property
    def generation(self) -> int:
        return self._generation

    def update(self, bucket: DepthBucket) -> QsaMetadataLease:
        """Rewrite retained arrays for ``bucket`` and return their new lease.

        ``bucket`` is borrowed and never retained. Validation failures leave
        every array and the generation unchanged. Successful calls invalidate
        all earlier leases and borrowed contents while retaining view identity.
        """

        with _observed(
            self._tracer,
            "metadata",
            _safe_depth_label(getattr(bucket, "depth", None)),
            _safe_graph_batch_label(getattr(bucket, "graph_batch", None)),
        ):
            self._validate_bucket(bucket)
            query_width = bucket.query_width
            actual_rows = bucket.actual_rows
            graph_rows = bucket.graph_rows
            self._clear()
            row = 0
            for request, stream in enumerate(bucket.streams):
                self._query_start_loc[request] = row
                self._seq_lens[request] = stream.accepted_tokens + query_width
                self._stream_slots[request] = stream.slot
                for offset in range(query_width):
                    position = stream.accepted_tokens + offset
                    self._token_to_req[row] = request
                    self._logical_positions[row] = position
                    self._raw_ring_offsets[row] = position % QSA_RAW_RING_ROWS
                    if (position + 1) % QSA_COMPRESS_RATIO == 0:
                        self._compressed_positions[row] = position // QSA_COMPRESS_RATIO
                    row += 1
            for request in range(bucket.actual_batch, bucket.graph_batch + 1):
                self._query_start_loc[request] = actual_rows
            self._generation += 1
            return QsaMetadataLease(
                self._generation,
                bucket.depth,
                bucket.actual_batch,
                bucket.graph_batch,
                actual_rows,
                graph_rows,
            )

    @staticmethod
    def _validate_bucket(bucket: DepthBucket) -> None:
        if not isinstance(bucket, DepthBucket):
            raise DecodeContractError("QSA metadata requires a DepthBucket")
        try:
            if isinstance(bucket.depth, bool):
                raise ValueError("boolean depth")
            depth = Depth(bucket.depth)
        except (TypeError, ValueError) as exc:
            raise DecodeContractError("bucket depth must be K0 through K3") from exc
        if not 0 < len(bucket.streams) <= MAX_STREAMS:
            raise DecodeContractError("QSA bucket requires 1 through 16 streams")
        if (
            isinstance(bucket.graph_batch, bool)
            or bucket.graph_batch not in GRAPH_BATCHES
            or bucket.graph_batch < len(bucket.streams)
        ):
            raise DecodeContractError("QSA bucket graph batch is invalid")
        if bucket.graph_batch != _graph_batch(len(bucket.streams)):
            raise DecodeContractError("QSA bucket must use the smallest captured graph")
        previous_slot = PAD
        for stream in bucket.streams:
            if not isinstance(stream, StreamStep):
                raise DecodeContractError(
                    "QSA bucket streams must be StreamStep values"
                )
            try:
                if isinstance(stream.depth, bool):
                    raise ValueError("boolean depth")
                stream_depth = Depth(stream.depth)
            except (TypeError, ValueError) as exc:
                raise DecodeContractError(
                    "QSA bucket stream depth must be K0 through K3"
                ) from exc
            if stream_depth is not depth:
                raise DecodeContractError("QSA bucket contains a mismatched stream depth")
            if (
                isinstance(stream.slot, bool)
                or not isinstance(stream.slot, int)
                or not 0 <= stream.slot < MAX_STREAMS
                or stream.slot <= previous_slot
            ):
                raise DecodeContractError(
                    "QSA bucket stream slots must be unique and ascending"
                )
            if (
                isinstance(stream.accepted_tokens, bool)
                or not isinstance(stream.accepted_tokens, int)
                or not 0
                <= stream.accepted_tokens
                <= MAX_CONTEXT_TOKENS - (int(depth) + 1)
            ):
                raise DecodeContractError("QSA bucket would exceed the context")
            previous_slot = stream.slot

    def _clear(self) -> None:
        for index in range(MAX_STREAMS + 1):
            self._query_start_loc[index] = 0
        for index in range(MAX_STREAMS):
            self._seq_lens[index] = 0
            self._stream_slots[index] = PAD
        for index in range(MAX_QUERY_ROWS):
            self._token_to_req[index] = PAD
            self._logical_positions[index] = PAD
            self._raw_ring_offsets[index] = PAD
            self._compressed_positions[index] = PAD


@dataclass(frozen=True)
class PreparedDecode:
    """Borrowed metadata plus immutable graph selection for one K0 launch."""

    schedule: DecodeSchedule
    bucket: DepthBucket
    lease: QsaMetadataLease
    buffers: QsaBuffers


class DepthZeroDecodeExecutor:
    """K0 preparation boundary with no resident or executable MTP path."""

    def __init__(self, tracer: OtelTracer, *, mtp_resident: bool = False):
        if mtp_resident:
            raise DecodeContractError("K0 executor forbids resident MTP weights")
        self._planner = DepthBucketPlanner.depth_zero(tracer)
        self._metadata = QsaContinuationMetadata(tracer)

    @property
    def metadata(self) -> QsaContinuationMetadata:
        return self._metadata

    def prepare(self, streams: Sequence[StreamStep]) -> PreparedDecode:
        """Prepare one externally serialized K0 launch without model execution."""

        schedule = self._planner.plan(streams)
        if len(schedule.buckets) != 1 or schedule.buckets[0].depth is not Depth.K0:
            raise DecodeContractError("K0 executor requires exactly one K0 bucket")
        bucket = schedule.buckets[0]
        lease = self._metadata.update(bucket)
        return PreparedDecode(schedule, bucket, lease, self._metadata.buffers)


def _graph_batch(actual: int) -> int:
    for candidate in GRAPH_BATCHES:
        if actual <= candidate:
            return candidate
    raise DecodeContractError("no captured graph supports this batch")


def _safe_depth_label(value: object) -> str:
    try:
        return f"k{int(Depth(value))}"
    except (TypeError, ValueError):
        return "mixed"


def _safe_graph_batch_label(value: object) -> int | str:
    return value if value in GRAPH_BATCHES else "mixed"


@contextmanager
def _observed(
    tracer: OtelTracer, phase: str, depth: str, graph_batch: int | str
) -> Iterator[None]:
    with tracer.start_as_current_span("rocket.qwen38.decode") as span:
        span.set_attribute("phase", phase)
        span.set_attribute("depth", depth)
        span.set_attribute("graph_batch", graph_batch)
        try:
            yield
        except BaseException as exc:
            span.set_attribute("outcome", "failure")
            span.record_exception(exc)
            raise
        else:
            span.set_attribute("outcome", "success")
