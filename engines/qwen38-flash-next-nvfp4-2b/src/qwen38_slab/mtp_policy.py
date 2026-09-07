"""Deterministic adaptive lazy-MTP policy for the specialized Qwen3.8 engine.

The policy reuses the measured general-engine control: sixteen initial K1/K2/K3
probes, a 3% Wilson-bound promotion margin, two failing eight-round windows
before demotion, exploration every 64 eligible rounds, and a 2,698,026,496-byte
cold MTP charge amortized over 64 rounds.

Depth support extends to K7. Maximum depth is configured for exact concurrency
values only. Unconfigured decode concurrency fails closed. Current records
establish K7 availability at c1 and a K1 cap at c16, without establishing a c1
optimum or any cap for c2 through c15.

Inputs and outputs are immutable. The policy retains no mutable state and
performs no I/O or residency mutation. Ordered prefix residency commands request
runtime work without claiming it completed. Expected validation failures raise
``MtpPolicyError`` without a successor state. Canonical state JSON is bounded to
64 KiB so it can join an atomic prefix record.

OpenTelemetry cardinality is finite: operation has three values, session phase
five plus none, concurrency c1 through c16 plus none, depths K0 through K7 plus
none, reason seven plus none, and outcome two. Request and session identifiers
are excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from .decode import Depth as MtpDepth

SCHEMA = "rocket.qwen38-adaptive-mtp-policy.v1"
MAX_STATE_BYTES = 65_536
MAX_COUNTER = 65_535
MAX_ACCEPTED_TOKENS = 1_000_000_000
MAX_COST_NS = 3_600_000_000_000
MAX_MODELED_BYTES = 1 << 50
MAX_RATE_MILLI = 10_000_000
PPM = 1_000_000
AGGREGATE_FLOOR_MILLI = 320_000
PER_STREAM_FLOOR_MILLI = 20_000
PROMOTION_MARGIN_PPM = 30_000
DEMOTION_WINDOW_ROUNDS = 8
DEMOTION_FAILING_WINDOWS = 2
EXPLORATION_INTERVAL = 64
LAZY_WEIGHT_BYTES = 2_698_026_496
LAZY_RESIDENCY_ROUNDS = 64
PROBE_DEPTHS = (1, 1, 2, 1, 2, 3, 2, 3) * 2


class MtpPolicyError(ValueError):
    """Invalid policy input or state; no residency command was committed."""


class SessionPhase(str, Enum):
    """Explicit caller-owned session lifecycle."""

    PREFILL = "prefill"
    RESTORING = "restoring"
    EARLY_DECODE = "early_decode"
    STEADY_DECODE = "steady_decode"
    DRAINING = "draining"


class DecisionReason(str, Enum):
    """Finite decision reason."""

    PHASE_K0 = "phase_k0"
    REMAINING_CAP = "remaining_cap"
    PROBE = "probe"
    POLICY = "policy"
    EXPLORE = "explore"
    PROMOTE = "promote"
    DEMOTE = "demote"


class ResidencyAction(str, Enum):
    """Ordered depth-residency request."""

    LOAD = "load"
    EVICT = "evict"


@dataclass(frozen=True)
class ConcurrencyCeiling:
    """Evidence-scoped maximum depth for one exact concurrency."""

    concurrency: int
    max_depth: MtpDepth

    def __post_init__(self) -> None:
        _require_int(self.concurrency, 1, 16, "ceiling concurrency")
        _require_depth(self.max_depth, "ceiling max_depth")


@dataclass(frozen=True)
class PolicyConfig:
    """Immutable evidence inputs and preserved policy thresholds."""

    ceilings: tuple[ConcurrencyCeiling, ...]
    promotion_margin_ppm: int = PROMOTION_MARGIN_PPM
    demotion_window_rounds: int = DEMOTION_WINDOW_ROUNDS
    demotion_failing_windows: int = DEMOTION_FAILING_WINDOWS
    exploration_interval: int = EXPLORATION_INTERVAL
    lazy_weight_bytes: int = LAZY_WEIGHT_BYTES
    lazy_residency_rounds: int = LAZY_RESIDENCY_ROUNDS

    def __post_init__(self) -> None:
        if not isinstance(self.ceilings, tuple) or not 0 < len(self.ceilings) <= 16:
            raise MtpPolicyError("ceilings must contain 1 through 16 exact entries")
        if any(not isinstance(value, ConcurrencyCeiling) for value in self.ceilings):
            raise MtpPolicyError("ceilings must contain ConcurrencyCeiling values")
        keys = tuple(value.concurrency for value in self.ceilings)
        if keys != tuple(sorted(set(keys))):
            raise MtpPolicyError("ceilings must have unique ascending concurrency")
        fixed = (
            (self.promotion_margin_ppm, PROMOTION_MARGIN_PPM, "promotion margin"),
            (self.demotion_window_rounds, DEMOTION_WINDOW_ROUNDS, "demotion window"),
            (self.demotion_failing_windows, DEMOTION_FAILING_WINDOWS, "demotion failures"),
            (self.exploration_interval, EXPLORATION_INTERVAL, "exploration interval"),
            (self.lazy_weight_bytes, LAZY_WEIGHT_BYTES, "lazy weight bytes"),
            (self.lazy_residency_rounds, LAZY_RESIDENCY_ROUNDS, "lazy residency rounds"),
        )
        for observed, expected, name in fixed:
            if observed != expected:
                raise MtpPolicyError(f"{name} must retain the measured value {expected}")

    def ceiling_for(self, concurrency: int) -> MtpDepth | None:
        """Return an exact ceiling, or ``None`` without extrapolating."""

        return next(
            (item.max_depth for item in self.ceilings if item.concurrency == concurrency),
            None,
        )

    @property
    def fingerprint(self) -> str:
        """Return the configuration identity stored with policy state."""

        record = {
            "ceilings": [[item.concurrency, int(item.max_depth)] for item in self.ceilings],
            "promotion_margin_ppm": self.promotion_margin_ppm,
            "demotion_window_rounds": self.demotion_window_rounds,
            "demotion_failing_windows": self.demotion_failing_windows,
            "exploration_interval": self.exploration_interval,
            "lazy_weight_bytes": self.lazy_weight_bytes,
            "lazy_residency_rounds": self.lazy_residency_rounds,
        }
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class PolicyObservation:
    """One completed measurement window.

    Position tuples contain exactly ``depth`` counters. Rates use
    milli-tokens/second. ``modeled_step_bytes`` covers one decode step;
    ``cost_ns`` and ``accepted_tokens`` cover the observation window.
    """

    concurrency: int
    depth: MtpDepth
    position_successes: tuple[int, ...]
    position_trials: tuple[int, ...]
    accepted_tokens: int
    cost_ns: int
    modeled_step_bytes: int
    aggregate_tok_s_milli: int
    per_stream_tok_s_milli: int
    whole_decode_roofline_tok_s_milli: int

    def __post_init__(self) -> None:
        _require_int(self.concurrency, 1, 16, "observation concurrency")
        _require_depth(self.depth, "observation depth")
        if (
            not isinstance(self.position_successes, tuple)
            or not isinstance(self.position_trials, tuple)
            or len(self.position_successes) != int(self.depth)
            or len(self.position_trials) != int(self.depth)
        ):
            raise MtpPolicyError("position evidence must contain one counter per draft position")
        for success, trials in zip(self.position_successes, self.position_trials):
            _require_int(trials, 1, MAX_ACCEPTED_TOKENS, "position trials")
            _require_int(success, 0, trials, "position successes")
        _require_int(self.accepted_tokens, 1, MAX_ACCEPTED_TOKENS, "accepted_tokens")
        _require_int(self.cost_ns, 1, MAX_COST_NS, "cost_ns")
        _require_int(self.modeled_step_bytes, 1, MAX_MODELED_BYTES, "modeled_step_bytes")
        _require_int(self.aggregate_tok_s_milli, 1, MAX_RATE_MILLI, "aggregate rate")
        _require_int(self.per_stream_tok_s_milli, 1, self.aggregate_tok_s_milli, "per-stream rate")
        _require_int(
            self.whole_decode_roofline_tok_s_milli,
            self.aggregate_tok_s_milli,
            MAX_RATE_MILLI,
            "whole-decode roofline rate",
        )

    @property
    def clears_acceptance_floors(self) -> bool:
        """Return whether both strict c16 acceptance floors are exceeded."""

        return (
            self.concurrency == 16
            and self.aggregate_tok_s_milli > AGGREGATE_FLOOR_MILLI
            and self.per_stream_tok_s_milli > PER_STREAM_FLOOR_MILLI
        )

    @property
    def roofline_efficiency_ppm(self) -> int:
        """Return floor-rounded aggregate throughput / whole-decode roofline."""

        return self.aggregate_tok_s_milli * PPM // self.whole_decode_roofline_tok_s_milli

    @property
    def cost_ns_per_accepted_token(self) -> int:
        """Return floor-rounded nanoseconds per accepted token."""

        return self.cost_ns // self.accepted_tokens


@dataclass(frozen=True)
class DepthEstimate:
    """Latest depth observation plus a saturating sample count."""

    observation: PolicyObservation
    samples: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, PolicyObservation):
            raise MtpPolicyError("estimate observation must be PolicyObservation")
        _require_int(self.samples, 1, MAX_COUNTER, "estimate samples")


@dataclass(frozen=True)
class CohortState:
    """Immutable state owned by one exact-concurrency cohort."""

    concurrency: int
    active_depth: MtpDepth
    decode_rounds: int
    eligible_rounds: int
    consecutive_failing_windows: int
    window_results: tuple[bool, ...]
    position_successes: tuple[int, ...]
    position_trials: tuple[int, ...]
    estimates: tuple[DepthEstimate, ...]

    def __post_init__(self) -> None:
        _require_int(self.concurrency, 1, 16, "cohort concurrency")
        _require_depth(self.active_depth, "cohort active depth")
        _require_int(self.decode_rounds, 0, MAX_COUNTER, "decode rounds")
        _require_int(self.eligible_rounds, 0, MAX_COUNTER, "eligible rounds")
        _require_int(
            self.consecutive_failing_windows,
            0,
            DEMOTION_FAILING_WINDOWS - 1,
            "failing windows",
        )
        if (
            not isinstance(self.window_results, tuple)
            or len(self.window_results) >= DEMOTION_WINDOW_ROUNDS
            or any(type(value) is not bool for value in self.window_results)
        ):
            raise MtpPolicyError("window results must be fewer than eight booleans")
        for name, values in (
            ("position successes", self.position_successes),
            ("position trials", self.position_trials),
        ):
            if not isinstance(values, tuple) or len(values) != 7:
                raise MtpPolicyError(f"{name} must contain seven counters")
            for value in values:
                _require_int(value, 0, MAX_ACCEPTED_TOKENS, name)
        if any(a > b for a, b in zip(self.position_successes, self.position_trials)):
            raise MtpPolicyError("position successes cannot exceed trials")
        if (
            not isinstance(self.estimates, tuple)
            or len(self.estimates) > 8
            or any(not isinstance(item, DepthEstimate) for item in self.estimates)
        ):
            raise MtpPolicyError("cohort estimates must contain at most eight values")
        if any(item.observation.concurrency != self.concurrency for item in self.estimates):
            raise MtpPolicyError("cohort estimates must match cohort concurrency")
        depths = tuple(item.observation.depth for item in self.estimates)
        if depths != tuple(sorted(set(depths), key=int)):
            raise MtpPolicyError("cohort estimates must have unique ascending depths")

    @classmethod
    def initial(cls, concurrency: int) -> "CohortState":
        return cls(concurrency, MtpDepth.K0, 0, 0, 0, (), (0,) * 7, (0,) * 7, ())


@dataclass(frozen=True)
class PolicyState:
    """Bounded immutable state for atomic-prefix serialization."""

    config_fingerprint: str
    phase: SessionPhase
    decisions: int
    cohorts: tuple[CohortState, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config_fingerprint, str)
            or len(self.config_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.config_fingerprint)
        ):
            raise MtpPolicyError("config fingerprint must be lowercase SHA-256")
        if not isinstance(self.phase, SessionPhase):
            raise MtpPolicyError("policy phase must be SessionPhase")
        _require_int(self.decisions, 0, MAX_COUNTER, "policy decisions")
        if (
            not isinstance(self.cohorts, tuple)
            or not 0 < len(self.cohorts) <= 16
            or any(not isinstance(item, CohortState) for item in self.cohorts)
        ):
            raise MtpPolicyError("policy state must contain 1 through 16 cohorts")
        keys = tuple(item.concurrency for item in self.cohorts)
        if keys != tuple(sorted(set(keys))):
            raise MtpPolicyError("policy cohorts must have unique ascending concurrency")


@dataclass(frozen=True)
class PolicyEvent:
    """Typed scheduler event processed to quiescence."""

    phase: SessionPhase
    concurrency: int
    remaining_tokens: int
    resident_depth: MtpDepth
    observation: PolicyObservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, SessionPhase):
            raise MtpPolicyError("event phase must be SessionPhase")
        _require_int(self.concurrency, 1, 16, "event concurrency")
        _require_int(self.remaining_tokens, 1, MAX_ACCEPTED_TOKENS, "remaining_tokens")
        _require_depth(self.resident_depth, "resident depth")
        if self.observation is not None and not isinstance(self.observation, PolicyObservation):
            raise MtpPolicyError("event observation must be PolicyObservation or None")


@dataclass(frozen=True)
class ResidencyCommand:
    """One ordered load or eviction for a depth-residency prefix."""

    action: ResidencyAction
    depth: MtpDepth

    def __post_init__(self) -> None:
        if not isinstance(self.action, ResidencyAction):
            raise MtpPolicyError("residency action must be load or evict")
        _require_depth(self.depth, "residency depth")
        if self.depth is MtpDepth.K0:
            raise MtpPolicyError("K0 has no MTP residency")


@dataclass(frozen=True)
class PolicyDecision:
    """Owned output whose commands precede execution of the selected depth."""

    selected_depth: MtpDepth
    reason: DecisionReason
    residency_commands: tuple[ResidencyCommand, ...]
    next_state: PolicyState


class _Span(Protocol):
    def __enter__(self) -> "_Span": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
    def set_attribute(self, key: str, value: str) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...


class OtelTracer(Protocol):
    def start_as_current_span(self, name: str) -> _Span: ...


_PHASE_TRANSITIONS = {
    SessionPhase.PREFILL: frozenset((SessionPhase.PREFILL, SessionPhase.RESTORING, SessionPhase.EARLY_DECODE, SessionPhase.DRAINING)),
    SessionPhase.RESTORING: frozenset((SessionPhase.RESTORING, SessionPhase.EARLY_DECODE, SessionPhase.DRAINING)),
    SessionPhase.EARLY_DECODE: frozenset((SessionPhase.EARLY_DECODE, SessionPhase.STEADY_DECODE, SessionPhase.DRAINING)),
    SessionPhase.STEADY_DECODE: frozenset((SessionPhase.STEADY_DECODE, SessionPhase.DRAINING)),
    SessionPhase.DRAINING: frozenset((SessionPhase.DRAINING,)),
}


class AdaptiveMtpPolicy:
    """Stateless policy owner with bounded OpenTelemetry side effects."""

    def __init__(self, tracer: OtelTracer, config: PolicyConfig):
        if tracer is None or not callable(getattr(tracer, "start_as_current_span", None)):
            raise MtpPolicyError("an OpenTelemetry tracer is required")
        if not isinstance(config, PolicyConfig):
            raise MtpPolicyError("config must be PolicyConfig")
        self._tracer = tracer
        self._config = config

    def initial_state(self) -> PolicyState:
        """Return PREFILL state for every configured concurrency."""

        return PolicyState(
            self._config.fingerprint,
            SessionPhase.PREFILL,
            0,
            tuple(CohortState.initial(item.concurrency) for item in self._config.ceilings),
        )

    def decide(self, state: PolicyState, event: PolicyEvent) -> PolicyDecision:
        """Process one event atomically and return commands plus successor state."""

        phase = event.phase.value if isinstance(event, PolicyEvent) else "none"
        concurrency = f"c{event.concurrency}" if isinstance(event, PolicyEvent) else "none"
        active = "none"
        if isinstance(state, PolicyState) and isinstance(event, PolicyEvent):
            cohort = _cohort_for(state.cohorts, event.concurrency)
            active = _depth_label(cohort.active_depth) if cohort is not None else "none"
        with self._tracer.start_as_current_span("rocket.qwen38.mtp.policy") as span:
            _span_defaults(span, "decide", phase, concurrency, active)
            try:
                decision = self._decide(state, event)
            except Exception as exc:
                span.record_exception(exc)
                raise
            span.set_attribute("selected_depth", _depth_label(decision.selected_depth))
            span.set_attribute("reason", decision.reason.value)
            span.set_attribute("outcome", "success")
            return decision

    def dump_state(self, state: PolicyState) -> bytes:
        """Return owned canonical JSON bounded to 64 KiB."""

        with self._tracer.start_as_current_span("rocket.qwen38.mtp.policy") as span:
            _span_defaults(span, "dump", "none", "none", "none")
            try:
                encoded = json.dumps(
                    _state_to_record(state), sort_keys=True, separators=(",", ":")
                ).encode("ascii")
                if not 0 < len(encoded) <= MAX_STATE_BYTES:
                    raise MtpPolicyError("serialized policy state exceeds 65536 bytes")
            except Exception as exc:
                span.record_exception(exc)
                raise
            span.set_attribute("outcome", "success")
            return encoded

    def load_state(self, encoded: bytes) -> PolicyState:
        """Validate bounded JSON and matching policy configuration."""

        with self._tracer.start_as_current_span("rocket.qwen38.mtp.policy") as span:
            _span_defaults(span, "load", "none", "none", "none")
            try:
                if not isinstance(encoded, bytes) or not 0 < len(encoded) <= MAX_STATE_BYTES:
                    raise MtpPolicyError("serialized policy state must be 1 through 65536 bytes")
                try:
                    raw = json.loads(encoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise MtpPolicyError("serialized policy state is invalid JSON") from exc
                state = _state_from_record(raw)
                self._validate_state(state)
            except Exception as exc:
                span.record_exception(exc)
                raise
            span.set_attribute("outcome", "success")
            return state

    def _decide(self, state: PolicyState, event: PolicyEvent) -> PolicyDecision:
        self._validate_state(state)
        if not isinstance(event, PolicyEvent):
            raise MtpPolicyError("event must be PolicyEvent")
        if event.phase not in _PHASE_TRANSITIONS[state.phase]:
            raise MtpPolicyError(
                f"session phase cannot move from {state.phase.value} to {event.phase.value}"
            )
        cohort = _cohort_for(state.cohorts, event.concurrency)
        if cohort is None:
            if event.phase in (SessionPhase.EARLY_DECODE, SessionPhase.STEADY_DECODE):
                raise MtpPolicyError(
                    f"decode concurrency c{event.concurrency} has no exact ceiling"
                )
            if event.observation is not None:
                raise MtpPolicyError("unconfigured concurrency cannot contribute observations")
            next_state = PolicyState(
                state.config_fingerprint,
                event.phase,
                min(MAX_COUNTER, state.decisions + 1),
                state.cohorts,
            )
            return PolicyDecision(
                MtpDepth.K0,
                DecisionReason.PHASE_K0,
                _residency_commands(event.resident_depth, MtpDepth.K0),
                next_state,
            )
        if event.observation is not None:
            if event.observation.concurrency != event.concurrency:
                raise MtpPolicyError("observation concurrency does not match its event")
            if event.phase not in (
                SessionPhase.EARLY_DECODE,
                SessionPhase.STEADY_DECODE,
            ):
                raise MtpPolicyError("observations are accepted only during decode")
            cohort = _observe(cohort, event.observation)

        if event.phase in (SessionPhase.PREFILL, SessionPhase.RESTORING, SessionPhase.DRAINING):
            selected = MtpDepth.K0
            reason = DecisionReason.PHASE_K0
        else:
            ceiling = self._config.ceiling_for(event.concurrency)
            if ceiling is None:
                raise MtpPolicyError(
                    f"decode concurrency c{event.concurrency} has no exact ceiling"
                )
            cap = MtpDepth(min(int(ceiling), event.remaining_tokens - 1))
            recommended = _recommended_depth(cohort, cap, event.resident_depth, self._config)
            transition_reason: DecisionReason | None = None
            if event.observation is not None:
                cohort, transition_reason = _apply_window(
                    cohort,
                    current_passes=recommended >= cohort.active_depth,
                    promoted_depth=recommended,
                    config=self._config,
                )
            if cohort.active_depth > ceiling:
                cohort = replace(cohort, active_depth=ceiling)
                transition_reason = DecisionReason.DEMOTE

            if cohort.decode_rounds < len(PROBE_DEPTHS):
                requested = MtpDepth(PROBE_DEPTHS[cohort.decode_rounds])
                selected = MtpDepth(min(int(requested), int(cap)))
                reason = DecisionReason.PROBE
            elif (
                cohort.eligible_rounds
                and cohort.eligible_rounds % self._config.exploration_interval == 0
            ):
                requested = MtpDepth(min(7, int(cohort.active_depth) + 1))
                selected = MtpDepth(min(int(requested), int(cap)))
                reason = DecisionReason.EXPLORE
            else:
                requested = cohort.active_depth
                selected = MtpDepth(min(int(requested), int(cap)))
                reason = DecisionReason.POLICY
            if selected < requested:
                reason = DecisionReason.REMAINING_CAP
            elif transition_reason is not None and selected is cohort.active_depth:
                reason = transition_reason
            cohort = replace(
                cohort,
                decode_rounds=min(MAX_COUNTER, cohort.decode_rounds + 1),
                eligible_rounds=min(MAX_COUNTER, cohort.eligible_rounds + 1),
            )

        cohorts = tuple(
            sorted(
                tuple(item for item in state.cohorts if item.concurrency != event.concurrency)
                + (cohort,),
                key=lambda item: item.concurrency,
            )
        )
        next_state = PolicyState(
            state.config_fingerprint,
            event.phase,
            min(MAX_COUNTER, state.decisions + 1),
            cohorts,
        )
        return PolicyDecision(
            selected,
            reason,
            _residency_commands(event.resident_depth, selected),
            next_state,
        )

    def _validate_state(self, state: PolicyState) -> None:
        if not isinstance(state, PolicyState):
            raise MtpPolicyError("state must be PolicyState")
        if state.config_fingerprint != self._config.fingerprint:
            raise MtpPolicyError("policy state configuration fingerprint mismatch")
        configured = tuple(item.concurrency for item in self._config.ceilings)
        if tuple(item.concurrency for item in state.cohorts) != configured:
            raise MtpPolicyError("policy state cohort inventory does not match configuration")
        for cohort in state.cohorts:
            ceiling = self._config.ceiling_for(cohort.concurrency)
            if ceiling is None or cohort.active_depth > ceiling:
                raise MtpPolicyError("policy state active depth exceeds its exact ceiling")


def _observe(cohort: CohortState, observation: PolicyObservation) -> CohortState:
    successes = list(cohort.position_successes)
    trials = list(cohort.position_trials)
    for index, (success, trial) in enumerate(
        zip(observation.position_successes, observation.position_trials)
    ):
        successes[index] = min(MAX_ACCEPTED_TOKENS, successes[index] + success)
        trials[index] = min(MAX_ACCEPTED_TOKENS, trials[index] + trial)
        successes[index] = min(successes[index], trials[index])
    estimates = []
    found = False
    for estimate in cohort.estimates:
        if estimate.observation.depth is observation.depth:
            estimates.append(
                DepthEstimate(observation, min(MAX_COUNTER, estimate.samples + 1))
            )
            found = True
        else:
            estimates.append(estimate)
    if not found:
        estimates.append(DepthEstimate(observation, 1))
    return replace(
        cohort,
        position_successes=tuple(successes),
        position_trials=tuple(trials),
        estimates=tuple(sorted(estimates, key=lambda item: int(item.observation.depth))),
    )


def _apply_window(
    cohort: CohortState,
    *,
    current_passes: bool,
    promoted_depth: MtpDepth,
    config: PolicyConfig,
) -> tuple[CohortState, DecisionReason | None]:
    results = cohort.window_results + (current_passes,)
    if len(results) < config.demotion_window_rounds:
        return replace(cohort, window_results=results), None
    failed = not all(results)
    failing = cohort.consecutive_failing_windows + 1 if failed else 0
    depth = cohort.active_depth
    reason: DecisionReason | None = None
    if failing >= config.demotion_failing_windows and depth is not MtpDepth.K0:
        depth = MtpDepth(int(depth) - 1)
        failing = 0
        reason = DecisionReason.DEMOTE
    elif not failed and promoted_depth > depth:
        depth = promoted_depth
        reason = DecisionReason.PROMOTE
    return (
        replace(
            cohort,
            active_depth=depth,
            consecutive_failing_windows=failing,
            window_results=(),
        ),
        reason,
    )


def _recommended_depth(
    cohort: CohortState,
    cap: MtpDepth,
    resident_depth: MtpDepth,
    config: PolicyConfig,
) -> MtpDepth:
    selected = MtpDepth.K0
    incumbent = _estimate_for(cohort, selected)
    if incumbent is None:
        return selected
    for raw_depth in range(1, int(cap) + 1):
        candidate_depth = MtpDepth(raw_depth)
        candidate = _estimate_for(cohort, candidate_depth)
        if candidate is None or not _confident_promotion(
            candidate, incumbent, cohort, resident_depth, config
        ):
            break
        selected = candidate_depth
        incumbent = candidate
    return selected


def _confident_promotion(
    candidate: DepthEstimate,
    incumbent: DepthEstimate,
    cohort: CohortState,
    resident_depth: MtpDepth,
    config: PolicyConfig,
) -> bool:
    left = candidate.observation
    right = incumbent.observation
    if left.clears_acceptance_floors != right.clears_acceptance_floors:
        return left.clears_acceptance_floors
    left_lower, _left_mean, _left_upper = _expected_tokens(cohort, left.depth)
    _right_lower, _right_mean, right_upper = _expected_tokens(cohort, right.depth)
    if left_lower <= 0 or right_upper <= 0:
        return False
    lazy = (
        config.lazy_weight_bytes / config.lazy_residency_rounds
        if left.depth > resident_depth and left.depth is not MtpDepth.K0
        else 0.0
    )
    candidate_worst_bytes = (left.modeled_step_bytes + lazy) / left_lower
    incumbent_best_bytes = right.modeled_step_bytes / right_upper
    if candidate_worst_bytes > incumbent_best_bytes * (
        1.0 - config.promotion_margin_ppm / PPM
    ):
        return False
    if (
        left.roofline_efficiency_ppm / candidate_worst_bytes
        <= right.roofline_efficiency_ppm / incumbent_best_bytes
    ):
        return False
    return left.cost_ns_per_accepted_token <= right.cost_ns_per_accepted_token


def _expected_tokens(
    cohort: CohortState, depth: MtpDepth
) -> tuple[float, float, float]:
    lower = mean = upper = 1.0
    for index in range(int(depth)):
        successes = cohort.position_successes[index]
        trials = cohort.position_trials[index]
        lo, hi = _wilson(successes, trials)
        lower += lo
        upper += hi
        mean += successes / trials if trials else 0.0
    return lower, mean, upper


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    z = 1.96
    probability = successes / trials
    scale = 1.0 + z * z / trials
    center = (probability + z * z / (2 * trials)) / scale
    radius = z * math.sqrt(
        probability * (1.0 - probability) / trials + z * z / (4 * trials * trials)
    ) / scale
    return max(0.0, center - radius), min(1.0, center + radius)


def _estimate_for(cohort: CohortState, depth: MtpDepth) -> DepthEstimate | None:
    return next(
        (item for item in cohort.estimates if item.observation.depth is depth), None
    )


def _cohort_for(cohorts: tuple[CohortState, ...], concurrency: int) -> CohortState | None:
    return next((item for item in cohorts if item.concurrency == concurrency), None)


def _residency_commands(
    resident_depth: MtpDepth, selected_depth: MtpDepth
) -> tuple[ResidencyCommand, ...]:
    if selected_depth > resident_depth:
        return tuple(
            ResidencyCommand(ResidencyAction.LOAD, MtpDepth(depth))
            for depth in range(int(resident_depth) + 1, int(selected_depth) + 1)
        )
    return tuple(
        ResidencyCommand(ResidencyAction.EVICT, MtpDepth(depth))
        for depth in range(int(resident_depth), int(selected_depth), -1)
    )


def _require_int(value: object, minimum: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise MtpPolicyError(f"{name} must be an integer in {minimum} through {maximum}")


def _require_depth(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, MtpDepth):
        raise MtpPolicyError(f"{name} must be K0 through K7")


def _depth_label(depth: MtpDepth) -> str:
    return f"k{int(depth)}"


def _span_defaults(
    span: _Span, operation: str, phase: str, concurrency: str, active_depth: str
) -> None:
    span.set_attribute("operation", operation)
    span.set_attribute("session_phase", phase)
    span.set_attribute("concurrency", concurrency)
    span.set_attribute("active_depth", active_depth)
    span.set_attribute("selected_depth", "none")
    span.set_attribute("reason", "none")
    span.set_attribute("outcome", "failure")


def _state_to_record(state: PolicyState) -> dict[str, object]:
    if not isinstance(state, PolicyState):
        raise MtpPolicyError("state must be PolicyState")
    return {
        "schema": SCHEMA,
        "config": state.config_fingerprint,
        "phase": state.phase.value,
        "decisions": state.decisions,
        "cohorts": [_cohort_to_record(cohort) for cohort in state.cohorts],
    }


def _cohort_to_record(cohort: CohortState) -> dict[str, object]:
    return {
        "concurrency": cohort.concurrency,
        "active_depth": int(cohort.active_depth),
        "decode_rounds": cohort.decode_rounds,
        "eligible_rounds": cohort.eligible_rounds,
        "failing_windows": cohort.consecutive_failing_windows,
        "window_results": list(cohort.window_results),
        "position_successes": list(cohort.position_successes),
        "position_trials": list(cohort.position_trials),
        "estimates": [_estimate_to_record(item) for item in cohort.estimates],
    }


def _estimate_to_record(estimate: DepthEstimate) -> dict[str, object]:
    observation = estimate.observation
    return {
        "concurrency": observation.concurrency,
        "depth": int(observation.depth),
        "position_successes": list(observation.position_successes),
        "position_trials": list(observation.position_trials),
        "accepted_tokens": observation.accepted_tokens,
        "cost_ns": observation.cost_ns,
        "modeled_step_bytes": observation.modeled_step_bytes,
        "aggregate_rate": observation.aggregate_tok_s_milli,
        "per_stream_rate": observation.per_stream_tok_s_milli,
        "roofline_rate": observation.whole_decode_roofline_tok_s_milli,
        "samples": estimate.samples,
    }


def _state_from_record(raw: object) -> PolicyState:
    value = _exact_dict(
        raw,
        {"schema", "config", "phase", "decisions", "cohorts"},
        "policy state",
    )
    if value["schema"] != SCHEMA:
        raise MtpPolicyError("policy state schema is unsupported")
    cohorts_raw = value["cohorts"]
    if not isinstance(cohorts_raw, list) or not 0 < len(cohorts_raw) <= 16:
        raise MtpPolicyError("serialized cohorts must contain 1 through 16 values")
    try:
        phase = SessionPhase(value["phase"])
    except (TypeError, ValueError) as exc:
        raise MtpPolicyError("serialized policy phase is invalid") from exc
    return PolicyState(
        value["config"],
        phase,
        value["decisions"],
        tuple(_cohort_from_record(item) for item in cohorts_raw),
    )


def _cohort_from_record(raw: object) -> CohortState:
    value = _exact_dict(
        raw,
        {
            "concurrency",
            "active_depth",
            "decode_rounds",
            "eligible_rounds",
            "failing_windows",
            "window_results",
            "position_successes",
            "position_trials",
            "estimates",
        },
        "cohort",
    )
    estimates = value["estimates"]
    if not isinstance(estimates, list) or len(estimates) > 8:
        raise MtpPolicyError("serialized estimates must contain at most eight values")
    return CohortState(
        value["concurrency"],
        _decoded_depth(value["active_depth"], "active depth"),
        value["decode_rounds"],
        value["eligible_rounds"],
        value["failing_windows"],
        _bool_tuple(value["window_results"], "window results"),
        _int_tuple(value["position_successes"], 7, "position successes"),
        _int_tuple(value["position_trials"], 7, "position trials"),
        tuple(_estimate_from_record(item) for item in estimates),
    )


def _estimate_from_record(raw: object) -> DepthEstimate:
    value = _exact_dict(
        raw,
        {
            "concurrency",
            "depth",
            "position_successes",
            "position_trials",
            "accepted_tokens",
            "cost_ns",
            "modeled_step_bytes",
            "aggregate_rate",
            "per_stream_rate",
            "roofline_rate",
            "samples",
        },
        "estimate",
    )
    depth = _decoded_depth(value["depth"], "estimate depth")
    observation = PolicyObservation(
        value["concurrency"],
        depth,
        _int_tuple(value["position_successes"], int(depth), "position successes"),
        _int_tuple(value["position_trials"], int(depth), "position trials"),
        value["accepted_tokens"],
        value["cost_ns"],
        value["modeled_step_bytes"],
        value["aggregate_rate"],
        value["per_stream_rate"],
        value["roofline_rate"],
    )
    return DepthEstimate(observation, value["samples"])


def _decoded_depth(raw: object, name: str) -> MtpDepth:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise MtpPolicyError(f"{name} must be an integer")
    try:
        return MtpDepth(raw)
    except ValueError as exc:
        raise MtpPolicyError(f"{name} must be K0 through K7") from exc


def _exact_dict(raw: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise MtpPolicyError(f"{name} fields are invalid")
    return raw


def _int_tuple(raw: object, length: int, name: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or len(raw) != length:
        raise MtpPolicyError(f"{name} length is invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise MtpPolicyError(f"{name} values must be integers")
    return tuple(raw)


def _bool_tuple(raw: object, name: str) -> tuple[bool, ...]:
    if not isinstance(raw, list) or any(type(value) is not bool for value in raw):
        raise MtpPolicyError(f"{name} values must be booleans")
    return tuple(raw)


__all__ = [
    "AGGREGATE_FLOOR_MILLI",
    "AdaptiveMtpPolicy",
    "ConcurrencyCeiling",
    "DecisionReason",
    "MtpDepth",
    "MtpPolicyError",
    "PER_STREAM_FLOOR_MILLI",
    "PolicyConfig",
    "PolicyDecision",
    "PolicyEvent",
    "PolicyObservation",
    "PolicyState",
    "ResidencyAction",
    "ResidencyCommand",
    "SessionPhase",
]
