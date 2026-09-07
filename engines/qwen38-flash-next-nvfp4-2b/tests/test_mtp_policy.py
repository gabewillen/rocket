from __future__ import annotations

import json
import unittest
from dataclasses import replace

from qwen38_slab.mtp_policy import (
    MAX_ACCEPTED_TOKENS,
    MAX_COST_NS,
    MAX_COUNTER,
    MAX_MODELED_BYTES,
    MAX_RATE_MILLI,
    MAX_STATE_BYTES,
    PROBE_DEPTHS,
    AdaptiveMtpPolicy,
    CohortState,
    ConcurrencyCeiling,
    DecisionReason,
    DepthEstimate,
    MtpDepth,
    MtpPolicyError,
    PolicyConfig,
    PolicyEvent,
    PolicyObservation,
    PolicyState,
    ResidencyAction,
    SessionPhase,
    matched_live_policy_config,
)


class Span:
    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}
        self.exceptions: list[str] = []

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value

    def record_exception(self, exception: BaseException) -> None:
        self.exceptions.append(type(exception).__name__)


class Tracer:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def start_as_current_span(self, name: str) -> Span:
        span = Span()
        span.name = name
        self.spans.append(span)
        return span


def observation(
    depth: MtpDepth,
    *,
    concurrency: int = 1,
    successes: tuple[int, ...] | None = None,
    trials: tuple[int, ...] | None = None,
    accepted: int = 1000,
    cost_ns: int = 1_000_000,
    step_bytes: int = 100_000,
    aggregate: int = 400_000,
    per_stream: int = 25_000,
    roofline: int = 500_000,
) -> PolicyObservation:
    count = int(depth)
    return PolicyObservation(
        concurrency=concurrency,
        depth=depth,
        position_successes=successes if successes is not None else (900,) * count,
        position_trials=trials if trials is not None else (1000,) * count,
        accepted_tokens=accepted,
        cost_ns=cost_ns,
        modeled_step_bytes=step_bytes,
        aggregate_tok_s_milli=aggregate,
        per_stream_tok_s_milli=per_stream,
        whole_decode_roofline_tok_s_milli=roofline,
    )


class AdaptiveMtpPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracer = Tracer()
        self.config = PolicyConfig(
            (
                ConcurrencyCeiling(1, MtpDepth.K7),
                ConcurrencyCeiling(16, MtpDepth.K1),
            )
        )
        self.policy = AdaptiveMtpPolicy(self.tracer, self.config)

    def test_forced_k0_probe_schedule_remaining_cap_and_lazy_commands(self) -> None:
        state = self.policy.initial_state()
        prefill = self.policy.decide(
            state, PolicyEvent(SessionPhase.PREFILL, 1, 100, MtpDepth.K7)
        )
        self.assertEqual(prefill.selected_depth, MtpDepth.K0)
        self.assertEqual(prefill.reason, DecisionReason.PHASE_K0)
        self.assertEqual(
            [(item.action, item.depth) for item in prefill.residency_commands],
            [(ResidencyAction.EVICT, MtpDepth(value)) for value in range(7, 0, -1)],
        )

        state = self.policy.initial_state()
        selected = []
        for _ in PROBE_DEPTHS:
            decision = self.policy.decide(
                state,
                PolicyEvent(SessionPhase.EARLY_DECODE, 1, 100, MtpDepth.K0),
            )
            selected.append(int(decision.selected_depth))
            state = decision.next_state
        self.assertEqual(tuple(selected), PROBE_DEPTHS)
        capped = self.policy.decide(
            self.policy.initial_state(),
            PolicyEvent(SessionPhase.EARLY_DECODE, 1, 1, MtpDepth.K0),
        )
        self.assertEqual((capped.selected_depth, capped.reason), (MtpDepth.K0, DecisionReason.REMAINING_CAP))

    def test_exact_concurrency_caps_do_not_extrapolate(self) -> None:
        with self.assertRaisesRegex(MtpPolicyError, "c8 has no exact ceiling"):
            self.policy.decide(
                self.policy.initial_state(),
                PolicyEvent(SessionPhase.EARLY_DECODE, 8, 100, MtpDepth.K0),
            )

        state = self.policy.initial_state()
        observed_depths = []
        for _ in PROBE_DEPTHS:
            decision = self.policy.decide(
                state,
                PolicyEvent(SessionPhase.EARLY_DECODE, 16, 100, MtpDepth.K0),
            )
            observed_depths.append(decision.selected_depth)
            self.assertTrue(all(command.depth <= MtpDepth.K1 for command in decision.residency_commands))
            state = decision.next_state
        self.assertEqual(set(observed_depths), {MtpDepth.K1})

        forged = replace(
            state,
            cohorts=tuple(
                replace(item, active_depth=MtpDepth.K7) if item.concurrency == 16 else item
                for item in state.cohorts
            ),
        )
        with self.assertRaisesRegex(MtpPolicyError, "exceeds its exact ceiling"):
            self.policy.decide(
                forged,
                PolicyEvent(SessionPhase.EARLY_DECODE, 16, 100, MtpDepth.K0),
            )

    def test_matched_k1_k7_envelope_keeps_low_depth_lazy_and_tapers_high(self) -> None:
        config = matched_live_policy_config()
        self.assertEqual(
            tuple((item.concurrency, item.max_depth) for item in config.ceilings),
            (
                (1, MtpDepth.K7),
                (2, MtpDepth.K7),
                (4, MtpDepth.K7),
                (8, MtpDepth.K1),
                (16, MtpDepth.K1),
            ),
        )
        policy = AdaptiveMtpPolicy(Tracer(), config)
        with self.assertRaisesRegex(MtpPolicyError, "c3 has no exact ceiling"):
            policy.decide(
                policy.initial_state(),
                PolicyEvent(SessionPhase.EARLY_DECODE, 3, 100, MtpDepth.K0),
            )

        for concurrency in (8, 16):
            state = policy.initial_state()
            for _ in PROBE_DEPTHS:
                decision = policy.decide(
                    state,
                    PolicyEvent(
                        SessionPhase.EARLY_DECODE,
                        concurrency,
                        100,
                        MtpDepth.K0,
                    ),
                )
                self.assertLessEqual(decision.selected_depth, MtpDepth.K1)
                self.assertTrue(
                    all(command.depth <= MtpDepth.K1 for command in decision.residency_commands)
                )
                state = decision.next_state

    def test_c1_can_explore_k7_without_claiming_it_is_optimal(self) -> None:
        state = self.policy.initial_state()
        cohort = next(item for item in state.cohorts if item.concurrency == 1)
        cohort = replace(
            cohort,
            active_depth=MtpDepth.K6,
            decode_rounds=len(PROBE_DEPTHS),
            eligible_rounds=64,
        )
        state = replace(
            state,
            phase=SessionPhase.STEADY_DECODE,
            cohorts=(cohort, state.cohorts[1]),
        )
        decision = self.policy.decide(
            state,
            PolicyEvent(SessionPhase.STEADY_DECODE, 1, 100, MtpDepth.K6),
        )
        self.assertEqual((decision.selected_depth, decision.reason), (MtpDepth.K7, DecisionReason.EXPLORE))
        self.assertEqual(
            decision.residency_commands,
            (decision.residency_commands[0],),
        )
        self.assertEqual(
            (decision.residency_commands[0].action, decision.residency_commands[0].depth),
            (ResidencyAction.LOAD, MtpDepth.K7),
        )
        self.assertEqual(decision.next_state.cohorts[0].active_depth, MtpDepth.K6)

    def test_wilson_promotion_and_two_failing_windows_demote(self) -> None:
        state = self.policy.initial_state()
        cohort = state.cohorts[1]
        k0 = observation(MtpDepth.K0, concurrency=16, step_bytes=100_000, aggregate=400_000)
        k1 = observation(
            MtpDepth.K1,
            concurrency=16,
            successes=(950,),
            trials=(1000,),
            accepted=1900,
            cost_ns=900_000,
            step_bytes=130_000,
            aggregate=450_000,
        )
        self.assertTrue(k0.clears_acceptance_floors)
        self.assertTrue(k1.clears_acceptance_floors)
        cohort = replace(
            cohort,
            decode_rounds=len(PROBE_DEPTHS),
            eligible_rounds=17,
            position_successes=(950, 0, 0, 0, 0, 0, 0),
            position_trials=(1000, 0, 0, 0, 0, 0, 0),
            estimates=(DepthEstimate(k0, 1), DepthEstimate(k1, 1)),
        )
        state = replace(state, phase=SessionPhase.STEADY_DECODE, cohorts=(state.cohorts[0], cohort))
        for _ in range(8):
            decision = self.policy.decide(
                state,
                PolicyEvent(SessionPhase.STEADY_DECODE, 16, 100, MtpDepth.K1, k1),
            )
            state = decision.next_state
        self.assertEqual((decision.selected_depth, decision.reason), (MtpDepth.K1, DecisionReason.PROMOTE))
        self.assertEqual(state.cohorts[1].active_depth, MtpDepth.K1)

        slow_k1 = observation(
            MtpDepth.K1,
            concurrency=16,
            successes=(50,),
            trials=(1000,),
            accepted=1050,
            cost_ns=2_000_000,
            step_bytes=200_000,
            aggregate=330_000,
        )
        for _ in range(16):
            decision = self.policy.decide(
                state,
                PolicyEvent(SessionPhase.STEADY_DECODE, 16, 100, MtpDepth.K1, slow_k1),
            )
            state = decision.next_state
        self.assertEqual((decision.selected_depth, decision.reason), (MtpDepth.K0, DecisionReason.DEMOTE))
        self.assertEqual(state.cohorts[1].active_depth, MtpDepth.K0)

    def test_matched_live_k1_k3_replay_stays_inside_measured_c16_cap(self) -> None:
        # Throughput/cost and final-window acceptance are copied from the three
        # matched c16 runs. The 500 tok/s roofline is an explicit replay input,
        # not a measured specialized-engine ceiling.
        matched = (
            PolicyObservation(16, MtpDepth.K1, (1153,), (1856,), 4096, 39_093_570_281, 43_058_601_575, 108_763, 12_462, 500_000),
            PolicyObservation(16, MtpDepth.K2, (770, 476), (1184, 1184), 4096, 39_861_595_448, 55_338_212_409, 106_564, 12_360, 500_000),
            PolicyObservation(16, MtpDepth.K3, (718, 443, 274), (1168, 1168, 1168), 4096, 41_636_169_139, 64_872_150_224, 101_811, 11_595, 500_000),
        )
        state = self.policy.initial_state()
        for item in matched:
            decision = self.policy.decide(
                state,
                PolicyEvent(SessionPhase.EARLY_DECODE, 16, 100, MtpDepth.K0, item),
            )
            self.assertLessEqual(decision.selected_depth, MtpDepth.K1)
            self.assertTrue(all(command.depth <= MtpDepth.K1 for command in decision.residency_commands))
            self.assertFalse(item.clears_acceptance_floors)
            state = decision.next_state
        c16 = next(item for item in state.cohorts if item.concurrency == 16)
        self.assertEqual(tuple(value.observation.depth for value in c16.estimates), (MtpDepth.K1, MtpDepth.K2, MtpDepth.K3))

    def test_state_is_canonical_bounded_and_atomic_record_compatible(self) -> None:
        state = self.policy.initial_state()
        encoded = self.policy.dump_state(state)
        self.assertLessEqual(len(encoded), MAX_STATE_BYTES)
        self.assertEqual(encoded, self.policy.dump_state(state))
        self.assertEqual(self.policy.load_state(encoded), state)
        prefix_record = json.dumps({"accepted_tokens": 37, "mtp_policy": json.loads(encoded)})
        self.assertEqual(json.loads(prefix_record)["mtp_policy"]["schema"], "rocket.qwen38-adaptive-mtp-policy.v1")

        malformed = json.loads(encoded)
        malformed["extra"] = True
        with self.assertRaisesRegex(MtpPolicyError, "fields"):
            self.policy.load_state(json.dumps(malformed).encode())
        with self.assertRaisesRegex(MtpPolicyError, "1 through 65536"):
            self.policy.load_state(b"x" * (MAX_STATE_BYTES + 1))
        other = AdaptiveMtpPolicy(
            Tracer(), PolicyConfig((ConcurrencyCeiling(1, MtpDepth.K3),))
        )
        with self.assertRaisesRegex(MtpPolicyError, "configuration fingerprint"):
            other.load_state(encoded)

    def test_maximum_valid_state_fits_serialization_envelope(self) -> None:
        config = PolicyConfig(
            tuple(ConcurrencyCeiling(value, MtpDepth.K7) for value in range(1, 17))
        )
        policy = AdaptiveMtpPolicy(Tracer(), config)
        cohorts = []
        for concurrency in range(1, 17):
            estimates = []
            for depth in MtpDepth:
                estimates.append(
                    DepthEstimate(
                        PolicyObservation(
                            concurrency,
                            depth,
                            (MAX_ACCEPTED_TOKENS,) * int(depth),
                            (MAX_ACCEPTED_TOKENS,) * int(depth),
                            MAX_ACCEPTED_TOKENS,
                            MAX_COST_NS,
                            MAX_MODELED_BYTES,
                            MAX_RATE_MILLI,
                            MAX_RATE_MILLI,
                            MAX_RATE_MILLI,
                        ),
                        MAX_COUNTER,
                    )
                )
            cohorts.append(
                CohortState(
                    concurrency,
                    MtpDepth.K7,
                    MAX_COUNTER,
                    MAX_COUNTER,
                    1,
                    (True,) * 7,
                    (MAX_ACCEPTED_TOKENS,) * 7,
                    (MAX_ACCEPTED_TOKENS,) * 7,
                    tuple(estimates),
                )
            )
        state = PolicyState(
            config.fingerprint,
            SessionPhase.STEADY_DECODE,
            MAX_COUNTER,
            tuple(cohorts),
        )
        encoded = policy.dump_state(state)
        self.assertEqual(len(encoded), 48_754)
        self.assertLessEqual(len(encoded), MAX_STATE_BYTES)
        self.assertEqual(policy.load_state(encoded), state)

    def test_otel_attributes_have_finite_cardinality(self) -> None:
        state = self.policy.initial_state()
        self.policy.decide(
            state, PolicyEvent(SessionPhase.EARLY_DECODE, 1, 100, MtpDepth.K0)
        )
        with self.assertRaises(MtpPolicyError):
            self.policy.decide(
                state, PolicyEvent(SessionPhase.EARLY_DECODE, 8, 100, MtpDepth.K0)
            )
        encoded = self.policy.dump_state(state)
        self.policy.load_state(encoded)
        allowed = {
            "operation", "session_phase", "concurrency", "active_depth",
            "selected_depth", "reason", "outcome",
        }
        self.assertTrue(self.tracer.spans)
        self.assertTrue(all(set(span.attributes) == allowed for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["operation"] in {"decide", "dump", "load"} for span in self.tracer.spans))
        self.assertTrue(all(span.attributes["selected_depth"] in {"none", *(f"k{x}" for x in range(8))} for span in self.tracer.spans))
        self.assertTrue(any(span.attributes["outcome"] == "failure" for span in self.tracer.spans))

    def test_sequence_and_observation_binding_fail_closed(self) -> None:
        state = self.policy.initial_state()
        draining = self.policy.decide(
            state, PolicyEvent(SessionPhase.DRAINING, 1, 100, MtpDepth.K0)
        ).next_state
        with self.assertRaisesRegex(MtpPolicyError, "cannot move"):
            self.policy.decide(
                draining,
                PolicyEvent(SessionPhase.EARLY_DECODE, 1, 100, MtpDepth.K0),
            )
        with self.assertRaisesRegex(MtpPolicyError, "only during decode"):
            self.policy.decide(
                state,
                PolicyEvent(
                    SessionPhase.PREFILL,
                    1,
                    100,
                    MtpDepth.K0,
                    observation(MtpDepth.K1),
                ),
            )
        with self.assertRaisesRegex(MtpPolicyError, "does not match"):
            self.policy.decide(
                state,
                PolicyEvent(
                    SessionPhase.EARLY_DECODE,
                    16,
                    100,
                    MtpDepth.K0,
                    observation(MtpDepth.K1, concurrency=1),
                ),
            )


if __name__ == "__main__":
    unittest.main()
