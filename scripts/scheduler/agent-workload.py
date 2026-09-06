#!/usr/bin/env python3
"""Forked-prefix agent workload generator.

Simulates the load shape of agent/subagent serving: one root session holds a
long shared system+context prefix, planner sessions fork bursts of 2-5
subagents that inherit the prefix and diverge, tool results land as
mid-stream prefill bursts, and a fraction of sessions go idle and resume.

Output is a reproducible (seeded) JSONL trace, one event per line, plus a
summary block at the end (see README.md in this directory).

Event schema (all events carry "t": tick, "seq": monotonic emission order):
  session_start  {parent_id: str|null, session: str, depth: int,
                   fork_point_tokens: int}
  prefill        {session: str, n_tokens: int, kind: "root"|"fork"|"tool"}
  decode         {session: str, n_tokens: int}
  idle           {session: str, seconds: float}
  resume         {session: str}

stdlib only. Usage:
  python3 agent-workload.py --concurrency 16 --depth 2 > trace.jsonl
  python3 agent-workload.py --validate --trace trace.jsonl
"""
import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_ATTENTION_YAML = REPO_ROOT / "fuels" / "glm-5.3-flash" / "attention.yaml"


# --------------------------------------------------------------------- KV
def read_kv_bytes_per_token(yaml_path: Path) -> dict:
    """Pull per-token KV bytes out of the fuel's attention geometry file.

    Reads fields, not the whole YAML (stdlib only, no PyYAML): the two
    per-token costs of the 11 MLA+indexer layers, the MLA layer count, and
    the fixed (context-independent) KDA recurrent+conv state per stream.
    Fields cited:
      fuels/glm-5.3-flash/attention.yaml
        layers.mla_count                        -> mla_count
        bytes.mla_kv_per_token_per_layer        -> mla_bytes_per_token_per_layer
        bytes.indexer_key_per_token_per_layer   -> indexer_bytes_per_token_per_layer
        bytes.kda_state_per_stream.total        -> kda_fixed_bytes_per_stream (bf16)
    """
    text = yaml_path.read_text()

    def grab(pattern: str, s: str) -> int:
        m = re.search(pattern, s, re.M)
        if not m:
            raise ValueError(f"attention.yaml: pattern not found: {pattern}")
        return int(m.group(1))

    mla_count = grab(r"^\s*mla_count:\s*(\d+)", text)
    mla_bpt = grab(r"^\s*mla_kv_per_token_per_layer:\s*(\d+)", text)
    idx_bpt = grab(r"^\s*indexer_key_per_token_per_layer:\s*(\d+)", text)

    kda_block = re.search(
        r"kda_state_per_stream:(.*?)\n\s*mla_kv_per_token_per_layer:", text, re.S
    )
    if not kda_block:
        raise ValueError("attention.yaml: kda_state_per_stream block not found")
    kda_fixed = grab(r"^\s*total:\s*(\d+)", kda_block.group(1))

    return {
        "mla_count": mla_count,
        "mla_bytes_per_token_per_layer": mla_bpt,
        "indexer_bytes_per_token_per_layer": idx_bpt,
        # growing KV cost per token across the 11 MLA+indexer layers
        "kv_bytes_per_token": (mla_bpt + idx_bpt) * mla_count,
        # fixed per-stream cost, independent of context length, bf16 KDA state
        "kda_fixed_bytes_per_stream": kda_fixed,
    }


# ---------------------------------------------------------------- session
@dataclass
class Session:
    id: str
    parent_id: Optional[str]
    depth: int
    is_root: bool
    fork_point_tokens: int  # tokens inherited from parent at fork time
    lifetime_ticks: int  # ticks before this session finishes
    tokens_generated: int = 0
    ticks_alive: int = 0
    status: str = "active"  # active | idle | done
    resume_at: int = -1
    tokens_until_tool_burst: int = 20
    last_served_tick: int = -1


# ------------------------------------------------------------- generator
def generate(args, rng: random.Random):
    events = []
    seq = 0

    def emit(t, ev):
        nonlocal seq
        ev = dict(ev)
        ev["t"] = t
        ev["seq"] = seq
        seq += 1
        events.append(ev)

    sessions: dict[str, Session] = {}
    next_id = [0]

    def new_id(prefix):
        next_id[0] += 1
        return f"{prefix}-{next_id[0]}"

    def start_session(t, parent: Optional[Session], fork_point_tokens, lifetime,
                       is_root=False):
        sid = new_id("root" if is_root else "sess")
        depth = 0 if is_root else parent.depth + 1
        s = Session(
            id=sid,
            parent_id=None if is_root else parent.id,
            depth=depth,
            is_root=is_root,
            fork_point_tokens=fork_point_tokens,
            lifetime_ticks=lifetime,
            tokens_until_tool_burst=rng.randint(20, 80),
        )
        sessions[sid] = s
        emit(t, {
            "event": "session_start",
            "session": sid,
            "parent_id": s.parent_id,
            "depth": depth,
            "fork_point_tokens": fork_point_tokens,
        })
        return s

    # Root session: the long shared system+context prefix, prefilled once.
    root = start_session(0, None, fork_point_tokens=0,
                          lifetime=args.duration_steps, is_root=True)
    emit(0, {"event": "prefill", "session": root.id,
             "n_tokens": args.prefix_tokens, "kind": "root"})
    root.fork_point_tokens = args.prefix_tokens
    stats = {
        "inherited_tokens": 0,      # tokens reused via shared prefix (not reprefilled)
        "fresh_tokens": args.prefix_tokens,  # prefill/decode tokens actually computed
        "peak_concurrent_decoding": 0,
        "total_sessions": 1,
        "idle_sessions": 0,
        "fork_events": 0,
    }
    # session -> total distinct-context length at last observation, used for
    # the with/without-sharing KV accounting at the end.
    context_len = {root.id: args.prefix_tokens}

    idle_target_frac = 0.20
    fork_chance_per_tick = 0.03  # planner bursts are rare relative to decode ticks

    for t in range(1, args.duration_steps + 1):
        # resume idle sessions whose wait has elapsed
        for s in sessions.values():
            if s.status == "idle" and t >= s.resume_at:
                s.status = "active"
                emit(t, {"event": "resume", "session": s.id})

        active = [s for s in sessions.values() if s.status == "active"]
        # round-robin: serve least-recently-served sessions first
        active.sort(key=lambda s: s.last_served_tick)
        admitted = active[: args.concurrency]
        stats["peak_concurrent_decoding"] = max(
            stats["peak_concurrent_decoding"], len(admitted)
        )

        for s in admitted:
            s.last_served_tick = t
            n = rng.randint(8, 32)
            emit(t, {"event": "decode", "session": s.id, "n_tokens": n})
            s.tokens_generated += n
            s.ticks_alive += 1
            stats["fresh_tokens"] += n
            context_len[s.id] = s.fork_point_tokens + s.tokens_generated

            # mid-stream tool-result prefill burst
            s.tokens_until_tool_burst -= n
            if s.tokens_until_tool_burst <= 0:
                burst = max(1, int(rng.gauss(args.tool_burst_tokens,
                                              args.tool_burst_tokens * 0.3)))
                emit(t, {"event": "prefill", "session": s.id,
                         "n_tokens": burst, "kind": "tool"})
                s.tokens_generated += burst
                stats["fresh_tokens"] += burst
                context_len[s.id] = s.fork_point_tokens + s.tokens_generated
                s.tokens_until_tool_burst = rng.randint(20, 80)

            # idle/resume: ~20% of sessions take one idle excursion
            if (s.status == "active" and not s.is_root
                    and rng.random() < idle_target_frac / max(s.lifetime_ticks, 1)):
                seconds = rng.uniform(5, 180)
                s.status = "idle"
                s.resume_at = t + max(1, int(seconds / 5))  # 5s per tick, illustrative
                emit(t, {"event": "idle", "session": s.id, "seconds": round(seconds, 1)})
                stats["idle_sessions"] += 1

            # planner fork burst: root and non-leaf-depth sessions spawn workers
            if (s.status == "active" and s.depth < args.depth
                    and rng.random() < fork_chance_per_tick):
                k = rng.randint(2, 5)
                stats["fork_events"] += 1
                fork_point = s.fork_point_tokens + s.tokens_generated
                # workers live shorter than the session that spawned them
                child_lifetime = max(10, int(
                    (args.duration_steps - t) * rng.uniform(0.1, 0.4)
                ))
                for _ in range(k):
                    child = start_session(t, s, fork_point, child_lifetime)
                    stats["total_sessions"] += 1
                    stats["inherited_tokens"] += fork_point
                    context_len[child.id] = fork_point
                    # small divergence prompt: subagent-specific instructions,
                    # not part of the shared/cached prefix
                    div = rng.randint(50, 200)
                    emit(t, {"event": "prefill", "session": child.id,
                             "n_tokens": div, "kind": "fork"})
                    child.tokens_generated += div
                    stats["fresh_tokens"] += div
                    context_len[child.id] = fork_point + div

            if s.ticks_alive >= s.lifetime_ticks:
                s.status = "done"

    return events, stats, context_len


def summarize(events, stats, context_len, kv):
    total_tokens = stats["inherited_tokens"] + stats["fresh_tokens"]
    prefix_shared_fraction = (
        stats["inherited_tokens"] / total_tokens if total_tokens else 0.0
    )

    kv_bpt = kv["kv_bytes_per_token"]
    kda_fixed = kv["kda_fixed_bytes_per_stream"]

    # Without sharing: every session pays KV for its full context independently.
    without_sharing = sum(
        clen * kv_bpt + kda_fixed for clen in context_len.values()
    )
    # With sharing: each session only pays for tokens unique to it (beyond
    # the point where it forked off its parent); the shared prefix is paid
    # once by the session that first extended into it. KDA state is a
    # recurrent hidden state per stream and is not shareable across forks.
    # Building this from the trace: session's "unique_tokens" = context_len -
    # fork_point_tokens for non-root; root's unique_tokens = its full context.
    return total_tokens, prefix_shared_fraction, without_sharing, kv_bpt, kda_fixed


def cmd_generate(args):
    rng = random.Random(args.seed)
    kv = read_kv_bytes_per_token(Path(args.attention_yaml))
    events, stats, context_len = generate(args, rng)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        for ev in events:
            out.write(json.dumps(ev, sort_keys=True) + "\n")

        # unique (non-inherited) tokens per session for the sharing accounting
        fork_point_by_session = {}
        for ev in events:
            if ev["event"] == "session_start":
                fork_point_by_session[ev["session"]] = ev["fork_point_tokens"]

        kv_bpt = kv["kv_bytes_per_token"]
        kda_fixed = kv["kda_fixed_bytes_per_stream"]
        without_sharing = 0
        with_sharing = 0
        for sid, clen in context_len.items():
            fp = fork_point_by_session.get(sid, 0)
            without_sharing += clen * kv_bpt + kda_fixed
            with_sharing += (clen - fp) * kv_bpt + kda_fixed
        # the shared prefix itself is paid once, by the root, inside
        # with_sharing already (root's fp=0 so its full context counts).

        total_tokens = stats["inherited_tokens"] + stats["fresh_tokens"]
        prefix_shared_fraction = (
            stats["inherited_tokens"] / total_tokens if total_tokens else 0.0
        )

        summary = {
            "event": "summary",
            "total_sessions": stats["total_sessions"],
            "fork_events": stats["fork_events"],
            "idle_sessions": stats["idle_sessions"],
            "peak_concurrent_decoding": stats["peak_concurrent_decoding"],
            "total_tokens": total_tokens,
            "inherited_tokens": stats["inherited_tokens"],
            "fresh_tokens": stats["fresh_tokens"],
            "prefix_shared_fraction": round(prefix_shared_fraction, 4),
            "kv_bytes_per_token_source": "fuels/glm-5.3-flash/attention.yaml",
            "kv_bytes_per_token": kv_bpt,
            "kda_fixed_bytes_per_stream": kda_fixed,
            "kv_bytes_without_sharing": without_sharing,
            "kv_bytes_with_sharing": with_sharing,
            "kv_sharing_ratio": round(with_sharing / without_sharing, 4)
            if without_sharing else 0.0,
        }
        out.write(json.dumps(summary, sort_keys=True) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()


# -------------------------------------------------------------- validate
def cmd_validate(args):
    src = sys.stdin if args.trace == "-" else open(args.trace)
    started = {}  # session -> t of session_start
    errors = []
    last_t = -1
    line_no = 0
    try:
        for line in src:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            kind = ev.get("event")
            if kind == "summary":
                continue
            t = ev.get("t")
            if t is None or t < last_t:
                errors.append(f"line {line_no}: non-monotonic or missing t={t}")
            else:
                last_t = t

            if kind == "session_start":
                sid = ev["session"]
                if sid in started:
                    errors.append(f"line {line_no}: duplicate session_start for {sid}")
                parent = ev.get("parent_id")
                if parent is not None and parent not in started:
                    errors.append(
                        f"line {line_no}: fork references unknown/not-yet-live "
                        f"parent {parent} for session {sid}"
                    )
                started[sid] = t
            elif kind in ("prefill", "decode", "idle", "resume"):
                sid = ev.get("session")
                if sid not in started:
                    errors.append(
                        f"line {line_no}: {kind} for session {sid} before its "
                        f"session_start"
                    )
                elif started[sid] > t:
                    errors.append(
                        f"line {line_no}: {kind} for session {sid} at t={t} "
                        f"precedes its session_start at t={started[sid]}"
                    )
            else:
                errors.append(f"line {line_no}: unknown event type {kind!r}")
    finally:
        if src is not sys.stdin:
            src.close()

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        print(f"validate: {len(errors)} error(s) over {line_no} lines", file=sys.stderr)
        return 1
    print(f"validate: OK, {line_no} lines, {len(started)} sessions, causality holds")
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concurrency", type=int, default=16,
                   help="peak concurrent decoding streams admitted per tick")
    p.add_argument("--depth", type=int, default=2,
                   help="max fork depth (root is depth 0)")
    p.add_argument("--prefix-tokens", type=int, default=8192,
                   help="root shared system+context prefix length")
    p.add_argument("--tool-burst-tokens", type=float, default=300,
                   help="mean mid-stream tool-result prefill size")
    p.add_argument("--duration-steps", type=int, default=2000,
                   help="number of scheduler ticks to simulate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="-", help="output path, - for stdout")
    p.add_argument("--attention-yaml", default=str(DEFAULT_ATTENTION_YAML),
                   help="path to the fuel's attention.yaml for KV byte costs")
    p.add_argument("--validate", action="store_true",
                   help="replay --trace and check causality instead of generating")
    p.add_argument("--trace", default="-",
                   help="trace file to replay in --validate mode, - for stdin")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.validate:
        return cmd_validate(args)
    cmd_generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
