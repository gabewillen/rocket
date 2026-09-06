# agent-workload.py

Forked-prefix agent workload generator. Reproduces the load shape of
agent/subagent serving for the throughput sweep (c8..c64): a root session
holds a long shared prefix, planner sessions burst-fork 2-5 subagents that
inherit the prefix and diverge, tool results land as mid-stream prefill
bursts, and some sessions idle and resume.

## Model

- Root session prefills `--prefix-tokens` once (default 8192). This is the
  shared system+context prefix.
- A tick-based scheduler admits up to `--concurrency` active sessions per
  tick (round-robin by least-recently-served), each decoding 8-32 tokens.
- Any active, non-root session below `--depth` has a small per-tick chance
  of forking a burst of 2-5 children. Children inherit `fork_point_tokens`
  (parent's prefix + generated tokens at fork time, shared/cached, not
  reprefilled) plus a small 50-200 token divergence prefill (their own
  instructions). Children get shorter lifetimes than their parent's
  remaining ticks.
- Every 20-80 decoded tokens, a session gets a mid-stream tool-result
  prefill burst, sized around `--tool-burst-tokens` (Gaussian, sd = 0.3 *
  mean).
- ~20% of non-root sessions take one idle excursion (5-180s) and resume.

## Output

JSONL trace to stdout (or `--out FILE`): `session_start`, `prefill`,
`decode`, `idle`, `resume` events, each carrying `t` (tick) and a monotonic
`seq`. Terminated by one `summary` line with:

- `total_tokens`, `inherited_tokens`, `fresh_tokens`,
  `prefix_shared_fraction` (inherited / total, by token count)
- `peak_concurrent_decoding`
- `kv_bytes_per_token`, `kda_fixed_bytes_per_stream`: read from
  `fuels/glm-5.3-flash/attention.yaml` (`layers.mla_count`,
  `bytes.mla_kv_per_token_per_layer`, `bytes.indexer_key_per_token_per_layer`,
  `bytes.kda_state_per_stream.total`), not hardcoded
- `kv_bytes_without_sharing`: every session's full context billed
  independently
- `kv_bytes_with_sharing`: each session billed only for tokens past its
  fork point (KDA recurrent state is never shareable across forks, so it's
  charged per session either way)
- `kv_sharing_ratio` = with / without

## Defaults

`--concurrency 16 --depth 2 --prefix-tokens 8192 --tool-burst-tokens 300
--duration-steps 2000 --seed 0`

## Usage

```
python3 agent-workload.py --concurrency 16 --depth 2 --duration-steps 2000 > trace.jsonl
python3 agent-workload.py --validate --trace trace.jsonl
```

`--validate` replays a trace and checks causality: no decode/prefill/
idle/resume before its session's `session_start`, and every fork's
`parent_id` must already be live. Exits nonzero on violation.

Sweep across the goal's concurrency range:

```
for c in 8 16 32 64; do
  python3 agent-workload.py --concurrency $c --seed 0 --out /tmp/trace-$c.jsonl
done
```
