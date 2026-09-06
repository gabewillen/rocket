#!/usr/bin/env bash
# Streams per booster on the forked-prefix agent workload, from the engine's
# own page tables.
#
# Generates the c32 trace at three root-prefix lengths and replays each one
# through PagePool / PrefixTree / KvCache, reporting peak resident KV with
# and without page sharing. The 212000 row is the one comparable to
# blog/posts/cache/2026-09-06-indexer-caches-gates-too/: its sessions sit at
# 262k-scale contexts, so its unshared column reproduces that entry's count.
#
#   scripts/cache/kv-capacity-replay.sh [outdir]
#
# Cites: scripts/scheduler/agent-workload.py
#        engines/glm5-moe-nvfp4-2b/bench/kv_capacity_replay.cc
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE="$ROOT/engines/glm5-moe-nvfp4-2b"
BIN="$ENGINE/build/bench/bench-kv-capacity-replay"
OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT"

if [[ ! -x "$BIN" ]]; then
  cmake -S "$ENGINE" -B "$ENGINE/build" >/dev/null
  cmake --build "$ENGINE/build" --target bench-kv-capacity-replay -j8 >/dev/null
fi

# bf16 is attention.yaml's figure and is what the published unshared stream
# count used; fp32 is what this engine actually allocates (model.h).
for KDA in 76316672:bf16 147619840:fp32; do
  BYTES="${KDA%%:*}"; NAME="${KDA##*:}"
  echo "### KDA state $NAME ($BYTES B/stream)"
  printf '%12s %14s %16s %14s %12s\n' root_prefix shared_GiB_sess unshared_GiB_sess sessions_shared unshared
  for P in 8192 65536 212000; do
    TRACE="$OUT/c32-p$P.jsonl"
    [[ -f "$TRACE" ]] || python3 "$ROOT/scripts/scheduler/agent-workload.py" \
      --concurrency 32 --prefix-tokens "$P" --out "$TRACE"
    "$BIN" "$TRACE" --kda-bytes "$BYTES" | awk -v p="$P" '
      /bytes per session/ { sh=$4; un=$7 }
      /sessions held/     { shn=$3; unn=$5 }
      END { printf "%12s %14s %16s %14s %12s\n", p, sh, un, shn, unn }'
  done
done
echo "traces in $OUT"
