#!/usr/bin/env bash
# Two-rank prefix-cache crossover measurement. Run one length at a time so one
# GPU workload exists on the pair. A cold pass checkpoints the exact prompt;
# restore passes start in new processes and validate durable manifest replay.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PEER=${PEER:-192.168.100.11}
PREFIX_TOKENS=${PREFIX_TOKENS:-2048}
REPEATS=${REPEATS:-3}
CACHE_BYTES=${CACHE_BYTES:-128GiB}
CACHE_STAGING=${CACHE_STAGING:-128MiB}
CACHE_QUEUE=${CACHE_QUEUE:-4}
BATCH=${BATCH:-8}
SPEC=${SPEC:-8}
NEW_TOKENS=${NEW_TOKENS:-32}
PROMPT_FILE=${PROMPT_FILE:-/home/glwillen/.cache/rocket-prefix-prompt.txt}
CACHE_DIR=${CACHE_DIR:-/home/glwillen/.cache/rocket-kv-offload-crossover-$PREFIX_TOKENS}
OUT=${OUT:-/tmp/glm53-prefix-$PREFIX_TOKENS}
PORT_BASE=${PORT_BASE:-18900}
SKIP_BASELINE=${SKIP_BASELINE:-0}

export ROCKET_FUEL_NVFP4_DIR=${ROCKET_FUEL_NVFP4_DIR:-/home/glwillen/.cache/rocket-fuels/glm-5.3-flash-nvfp4}
export ROCKET_PACKED_WEIGHTS=${ROCKET_PACKED_WEIGHTS:-$ROCKET_FUEL_NVFP4_DIR/resident-direct.bin}
export ROCKET_EXPERT_SLAB_DIR=${ROCKET_EXPERT_SLAB_DIR:-$ROCKET_FUEL_NVFP4_DIR/expert-slabs-v1}
export ROCKET_RESIDENT_DIRECT=1
export ROCKET_DFLASH2_DIR=${ROCKET_DFLASH2_DIR:-/home/glwillen/.cache/rocket-glm53-exl3-host/models/glm-5.3-flash-dflash2}
export ROCKET_FP8_ATTN_DIR=${ROCKET_FP8_ATTN_DIR:-/home/glwillen/.cache/rocket-fuels/overlays}
export ROCKET_KDA_QKV_FP8_DIR=${ROCKET_KDA_QKV_FP8_DIR:-/home/glwillen/.cache/rocket-fuels/overlays/kda-qkv-fp8}
export ROCKET_KDA_CUBLAS=1 ROCKET_CUBLAS_ALL=1 ROCKET_RDMA_WAIT_S=${ROCKET_RDMA_WAIT_S:-600} GLM53_ALLOW_LAUNCH=1
export ROCKET_TOKEN_IDS=1
export BATCH SPEC MAX_TOKENS=${MAX_TOKENS:-$((PREFIX_TOKENS + 256))} TOKENS=$NEW_TOKENS
# The committed expert slabs contain all 6048 rank-owned experts and require
# the production 90 GiB slot geometry.
export EXPERT_CACHE_GIB=${EXPERT_CACHE_GIB:-90} PRELOAD_OWNED=${PRELOAD_OWNED:-1}
export PREFILL_CHUNK=${PREFILL_CHUNK:-32} PREFILL_TAIL=${PREFILL_TAIL:-32}
"$REPO/scripts/runtime/glm53-prefix-cache-bench.py" --make-prompt "$PROMPT_FILE" \
  --minimum-tokens "$PREFIX_TOKENS" --tokenizer "$ROCKET_FUEL_NVFP4_DIR/tokenizer.json" >/dev/null
ssh -o BatchMode=yes "$PEER" "mkdir -p $(printf %q "$(dirname "$PROMPT_FILE")")"
rsync -a "$PROMPT_FILE" "$PEER:$PROMPT_FILE"
export PROMPT_FILE PROMPT_TOKEN_LIMIT=$PREFIX_TOKENS

unset PREFIX_CACHE_DIR PREFIX_CACHE_BYTES PREFIX_CACHE_STAGING_BYTES PREFIX_CACHE_QUEUE_DEPTH
if [[ $SKIP_BASELINE != 1 ]]; then
  PORT=$PORT_BASE "$REPO/scripts/runtime/glm53-nvfp4-spec-bench.sh" >"$OUT-baseline.log" 2>&1
fi
export PREFIX_CACHE_DIR=$CACHE_DIR PREFIX_CACHE_BYTES=$CACHE_BYTES
export PREFIX_CACHE_STAGING_BYTES=$CACHE_STAGING PREFIX_CACHE_QUEUE_DEPTH=$CACHE_QUEUE
rm -rf "$CACHE_DIR"
ssh -o BatchMode=yes "$PEER" "rm -rf $(printf %q "$CACHE_DIR")"
PORT=$((PORT_BASE + 1)) "$REPO/scripts/runtime/glm53-nvfp4-spec-bench.sh" >"$OUT-cold.log" 2>&1
for ((i=1; i<=REPEATS; ++i)); do
  PORT=$((PORT_BASE + 1 + i)) "$REPO/scripts/runtime/glm53-nvfp4-spec-bench.sh" \
    >"$OUT-restore-$i.log" 2>&1
done
logs=("$OUT-cold.log" "$OUT"-restore-*.log)
[[ $SKIP_BASELINE == 1 ]] || logs=("$OUT-baseline.log" "${logs[@]}")
"$REPO/scripts/runtime/glm53-prefix-cache-bench.py" --summarize "${logs[@]}"
