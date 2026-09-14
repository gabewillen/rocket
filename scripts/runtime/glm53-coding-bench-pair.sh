#!/usr/bin/env bash
# Run one distinct-prompt coding workload phase on the two-node GLM-5.3 pair.
set -euo pipefail
PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
HEAD=${HEAD:-${ROCKET_HEAD:-192.168.100.10}}
PORT=${PORT:-${ROCKET_PORT:-18782}}
BATCH=${BATCH:-8}
SPEC=${SPEC:-7}
TOKENS=${TOKENS:-512}
EXPERT_CACHE_GIB=${EXPERT_CACHE_GIB:-90}
MAX_TOKENS=${MAX_TOKENS:-65536}
PREFILL_CHUNK=${PREFILL_CHUNK:-32}
PREFILL_TAIL=${PREFILL_TAIL:-32}
PRELOAD_OWNED=${PRELOAD_OWNED:-1}
PROMPT_LIST=${PROMPT_LIST:?set PROMPT_LIST to a file containing one prompt path per stream}
RESULT_JSON=${RESULT_JSON:-/tmp/glm53-coding-result.json}
PROMPT_TOKEN_LIMIT=${PROMPT_TOKEN_LIMIT:-0}
PREFIX_CACHE_DIR=${PREFIX_CACHE_DIR-$HOME/.cache/rocket-prefix-cache/glm53-coding}
PREFIX_CACHE_BYTES=${PREFIX_CACHE_BYTES:-auto}
PREFIX_CACHE_STAGING_BYTES=${PREFIX_CACHE_STAGING_BYTES:-128MiB}
PREFIX_CACHE_QUEUE_DEPTH=${PREFIX_CACHE_QUEUE_DEPTH:-4}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN="$REPO/engines/glm5-moe-nvfp4-2b/build/rocket-coding-bench"

export ROCKET_FUEL_NVFP4_DIR=${ROCKET_FUEL_NVFP4_DIR:-$HOME/.cache/rocket-fuels/glm-5.3-flash-nvfp4}
export ROCKET_PACKED_WEIGHTS=${ROCKET_PACKED_WEIGHTS:-$ROCKET_FUEL_NVFP4_DIR/resident-direct.bin}
export ROCKET_EXPERT_SLAB_DIR=${ROCKET_EXPERT_SLAB_DIR:-$ROCKET_FUEL_NVFP4_DIR/expert-slabs-v1}
export ROCKET_DFLASH2_DIR=${ROCKET_DFLASH2_DIR:-$HOME/.cache/rocket-glm53-exl3-host/models/glm-5.3-flash-dflash2}
export ROCKET_RESIDENT_DIRECT=${ROCKET_RESIDENT_DIRECT:-1}
export ROCKET_KDA_CUBLAS=${ROCKET_KDA_CUBLAS:-1}
export ROCKET_CUBLAS_ALL=${ROCKET_CUBLAS_ALL:-1}
export ROCKET_RDMA_WAIT_S=${ROCKET_RDMA_WAIT_S:-600}
export GLM53_ALLOW_LAUNCH=${GLM53_ALLOW_LAUNCH:-1}

[[ -x $BIN ]] || { echo "missing $BIN" >&2; exit 2; }
mapfile -t prompts < "$PROMPT_LIST"
(( ${#prompts[@]} == BATCH )) || { echo "prompt count ${#prompts[@]} != batch $BATCH" >&2; exit 2; }
rsync -a "$BIN" "$PEER:$BIN"
{
  dirname "$PROMPT_LIST"
  for p in "${prompts[@]}"; do dirname "$p"; done
} | sort -u | while read -r dir; do
  quoted_dir=$(printf %q "$dir")
  ssh -o BatchMode=yes "$PEER" "mkdir -p $quoted_dir"
done
rsync -a "$PROMPT_LIST" "$PEER:$PROMPT_LIST"
for p in "${prompts[@]}"; do rsync -a "$p" "$PEER:$p"; done

cache_free_bytes() { df -B1 --output=avail "$1" | awk 'NR==2 {print $1}'; }
common=(--prompt-list "$PROMPT_LIST" --tokens "$TOKENS" --batch "$BATCH" --spec "$SPEC"
  --expert-cache-gib "$EXPERT_CACHE_GIB" --max-tokens "$MAX_TOKENS"
  --prefill-chunk "$PREFILL_CHUNK" --prefill-tail "$PREFILL_TAIL" --preload-owned "$PRELOAD_OWNED")
(( PROMPT_TOKEN_LIMIT > 0 )) && common+=(--prompt-token-limit "$PROMPT_TOKEN_LIMIT")
if [[ -n $PREFIX_CACHE_DIR ]]; then
  mkdir -p "$PREFIX_CACHE_DIR"
  quoted=$(printf %q "$PREFIX_CACHE_DIR")
  ssh -o BatchMode=yes "$PEER" "mkdir -p $quoted"
  if [[ $PREFIX_CACHE_BYTES == auto ]]; then
    lf=$(cache_free_bytes "$PREFIX_CACHE_DIR")
    rf=$(ssh -o BatchMode=yes "$PEER" "df -B1 --output=avail $quoted | awk 'NR==2 {print \\$1}'")
    (( rf < lf )) && lf=$rf
    PREFIX_CACHE_BYTES=$((lf / 2))
    cap=$((lf - 64 * 1024 * 1024 * 1024))
    (( cap <= 0 )) && { echo 'prefix cache needs >64 GiB free' >&2; exit 2; }
    (( cap < PREFIX_CACHE_BYTES )) && PREFIX_CACHE_BYTES=$cap
    PREFIX_CACHE_BYTES=$((PREFIX_CACHE_BYTES / 65536 * 65536))
  fi
  common+=(--prefix-cache-dir "$PREFIX_CACHE_DIR" --prefix-cache-bytes "$PREFIX_CACHE_BYTES"
    --prefix-cache-staging-bytes "$PREFIX_CACHE_STAGING_BYTES" --prefix-cache-queue-depth "$PREFIX_CACHE_QUEUE_DEPTH")
fi

remote=(env)
for v in ROCKET_FUEL_NVFP4_DIR ROCKET_PACKED_WEIGHTS ROCKET_EXPERT_SLAB_DIR ROCKET_RESIDENT_DIRECT ROCKET_RDMA_WAIT_S ROCKET_KDA_CUBLAS ROCKET_CUBLAS_ALL ROCKET_DFLASH2_DIR GLM53_ALLOW_LAUNCH; do
  [[ -n ${!v:-} ]] && remote+=("$v=${!v}")
done
remote+=("$BIN" --rank 1 --host "$HEAD" --port "$PORT" --result-json /tmp/glm53-coding-rank1.json "${common[@]}")
printf -v remote_cmd '%q ' "${remote[@]}"
ssh -o BatchMode=yes -n "$PEER" "$remote_cmd" >/tmp/glm53-coding-rank1.log 2>&1 &
peer_pid=$!
trap 'kill $peer_pid 2>/dev/null || true' EXIT
"$BIN" --rank 0 --host "$HEAD" --port "$PORT" --result-json "$RESULT_JSON" "${common[@]}"
wait "$peer_pid"
