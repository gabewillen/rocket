#!/usr/bin/env bash
# NVFP4 CUDA engine spec-decode benchmark on the two-node pair.
# rank 0 here, rank 1 on the peer over ssh. Both ranks run the same decode
# loop; rank 0 prints the timing.
#
#   scripts/runtime/glm53-nvfp4-spec-bench.sh --batch 8 --spec 4 --tokens 200
set -euo pipefail

PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
HEAD=${HEAD:-${ROCKET_HEAD:-192.168.100.10}}
PORT=${PORT:-${ROCKET_PORT:-18780}}
BATCH=${BATCH:-8}
SPEC=${SPEC:-7}
TOKENS=${TOKENS:-200}
EXPERT_CACHE_GIB=${EXPERT_CACHE_GIB:-90}
MAX_TOKENS=${MAX_TOKENS:-4096}
PREFILL_CHUNK=${PREFILL_CHUNK:-32}
PREFILL_TAIL=${PREFILL_TAIL:-32}
PRELOAD_OWNED=${PRELOAD_OWNED:-1}
PROMPT=${PROMPT:-"Count from 1 to 50: 1, 2, 3,"}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_TOKEN_LIMIT=${PROMPT_TOKEN_LIMIT:-0}
PREFIX_CACHE_DIR=${PREFIX_CACHE_DIR-$HOME/.cache/rocket-prefix-cache/glm53}
PREFIX_CACHE_BYTES=${PREFIX_CACHE_BYTES:-auto}
PREFIX_CACHE_STAGING_BYTES=${PREFIX_CACHE_STAGING_BYTES:-128MiB}
PREFIX_CACHE_QUEUE_DEPTH=${PREFIX_CACHE_QUEUE_DEPTH:-4}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN="$REPO/engines/glm5-moe-nvfp4-2b/build/rocket-decode"

# Production defaults. Explicit caller values still win.
export ROCKET_FUEL_NVFP4_DIR=${ROCKET_FUEL_NVFP4_DIR:-$HOME/.cache/rocket-fuels/glm-5.3-flash-nvfp4}
export ROCKET_PACKED_WEIGHTS=${ROCKET_PACKED_WEIGHTS:-$ROCKET_FUEL_NVFP4_DIR/resident-direct.bin}
export ROCKET_EXPERT_SLAB_DIR=${ROCKET_EXPERT_SLAB_DIR:-$ROCKET_FUEL_NVFP4_DIR/expert-slabs-v1}
export ROCKET_DFLASH2_DIR=${ROCKET_DFLASH2_DIR:-$HOME/.cache/rocket-glm53-exl3-host/models/glm-5.3-flash-dflash2}
export ROCKET_FP4_SHARED_DIR=${ROCKET_FP4_SHARED_DIR:-$HOME/.cache/rocket-fuels/overlays}
export ROCKET_RESIDENT_DIRECT=${ROCKET_RESIDENT_DIRECT:-1}
export ROCKET_KDA_CUBLAS=${ROCKET_KDA_CUBLAS:-1}
export ROCKET_CUBLAS_ALL=${ROCKET_CUBLAS_ALL:-1}
export ROCKET_RDMA_WAIT_S=${ROCKET_RDMA_WAIT_S:-600}
export GLM53_ALLOW_LAUNCH=${GLM53_ALLOW_LAUNCH:-1}

cache_free_bytes() {
  df -B1 --output=avail "$1" | awk 'NR==2 {print $1}'
}

rsync -a "$BIN" "$PEER:$BIN" 2>/dev/null || true

common=(
  --prompt "$PROMPT" --tokens "$TOKENS" --batch "$BATCH" --spec "$SPEC"
  --expert-cache-gib "$EXPERT_CACHE_GIB" --max-tokens "$MAX_TOKENS" --prefill-chunk "$PREFILL_CHUNK" --prefill-tail "$PREFILL_TAIL" --preload-owned "$PRELOAD_OWNED"
)
if [[ -n $PROMPT_FILE ]]; then
  rsync -a "$PROMPT_FILE" "$PEER:$PROMPT_FILE"
  common+=(--prompt-file "$PROMPT_FILE")
fi
if (( PROMPT_TOKEN_LIMIT > 0 )); then common+=(--prompt-token-limit "$PROMPT_TOKEN_LIMIT"); fi
if [[ -n $PREFIX_CACHE_DIR ]]; then
  mkdir -p "$PREFIX_CACHE_DIR"
  quoted_cache_dir=$(printf %q "$PREFIX_CACHE_DIR")
  ssh -o BatchMode=yes "$PEER" "mkdir -p $quoted_cache_dir"
  if [[ $PREFIX_CACHE_BYTES == auto ]]; then
    local_free=$(cache_free_bytes "$PREFIX_CACHE_DIR")
    remote_free=$(ssh -o BatchMode=yes "$PEER" \
      "df -B1 --output=avail $quoted_cache_dir | awk 'NR==2 {print \$1}'")
    free_bytes=$local_free
    (( remote_free < free_bytes )) && free_bytes=$remote_free
    half_free=$((free_bytes / 2))
    headroom_cap=$((free_bytes - 64 * 1024 * 1024 * 1024))
    (( headroom_cap <= 0 )) && { echo 'prefix cache needs more than 64 GiB free on both nodes' >&2; exit 2; }
    PREFIX_CACHE_BYTES=$half_free
    (( headroom_cap < PREFIX_CACHE_BYTES )) && PREFIX_CACHE_BYTES=$headroom_cap
    PREFIX_CACHE_BYTES=$((PREFIX_CACHE_BYTES / 65536 * 65536))
    printf 'prefix cache quota: %s bytes (up to 50%% of pair-minimum free space)\n' "$PREFIX_CACHE_BYTES"
  fi
  common+=(--prefix-cache-dir "$PREFIX_CACHE_DIR" --prefix-cache-bytes "$PREFIX_CACHE_BYTES"
           --prefix-cache-staging-bytes "$PREFIX_CACHE_STAGING_BYTES"
           --prefix-cache-queue-depth "$PREFIX_CACHE_QUEUE_DEPTH")
fi
if [[ -n ${DRAFT_FILE:-} ]]; then
  rsync -a "$DRAFT_FILE" "$PEER:$DRAFT_FILE"
  common+=(--draft-file "$DRAFT_FILE")
fi
remote=(env)
for v in ROCKET_FUEL_NVFP4_DIR ROCKET_PACKED_WEIGHTS ROCKET_EXPERT_SLAB_DIR ROCKET_RESIDENT_DIRECT ROCKET_LOAD_PROFILE ROCKET_FP8_ATTN_DIR ROCKET_KDA_QKV_FP8_DIR ROCKET_KDA_QKV_NVFP4_DIR ROCKET_FP4_SHARED_DIR ROCKET_RDMA_WAIT_S \
         ROCKET_FABRIC_TRACE ROCKET_SPEC_TRACE ROCKET_SPEC_DEBUG ROCKET_MOE_TRACE ROCKET_DEBUG_MOE \
         ROCKET_DEBUG_KDA ROCKET_KDA_CUBLAS ROCKET_CUBLAS_ALL ROCKET_FP8_CUBLAS ROCKET_DFLASH2_DIR ROCKET_TOKEN_IDS ROCKET_LHC_CHECKSUM ROCKET_KV_CHECKSUM \
         GLM53_ALLOW_LAUNCH; do
  [[ -n ${!v:-} ]] && remote+=("$v=${!v}")
done
remote+=("$BIN" --rank 1 --host "$HEAD" --port "$PORT" "${common[@]}")
printf -v remote_cmd '%q ' "${remote[@]}"

ssh -o BatchMode=yes -n "$PEER" "$remote_cmd" >/tmp/spec-rank1.log 2>&1 &
peer_pid=$!
trap 'kill $peer_pid 2>/dev/null || true' EXIT

"$BIN" --rank 0 --host "$HEAD" --port "$PORT" "${common[@]}"
rc=$?
kill $peer_pid 2>/dev/null || true
exit $rc
