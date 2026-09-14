#!/usr/bin/env bash
# NVFP4 CUDA engine spec-decode benchmark on the two-node pair.
# rank 0 here, rank 1 on the peer over ssh. Both ranks run the same decode
# loop; rank 0 prints the timing.
#
#   scripts/runtime/glm53-nvfp4-spec-bench.sh --batch 8 --spec 4 --tokens 200
set -uo pipefail

PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
HEAD=${HEAD:-${ROCKET_HEAD:-192.168.100.10}}
PORT=${PORT:-${ROCKET_PORT:-18780}}
BATCH=${BATCH:-8}
SPEC=${SPEC:-4}
TOKENS=${TOKENS:-200}
EXPERT_CACHE_GIB=${EXPERT_CACHE_GIB:-92}
MAX_TOKENS=${MAX_TOKENS:-4096}
PREFILL_CHUNK=${PREFILL_CHUNK:-32}
PREFILL_TAIL=${PREFILL_TAIL:-32}
PRELOAD_OWNED=${PRELOAD_OWNED:-1}
PROMPT=${PROMPT:-"Count from 1 to 50: 1, 2, 3,"}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_TOKEN_LIMIT=${PROMPT_TOKEN_LIMIT:-0}
PREFIX_CACHE_DIR=${PREFIX_CACHE_DIR:-}
PREFIX_CACHE_BYTES=${PREFIX_CACHE_BYTES:-128GiB}
PREFIX_CACHE_STAGING_BYTES=${PREFIX_CACHE_STAGING_BYTES:-128MiB}
PREFIX_CACHE_QUEUE_DEPTH=${PREFIX_CACHE_QUEUE_DEPTH:-4}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN="$REPO/engines/glm5-moe-nvfp4-2b/build/rocket-decode"

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
  common+=(--prefix-cache-dir "$PREFIX_CACHE_DIR" --prefix-cache-bytes "$PREFIX_CACHE_BYTES"
           --prefix-cache-staging-bytes "$PREFIX_CACHE_STAGING_BYTES"
           --prefix-cache-queue-depth "$PREFIX_CACHE_QUEUE_DEPTH")
  mkdir -p "$PREFIX_CACHE_DIR"
  ssh -o BatchMode=yes "$PEER" "mkdir -p $(printf %q "$PREFIX_CACHE_DIR")"
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
