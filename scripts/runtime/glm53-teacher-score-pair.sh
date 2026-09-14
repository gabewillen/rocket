#!/usr/bin/env bash
# Two-rank C8 teacher-forced accuracy and throughput scorer.
set -euo pipefail

PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
HEAD=${HEAD:-${ROCKET_HEAD:-192.168.100.10}}
PORT=${PORT:-${ROCKET_PORT:-19190}}
BATCH=${BATCH:-8}
EXPERT_CACHE_GIB=${EXPERT_CACHE_GIB:-90}
MAX_TOKENS=${MAX_TOKENS:-128}
MAX_SCORE_TOKENS=${MAX_SCORE_TOKENS:-64}
SCORE_CHUNK=${SCORE_CHUNK:-7}
TEXT_FILE=${TEXT_FILE:-}
TOKEN_FILE=${TOKEN_FILE:-}
OUTPUT=${OUTPUT:?set OUTPUT}
LOGITS=${LOGITS:-}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN="$REPO/engines/glm5-moe-nvfp4-2b/build/rocket-teacher-score"

[[ -x $BIN ]] || { echo "missing $BIN; build rocket-teacher-score" >&2; exit 2; }
[[ -n $TEXT_FILE && -n $TOKEN_FILE ]] && { echo 'choose TEXT_FILE or TOKEN_FILE' >&2; exit 2; }
INPUT=${TEXT_FILE:-$TOKEN_FILE}
[[ -n $INPUT ]] || { echo 'set TEXT_FILE or TOKEN_FILE' >&2; exit 2; }

export ROCKET_FUEL_NVFP4_DIR=${ROCKET_FUEL_NVFP4_DIR:-$HOME/.cache/rocket-fuels/glm-5.3-flash-nvfp4}
export ROCKET_PACKED_WEIGHTS=${ROCKET_PACKED_WEIGHTS:-$ROCKET_FUEL_NVFP4_DIR/resident-direct.bin}
export ROCKET_EXPERT_SLAB_DIR=${ROCKET_EXPERT_SLAB_DIR:-$ROCKET_FUEL_NVFP4_DIR/expert-slabs-v1}
export ROCKET_RESIDENT_DIRECT=${ROCKET_RESIDENT_DIRECT:-1}
export ROCKET_KDA_CUBLAS=${ROCKET_KDA_CUBLAS:-1}
export ROCKET_CUBLAS_ALL=${ROCKET_CUBLAS_ALL:-1}
export ROCKET_RDMA_WAIT_S=${ROCKET_RDMA_WAIT_S:-600}
export GLM53_ALLOW_LAUNCH=${GLM53_ALLOW_LAUNCH:-1}

rsync -a "$BIN" "$PEER:$BIN"
rsync -a "$INPUT" "$PEER:$INPUT"
common=(--output "$OUTPUT" --batch "$BATCH" --expert-cache-gib "$EXPERT_CACHE_GIB"
        --max-tokens "$MAX_TOKENS" --max-score-tokens "$MAX_SCORE_TOKENS"
        --score-chunk "$SCORE_CHUNK")
[[ -n $TEXT_FILE ]] && common+=(--text-file "$TEXT_FILE") || common+=(--token-file "$TOKEN_FILE")
[[ -n $LOGITS ]] && common+=(--logits "$LOGITS")

remote=(env)
for v in ROCKET_FUEL_NVFP4_DIR ROCKET_PACKED_WEIGHTS ROCKET_EXPERT_SLAB_DIR ROCKET_RESIDENT_DIRECT \
         ROCKET_KDA_CUBLAS ROCKET_CUBLAS_ALL ROCKET_RDMA_WAIT_S GLM53_ALLOW_LAUNCH; do
  remote+=("$v=${!v}")
done
remote+=("$BIN" --rank 1 --host "$HEAD" --port "$PORT" "${common[@]}")
printf -v remote_cmd '%q ' "${remote[@]}"
ssh -o BatchMode=yes -n "$PEER" "$remote_cmd" >/tmp/teacher-score-rank1.log 2>&1 &
peer_pid=$!
trap 'kill $peer_pid 2>/dev/null || true' EXIT
"$BIN" --rank 0 --host "$HEAD" --port "$PORT" "${common[@]}"
wait "$peer_pid"
