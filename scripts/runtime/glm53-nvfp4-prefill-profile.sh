#!/usr/bin/env bash
# Privileged, capture-range Nsight profile of chunked c8 prefill. Rank 1 is
# launched as the calling user so SSH credentials never cross the sudo boundary.
set -euo pipefail

PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
HEAD=${HEAD:-${ROCKET_HEAD:-192.168.100.10}}
PORT=${PORT:-18781}
BATCH=${BATCH:-8}
SPEC=${SPEC:-8}
TOKENS=${TOKENS:-8}
EXPERT_CACHE_GIB=${EXPERT_CACHE_GIB:-90}
MAX_TOKENS=${MAX_TOKENS:-512}
PREFILL_CHUNK=${PREFILL_CHUNK:-32}
PREFILL_TAIL=${PREFILL_TAIL:-32}
PROMPT=${PROMPT:-"$(printf 'The capital of France is Paris. %.0s' {1..24})"}
OUT=${OUT:-/tmp/glm53-prefill-direct}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BIN="$REPO/engines/glm5-moe-nvfp4-2b/build/rocket-decode"

: "${ROCKET_FUEL_NVFP4_DIR:=/home/glwillen/.cache/rocket-fuels/glm-5.3-flash-nvfp4}"
: "${ROCKET_PACKED_WEIGHTS:=$ROCKET_FUEL_NVFP4_DIR/resident-direct.bin}"
: "${ROCKET_EXPERT_SLAB_DIR:=$ROCKET_FUEL_NVFP4_DIR/expert-slabs-v1}"
: "${ROCKET_FP8_ATTN_DIR:=/home/glwillen/.cache/rocket-fuels/overlays/fp8-all}"
: "${ROCKET_DFLASH2_DIR:=/home/glwillen/.cache/rocket-glm53-exl3-host/models/glm-5.3-flash-dflash2}"

rsync -a "$BIN" "$PEER:$BIN"
common=(--prompt "$PROMPT" --tokens "$TOKENS" --batch "$BATCH" --spec "$SPEC"
  --expert-cache-gib "$EXPERT_CACHE_GIB" --max-tokens "$MAX_TOKENS"
  --prefill-chunk "$PREFILL_CHUNK" --prefill-tail "$PREFILL_TAIL" --preload-owned 1)
env_args=(
  ROCKET_FUEL_NVFP4_DIR="$ROCKET_FUEL_NVFP4_DIR"
  ROCKET_PACKED_WEIGHTS="$ROCKET_PACKED_WEIGHTS"
  ROCKET_EXPERT_SLAB_DIR="$ROCKET_EXPERT_SLAB_DIR"
  ROCKET_RESIDENT_DIRECT=1
  ROCKET_FP8_ATTN_DIR="$ROCKET_FP8_ATTN_DIR"
  ROCKET_DFLASH2_DIR="$ROCKET_DFLASH2_DIR"
  ROCKET_KDA_CUBLAS=1 ROCKET_CUBLAS_ALL=1 ROCKET_PROFILE_PREFILL=1
  ROCKET_RDMA_WAIT_S=120 GLM53_ALLOW_LAUNCH=1)
remote=(env "${env_args[@]}" "$BIN" --rank 1 --host "$HEAD" --port "$PORT" "${common[@]}")
printf -v remote_cmd '%q ' "${remote[@]}"
ssh -o BatchMode=yes -n "$PEER" "$remote_cmd" >"${OUT}-rank1.log" 2>&1 &
peer_pid=$!
trap 'kill "$peer_pid" 2>/dev/null || true' EXIT

sudo env "${env_args[@]}" /usr/local/cuda/bin/nsys profile --force-overwrite=true \
  --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-devices=all --gpu-metrics-frequency=10000 -o "$OUT" \
  "$BIN" --rank 0 --host "$HEAD" --port "$PORT" "${common[@]}"
