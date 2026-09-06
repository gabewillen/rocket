#!/usr/bin/env bash
# Transport measurement for the two-booster fabric (engines/glm5-moe-nvfp4-2b/
# src/fabric/fabric.h), both ranks, at the message sizes the expert-parallel
# decode step uses.
#
#   scripts/fabric/fabric-microbench.sh [--rails 1|2] [--split-min <bytes>] \
#       [--peer <host>] [--port <port>]
#
# rank 0 runs here, rank 1 runs on the peer over ssh. The binary is copied to
# the same absolute path on the peer, which is the same $HOME on both boosters.
set -euo pipefail

RAILS=2
SPLIT_MIN=65536
PEER=${ROCKET_PEER:-192.168.100.11}
HEAD=${ROCKET_HEAD:-192.168.100.10}
PORT=${ROCKET_PORT:-18777}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rails) RAILS=$2; shift 2 ;;
    --split-min) SPLIT_MIN=$2; shift 2 ;;
    --peer) PEER=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bin="$repo/engines/glm5-moe-nvfp4-2b/build/rocket-fabric-bench"
[[ -x $bin ]] || { echo "build rocket-fabric-bench first" >&2; exit 2; }

ssh -o BatchMode=yes "$PEER" "mkdir -p $(dirname "$bin")"
rsync -a "$bin" "$PEER:$bin"

ssh -o BatchMode=yes -n "$PEER" \
  "$bin --rank 1 --host $HEAD --port $PORT --rails $RAILS --split-min $SPLIT_MIN" \
  >/tmp/rocket-fabric-rank1.log 2>&1 &
peer_pid=$!
trap 'kill $peer_pid 2>/dev/null || true' EXIT

"$bin" --rank 0 --host "$HEAD" --port "$PORT" --rails "$RAILS" --split-min "$SPLIT_MIN"
wait $peer_pid || true
echo "--- rank 1 ---"
cat /tmp/rocket-fabric-rank1.log
