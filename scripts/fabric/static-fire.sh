#!/usr/bin/env bash
# Static fire: light both boosters expert-parallel and hold the result against
# one booster's. This is the integration test for the two-booster arrangement.
#
#   scripts/fabric/static-fire.sh [--refresh-reference] [--sweep 1,8]
#                                 [--expert-cache-gib 20] [--expert-histogram-out PATH]
#                                 [--expert-histogram-in PATH] [--overlap 0|1] [--cuda-graph 0|1]
#
# rank 0 runs here, rank 1 on the peer over ssh. Both nodes have the same $HOME
# and the same NIM snapshot, so the binary and the fuel description are copied
# to the same absolute paths.
#
# Exits 77 (ctest skip) when the peer or the checkpoint is not on this node,
# matching the real-artifact tests in engines/glm5-moe-nvfp4-2b/tests/.
set -uo pipefail

PEER=${ROCKET_PEER:-192.168.100.11}
HEAD=${ROCKET_HEAD:-192.168.100.10}
PORT=${ROCKET_PORT:-18779}
SWEEP=1,8
CACHE_GIB=${ROCKET_EXPERT_CACHE_GIB:-20}
REFRESH=0
SKIP_PARITY=0
HIST_OUT=
HIST_IN=
OVERLAP=0
GRAPH=0
REF=${ROCKET_REFERENCE_TOKENS:-/tmp/rocket-static-fire-reference.txt}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh-reference) REFRESH=1; shift ;;
    --skip-parity) SKIP_PARITY=1; shift ;;
    --sweep) SWEEP=$2; shift 2 ;;
    --peer) PEER=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --expert-cache-gib) CACHE_GIB=$2; shift 2 ;;
    # Dumps the 288 routed-expert firing counts of the parity decode from
    # rank 0. Both ranks run the same router on the same input, so one rank's
    # histogram already covers every expert, owned and foreign alike.
    --expert-histogram-out) HIST_OUT=$2; shift 2 ;;
    # Partitions the experts by those counts instead of by id. The file is
    # copied to the peer below, so one path names the same content on both and
    # both ranks pack the same partition without exchanging it.
    --expert-histogram-in) HIST_IN=$2; shift 2 ;;
    # Overlap the routed-expert row exchange with the shared expert's FFN.
    --overlap) OVERLAP=$2; shift 2 ;;
    # CUDA graph capture of the decode step's non-routing work.
    --cuda-graph) GRAPH=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bin="$repo/engines/glm5-moe-nvfp4-2b/build/rocket-static-fire"
yaml="$repo/fuels/glm-5.3-flash/attention.yaml"

if [[ ! -x $bin ]]; then
  echo "rocket-static-fire is not built" >&2
  exit 77
fi
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -n "$PEER" true 2>/dev/null; then
  echo "peer $PEER not reachable" >&2
  exit 77
fi

ssh -o BatchMode=yes -n "$PEER" "mkdir -p $(dirname "$bin") $(dirname "$yaml")"
rsync -a "$bin" "$PEER:$bin"
rsync -a "$yaml" "$PEER:$yaml"
if [[ -n $HIST_IN ]]; then
  ssh -o BatchMode=yes -n "$PEER" "mkdir -p $(dirname "$HIST_IN")"
  rsync -a "$HIST_IN" "$PEER:$HIST_IN"
fi

if [[ $REFRESH -eq 1 || ! -s $REF ]]; then
  echo "== single-booster reference (all 288 experts) =="
  "$bin" --mode single --tokens-file "$REF" --sweep "$SWEEP" --expert-cache-gib "$CACHE_GIB" --skip-parity "$SKIP_PARITY"
  rc=$?
  [[ $rc -eq 77 ]] && exit 77
  [[ $rc -ne 0 ]] && { echo "reference run failed" >&2; exit "$rc"; }
fi

echo
echo "== expert-parallel, rank 0 here, rank 1 on $PEER =="
# Both ranks check the same tokens, so a mismatch is caught on whichever rank
# sees it. Copied before rank 1 starts, not after.
rsync -a "$REF" "$PEER:$REF"
common_args="--sweep $SWEEP --expert-cache-gib $CACHE_GIB --skip-parity $SKIP_PARITY"
common_args="$common_args --overlap $OVERLAP --cuda-graph $GRAPH"
[[ -n $HIST_IN ]] && common_args="$common_args --expert-histogram-in $HIST_IN"
ssh -o BatchMode=yes -n "$PEER" \
  "$bin --mode parallel --rank 1 --host $HEAD --port $PORT --tokens-file $REF \
        $common_args" >/tmp/rocket-static-fire-rank1.log 2>&1 &
peer_pid=$!
trap 'kill $peer_pid 2>/dev/null || true' EXIT

hist_args=()
[[ -n $HIST_OUT ]] && hist_args=(--expert-histogram-out "$HIST_OUT")
# shellcheck disable=SC2086
"$bin" --mode parallel --rank 0 --host "$HEAD" --port "$PORT" --tokens-file "$REF" \
       $common_args "${hist_args[@]}"
rc=$?
wait $peer_pid
peer_rc=$?
echo
echo "--- rank 1 ---"
cat /tmp/rocket-static-fire-rank1.log

if [[ $rc -ne 0 || $peer_rc -ne 0 ]]; then
  echo "static fire FAILED (rank0=$rc rank1=$peer_rc)" >&2
  exit 1
fi
exit 0
