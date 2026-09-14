#!/usr/bin/env bash
# Measure two-node NVML GPU power during the steady decode section.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
TOKENS=${TOKENS:-512}
PORT=${PORT:-19191}
SAMPLE_MS=${SAMPLE_MS:-200}
LOG=${LOG:-/tmp/glm53-power-bench.log}
LOCAL_POWER=${LOCAL_POWER:-/tmp/glm53-power-local.csv}
PEER_POWER=${PEER_POWER:-/tmp/glm53-power-peer.csv}

rm -f "$LOG" "$LOCAL_POWER" "$PEER_POWER"
TOKENS=$TOKENS PORT=$PORT stdbuf -oL -eL \
  "$REPO/scripts/runtime/glm53-nvfp4-spec-bench.sh" >"$LOG" 2>&1 &
bench_pid=$!
cleanup() {
  kill "${local_power_pid:-}" "${peer_power_pid:-}" "$bench_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 1800); do
  grep -q '^prompt ' "$LOG" && break
  kill -0 "$bench_pid" 2>/dev/null || { wait "$bench_pid"; exit $?; }
  sleep 0.1
done
grep -q '^prompt ' "$LOG" || { echo 'prompt marker did not appear' >&2; exit 1; }
# The production 17-token prompt takes about 2.8 seconds. Begin after it so
# power samples cover steady speculative decode rather than prefill.
sleep "${PREFILL_WAIT_S:-5}"

nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms "$SAMPLE_MS" \
  >"$LOCAL_POWER" 2>&1 &
local_power_pid=$!
ssh -o BatchMode=yes -n "$PEER" \
  "nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms $SAMPLE_MS" \
  >"$PEER_POWER" 2>&1 &
peer_power_pid=$!
wait "$bench_pid"
kill "$local_power_pid" "$peer_power_pid" 2>/dev/null || true
wait "$local_power_pid" "$peer_power_pid" 2>/dev/null || true
trap - EXIT

python3 - "$LOG" "$LOCAL_POWER" "$PEER_POWER" <<'PY'
import json, pathlib, re, statistics, sys
log = pathlib.Path(sys.argv[1]).read_text()

def powers(path):
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            out.append(float(line.rsplit(',', 1)[1].strip()))
        except (IndexError, ValueError):
            pass
    # Drop one second at each boundary. These samples can include prefill or
    # process teardown rather than steady decode.
    return out[5:-5] if len(out) > 10 else out

match = re.search(r'decode median\s+[0-9.]+ ms/step\s+\(([0-9.]+) agg tok/s at batch ([0-9]+), ([0-9.]+) tok/s/stream\)', log)
if not match:
    raise SystemExit('throughput line missing')
aggregate = float(match.group(1))
local = powers(sys.argv[2])
peer = powers(sys.argv[3])
if not local or not peer:
    raise SystemExit('power samples missing')
local_w = statistics.fmean(local)
peer_w = statistics.fmean(peer)
total_w = local_w + peer_w
result = {
    'aggregate_tokens_per_second': aggregate,
    'batch': int(match.group(2)),
    'tokens_per_second_per_stream': float(match.group(3)),
    'local_gpu_watts_mean': local_w,
    'peer_gpu_watts_mean': peer_w,
    'pair_gpu_watts_mean': total_w,
    'aggregate_tokens_per_second_per_gpu_watt': aggregate / total_w,
    'local_samples': len(local),
    'peer_samples': len(peer),
    'power_scope': 'NVML power.draw summed across both GB10 GPUs; excludes unreported platform power',
}
print(json.dumps(result, indent=2, sort_keys=True))
PY
