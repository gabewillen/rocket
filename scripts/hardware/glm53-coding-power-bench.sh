#!/usr/bin/env bash
# Sum NVML GPU energy over the exact distinct-prompt decode interval.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PEER=${PEER:-${ROCKET_PEER:-192.168.100.11}}
SAMPLE_MS=${SAMPLE_MS:-200}
MARKER=${DECODE_MARKER:-/tmp/glm53-coding-decode.marker}
RESULT_JSON=${RESULT_JSON:-/tmp/glm53-coding-power-run.json}
POWER_JSON=${POWER_JSON:-/tmp/glm53-coding-power.json}
LOCAL_POWER=${LOCAL_POWER:-/tmp/glm53-coding-power-local.csv}
PEER_POWER=${PEER_POWER:-/tmp/glm53-coding-power-peer.csv}
rm -f "$MARKER" "$RESULT_JSON" "$POWER_JSON" "$LOCAL_POWER" "$PEER_POWER"
DECODE_MARKER=$MARKER RESULT_JSON=$RESULT_JSON "$REPO/scripts/runtime/glm53-coding-bench-pair.sh" &
bench_pid=$!
trap 'kill ${lp:-} ${rp:-} $bench_pid 2>/dev/null || true' EXIT
for _ in $(seq 1 36000); do
  [[ -f $MARKER ]] && break
  kill -0 "$bench_pid" 2>/dev/null || { wait "$bench_pid"; exit $?; }
  sleep 0.2
done
[[ -f $MARKER ]] || { echo 'decode marker missing' >&2; exit 1; }
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms "$SAMPLE_MS" >"$LOCAL_POWER" 2>&1 & lp=$!
ssh -o BatchMode=yes -n "$PEER" "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms $SAMPLE_MS" >"$PEER_POWER" 2>&1 & rp=$!
while [[ $(cat "$MARKER" 2>/dev/null) != end ]]; do
  kill -0 "$bench_pid" 2>/dev/null || break
  sleep 0.2
done
kill "$lp" "$rp" 2>/dev/null || true
wait "$lp" "$rp" 2>/dev/null || true
wait "$bench_pid"
trap - EXIT
python3 - "$RESULT_JSON" "$LOCAL_POWER" "$PEER_POWER" "$SAMPLE_MS" "$POWER_JSON" <<'PY'
import json,pathlib,statistics,sys
result=json.load(open(sys.argv[1]))
def read(path):
 out=[]
 for x in pathlib.Path(path).read_text().splitlines():
  try: out.append(float(x.strip()))
  except ValueError: pass
 return out
local,peer=read(sys.argv[2]),read(sys.argv[3])
if not local or not peer: raise SystemExit('power samples missing')
n=min(len(local),len(peer)); pair=[local[i]+peer[i] for i in range(n)]
dt=float(sys.argv[4])/1000
energy=sum(pair)*dt
useful=result['useful_output_tokens']
out={'useful_output_tokens':useful,'decode_seconds':result['decode_ms']/1000,
 'local_gpu_watts_mean':statistics.fmean(local),'peer_gpu_watts_mean':statistics.fmean(peer),
 'pair_gpu_watts_mean':statistics.fmean(pair),'pair_gpu_joules':energy,
 'aggregate_useful_tok_s':result['aggregate_useful_tok_s'],
 'useful_tokens_per_pair_gpu_watt':result['aggregate_useful_tok_s']/statistics.fmean(pair),
 'useful_tokens_per_pair_gpu_joule':useful/energy,'samples':n,
 'scope':'NVML power.draw summed across both GB10 GPUs; excludes CPU, DRAM, SSD, fabric, PSU, and wall losses'}
pathlib.Path(sys.argv[5]).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
PY
