#!/usr/bin/env bash
# Single-booster static-fire run on one node with concurrent GPU clock/power
# and package thermal-zone sampling, to separate "GPU didn't boost" from
# "kswapd/kcompactd stole cycles" when one GB10 runs the same work slower
# than its pair.
#
#   scripts/hardware/single-booster-clock-thermal.sh [label] [cache-gib]
#
# Run once per node (this node only; ssh to the peer and run it there too
# for a comparison). Writes to /tmp/rocket-repro/<label>.{out,gpu.csv}.
# Needs engines/glm5-moe-nvfp4-2b/build/rocket-static-fire already built.
set -uo pipefail

label=${1:-run}
cache_gib=${2:-20}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
bin="$repo/engines/glm5-moe-nvfp4-2b/build/rocket-static-fire"
outdir=/tmp/rocket-repro
mkdir -p "$outdir"

if [[ ! -x $bin ]]; then
  echo "rocket-static-fire is not built" >&2
  exit 77
fi

echo "== package thermal zones before =="
for z in /sys/class/thermal/thermal_zone*; do
  [[ -r $z/temp ]] || continue
  printf '  %s %s %dC\n' "$z" "$(cat "$z/type" 2>/dev/null)" "$(($(cat "$z/temp") / 1000))"
done

nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,temperature.gpu.tlimit,\
clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,pstate \
  --format=csv -l 1 > "$outdir/$label.gpu.csv" &
gpu_pid=$!

"$bin" --mode single --sweep 1 --expert-cache-gib "$cache_gib" --skip-parity 1 \
  > "$outdir/$label.out" 2>&1
rc=$?

sleep 1
kill "$gpu_pid" 2>/dev/null

echo
echo "== $label (rc=$rc) =="
tail -3 "$outdir/$label.out"
echo "max SM clock during run:"
sort -t, -k2 -rn "$outdir/$label.gpu.csv" | head -1
