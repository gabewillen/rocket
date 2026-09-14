#!/usr/bin/env bash
# Capture one target-decode kernel after cudaProfilerStart; prefill is excluded.
set -euo pipefail
OUT=${NCU_OUT:-/tmp/glm53-coding-target-only.ncu-rep}
KERNEL=${NCU_KERNEL:-regex:mla_(scores|context)_kernel}
exec sudo -E /usr/local/cuda/bin/ncu --force-overwrite --export "$OUT" --profile-from-start off \
  --kernel-name "$KERNEL" --launch-count "${NCU_LAUNCH_COUNT:-2}" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active "$@"
