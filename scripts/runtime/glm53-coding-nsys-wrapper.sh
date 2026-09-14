#!/usr/bin/env bash
# Profile only the target decode range bracketed by cudaProfilerStart/Stop.
set -euo pipefail
OUT=${NSYS_OUT:-/tmp/glm53-coding-target-only}
exec nsys profile --force-overwrite=true --output "$OUT" \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none "$@"
