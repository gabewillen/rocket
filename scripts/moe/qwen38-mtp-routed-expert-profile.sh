#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "usage: $0 BUILD_DIR [DEVICE] [WARMUP] [SAMPLES]" >&2
  exit 2
fi

build_dir=$1
device=${2:-0}
warmup=${3:-20}
samples=${4:-100}
binary="${build_dir}/qwen38-mtp-routed-expert-profile"

test -x "${binary}"
echo "BINARY_SHA256 $(sha256sum "${binary}" | cut -d' ' -f1)"
nvidia-smi --id="${device}" --query-gpu=uuid,name,compute_cap,clocks.current.sm,clocks.current.graphics,clocks.current.memory,pstate,power.draw,temperature.gpu,memory.used --format=csv,noheader
CUDA_VISIBLE_DEVICES="${device}" "${binary}" 0 "${warmup}" "${samples}"
nvidia-smi --id="${device}" --query-gpu=uuid,name,compute_cap,clocks.current.sm,clocks.current.graphics,clocks.current.memory,pstate,power.draw,temperature.gpu,memory.used --format=csv,noheader
