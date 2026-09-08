#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# The measured vLLM image is 0.1.dev20073+g8e685d198 (image digest
# sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a).
# Its grouped-MoE scheduling references are fused_moe.py and
# experts/triton_moe.py (SHA256 d5955a3460746b66740b024f470aeca79bbebd10872c327f765f2d7fcb805f28
# and 1ace6e60427c94fc17f16490475637f6988f88e7e4796f58a752a753dd7fd397).
# Current official vLLM comparison:
# eb6b619ab256df1100cb75684a9ba78485506c3a, the same two paths plus
# configs/E=512,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json.
# The pinned and current GB10 config is byte-identical at SHA256
# 567272e93c10ccded2dadfe7f90ec38f548ac7e55245e61956663aa070994b52.
# Both revisions sort and pad routes by expert. The GB10 FP8 config uses
# BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, four warps, and grouped
# program ordering for expert-weight reuse. This harness measures Rocket's
# current one-route-per-CTA schedule before any schedule change.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BUILD_DIR OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

build_dir=$1
output_dir=$2
device=${3:-0}
binary="${build_dir}/qwen38-mtp-routed-expert-profile"
ncu=${ROCKET_NCU:-/usr/local/cuda/bin/ncu}
kernel='regex:.*gate_up_silu.*'
metrics='gpu__time_duration.sum,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__t_sectors_srcunit_tex_lookup_hit.sum,lts__t_sectors_srcunit_tex_lookup_miss.sum,lts__t_sectors_aperture_device_op_read.sum,lts__t_sectors_aperture_sysmem_op_read.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_misc_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_sleeping_per_warp_active.pct,smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct'
phase=preflight
profile_case=none

failure_envelope() {
  local status=$1
  local partial=none
  local tool_log=none
  trap - ERR
  if [[ "${profile_case}" != none &&
        -f "${output_dir}/${profile_case}.ncu-rep" &&
        ! -f "${output_dir}/${profile_case}.csv" ]]; then
    if timeout 15s sudo -n "${ncu}" \
      --import "${output_dir}/${profile_case}.ncu-rep" --page raw --csv \
      >"${output_dir}/${profile_case}.csv" 2>/dev/null; then
      :
    fi
  fi
  if [[ -f "${output_dir}/${profile_case}.csv" ]]; then
    partial="${output_dir}/${profile_case}.csv"
    if python3 scripts/moe/summarize-mtp-gate-up-ncu.py --incomplete \
      "${output_dir}/${profile_case}.csv" >&2; then
      :
    fi
  fi
  if [[ -f "${output_dir}/${profile_case}.ncu.log" ]]; then
    tool_log="${output_dir}/${profile_case}.ncu.log"
    tail -20 "${tool_log}" >&2
  fi
  printf 'INCOMPLETE\tvalid=false\tphase=%s\ttool_exit=%d\tkernel=%s\tcase=%s\tpartial_counters=%s\ttool_log=%s\n' \
    "${phase}" "${status}" "${kernel}" "${profile_case}" "${partial}" "${tool_log}" >&2
}
trap 'status=$?; failure_envelope "${status}"; exit "${status}"' ERR

test -x "${binary}"
test -x "${ncu}"
sudo -n true
mkdir -p "${output_dir}"
for artifact_case in c8-k7-r1 c16-k7-r1; do
  for suffix in .csv .ncu-rep .ncu.log; do
    if [[ -e "${output_dir}/${artifact_case}${suffix}" ]]; then
      echo "output artifact already exists: ${output_dir}/${artifact_case}${suffix}" >&2
      false
    fi
  done
done

head=$(git rev-parse HEAD)
if [[ -z "${ROCKET_EXPECTED_HEAD:-}" ]]; then
  echo "ROCKET_EXPECTED_HEAD is required" >&2
  false
fi
if [[ "${head}" != "${ROCKET_EXPECTED_HEAD}" ]]; then
  echo "identity drift: expected ${ROCKET_EXPECTED_HEAD}, found ${head}" >&2
  false
fi

local_table=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if rg -q '[0-9]' <<<"${local_table}"; then
  echo "local GPU compute table is not empty" >&2
  false
fi
if [[ -n "${ROCKET_PEER:-}" ]]; then
  peer_table=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${ROCKET_PEER}" \
    "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits")
  if rg -q '[0-9]' <<<"${peer_table}"; then
    echo "peer GPU compute table is not empty" >&2
    false
  fi
fi

echo "GIT_HEAD ${head}"
echo "BINARY_SHA256 $(sha256sum "${binary}" | cut -d' ' -f1)"
"${ncu}" --version | head -2
"${ncu}" --list-chips | rg -q '(^|[ ,])gb10b([ ,]|$)'
"${ncu}" --query-metrics --chips gb10b --metrics "${metrics}" >/dev/null
if [[ "${ROCKET_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "PREFLIGHT valid=true chip=gb10b cases=c8-k7-r1,c16-k7-r1 kernel=${kernel}"
  trap - ERR
  exit 0
fi
nvidia-smi --id="${device}" --query-gpu=uuid,name,compute_cap,clocks.current.sm,clocks.current.graphics,clocks.current.memory,pstate,power.draw,temperature.gpu,memory.used --format=csv,noheader

sections=(
  SpeedOfLight
  ComputeWorkloadAnalysis
  MemoryWorkloadAnalysis
  Occupancy
  SchedulerStats
  WarpStateStats
)
section_args=()
for section in "${sections[@]}"; do
  section_args+=(--section "${section}")
done
for profile_case in c8-k7-r1 c16-k7-r1; do
  phase=collect
  report="${output_dir}/${profile_case}"
  log="${output_dir}/${profile_case}.ncu.log"
  sudo -n env CUDA_VISIBLE_DEVICES="${device}" "${ncu}" \
    --target-processes application-only \
    --graph-profiling node \
    --kernel-name-base demangled \
    --kernel-name "${kernel}" \
    --launch-count 1 \
    --cache-control all \
    --clock-control none \
    "${section_args[@]}" \
    --metrics "${metrics}" \
    --export "${report}" --force-overwrite \
    "${binary}" 0 1 2 "${profile_case}" >"${log}" 2>&1

  phase=export
  sudo -n "${ncu}" --import "${report}.ncu-rep" --page raw --csv \
    >"${output_dir}/${profile_case}.csv"
  sudo -n chown "$(id -u):$(id -g)" "${report}.ncu-rep"
done

phase=summarize
python3 scripts/moe/summarize-mtp-gate-up-ncu.py \
  "${output_dir}/c8-k7-r1.csv" "${output_dir}/c16-k7-r1.csv"
nvidia-smi --id="${device}" --query-gpu=uuid,name,compute_cap,clocks.current.sm,clocks.current.graphics,clocks.current.memory,pstate,power.draw,temperature.gpu,memory.used --format=csv,noheader
trap - ERR
