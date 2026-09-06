#!/usr/bin/env bash
#
# !!! HAZARD - DO NOT RUN ON 6.17.13-rocket64k !!!
#
# On the custom 64 KiB-page kernel (6.17.13-rocket64k, aarch64/GB10), reading
# /sys/kernel/debug/page_owner_stacks/show_stacks past the first seq_file
# buffer (~4 KiB) wedges the reading task in uninterruptible (D) state
# permanently. `timeout` cannot reap it ("error waiting for command: No child
# processes"). Accumulating such readers took node gx10-2a13 fully offline on
# 2026-09-06 (no response on the RoCE rail or WiFi; needs a power cycle).
#
# Observed on gx10-2a13:
#   head -c  3000 show_stacks -> 3000 bytes in 8 ms      (ok)
#   head -c 30000 show_stacks -> 0 bytes, blocked 20 s   (wedged)
#   dd bs=1M      show_stacks -> 0 bytes, blocked 60 s   (wedged)
#
# Use the mlx5 driver's own firmware-page accounting instead. It is cheap,
# exact, and per-device:
#   sudo grep . /sys/kernel/debug/mlx5/*/pages/fw_pages_total
# Units there are 4 KiB firmware pages regardless of host PAGE_SIZE.
#
# The aggregation below is kept only for kernels where show_stacks is safe.
set -euo pipefail
echo "refusing to run: see hazard note in $0" >&2
exit 1

SRC=/sys/kernel/debug/page_owner_stacks/show_stacks
PAGE_BYTES=$(getconf PAGESIZE)

awk -v page_bytes="$PAGE_BYTES" -v mode="${1:---total}" '
function flush() {
  if (n > 0 && stack ~ /mlx5/) {
    total += n
    by[(deepest != "") ? deepest : "mlx5_unknown"] += n
  }
  stack = ""; n = 0; deepest = ""
}
/^[[:space:]]*$/ { flush(); next }
/^nr_base_pages:/ { n = $2 + 0; next }
{
  stack = stack $0 "\n"
  if ($0 ~ /\[mlx5/) { f = $1; sub(/\+.*/, "", f); deepest = f }
}
END {
  flush()
  if (mode == "--by-site")
    for (s in by) printf "%12d pages %10.1f MiB  %s\n", by[s], by[s]*page_bytes/1048576, s
  printf "TOTAL_PAGES=%d TOTAL_MIB=%.1f PAGE_BYTES=%d\n", total, total*page_bytes/1048576, page_bytes
}' "$SRC"
