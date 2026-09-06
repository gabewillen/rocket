#!/usr/bin/env bash
# Safe replacement for page_owner-based mlx5 accounting.
# Reports firmware (ICM) pages held per mlx5 PF. Read-only, microseconds.
# fw_pages_* are counted in 4 KiB firmware pages regardless of host PAGE_SIZE.
# Usage: sudo ./mlx5-fw-pages.sh
set -euo pipefail

FW_PAGE=4096
total=0
printf '%-16s %12s %10s\n' DEVICE FW_PAGES MiB
for d in /sys/kernel/debug/mlx5/*/; do
  [ -r "$d/pages/fw_pages_total" ] || continue
  n=$(cat "$d/pages/fw_pages_total")
  total=$((total + n))
  printf '%-16s %12d %10.1f\n' "$(basename "$d")" "$n" "$(echo "$n $FW_PAGE" | awk '{print $1*$2/1048576}')"
done
printf '%-16s %12d %10.1f\n' TOTAL "$total" "$(echo "$total $FW_PAGE" | awk '{print $1*$2/1048576}')"

echo
echo "breakdown (all devices):"
for k in fw_pages_vfs fw_pages_sfs fw_pages_host_pf fw_pages_ec_vfs \
         fw_pages_alloc_failed fw_pages_give_dropped fw_pages_reclaim_discard; do
  s=0
  for d in /sys/kernel/debug/mlx5/*/; do
    [ -r "$d/pages/$k" ] && s=$((s + $(cat "$d/pages/$k")))
  done
  printf '  %-26s %d\n' "$k" "$s"
done
