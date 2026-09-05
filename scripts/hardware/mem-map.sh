#!/usr/bin/env bash
# Waterfall from the 128 GiB part to MemAvailable on a GB10 spark.
# Needs root for /proc/iomem addresses and dmesg.
set -euo pipefail

part_kb=$((128 * 1024 * 1024))
# dmesg ring can rotate the boot line out; journalctl keeps it. awk avoids
# the grep -m1 SIGPIPE that pipefail would turn into an abort.
phys_kb=$( { dmesg; journalctl -k -b -q 2>/dev/null; } | awk '
  match($0, /Memory: [0-9]+K\/[0-9]+K/) {
    split(substr($0, RSTART, RLENGTH), p, /[\/K]/); kb=p[3] } END { print kb }')
carve_kb=$(grep -E '^[0-9a-f]+-[0-9a-f]+ : reserved' /proc/iomem |
  awk -F'[ :-]+' '{s+=(strtonum("0x"$2)-strtonum("0x"$1)+1)/1024} END {printf "%d", s}')
total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)

avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)

row() { printf "%-42s %12d kB %9.2f GiB\n" "$1" "$2" "$(echo "$2" | awk '{print $1/1048576}')"; }
row "128 GiB part"                         "$part_kb"
row "invisible to kernel (firmware)"       "$((part_kb - phys_kb))"
row "nomap carveouts (iomem reserved)"     "$carve_kb"
row "kernel image + memmap + boot"         "$((phys_kb - carve_kb - total_kb))"
row "MemTotal"                             "$total_kb"
row "in use (MemTotal - MemAvailable)"     "$((total_kb - avail_kb))"
row "MemAvailable"                         "$avail_kb"

echo
echo "largest carveouts:"
grep -E '^[0-9a-f]+-[0-9a-f]+ : reserved' /proc/iomem |
  awk -F'[ :-]+' '{n=(strtonum("0x"$2)-strtonum("0x"$1)+1)/1048576;
                   if (n>=16) printf "  %8.1f MiB  %s-%s\n", n, $1, $2}' | sort -rn
