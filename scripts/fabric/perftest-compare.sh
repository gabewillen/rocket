#!/usr/bin/env bash
# Raw perftest reference for the rocket fabric transport, at the same message
# sizes scripts/fabric/fabric-microbench.sh reports. The transport is only
# worth building on if it lands close to this.
#
#   scripts/fabric/perftest-compare.sh
#
# Unidirectional ib_write_bw on both rails concurrently (aggregate, the shape
# the transport's `oneway` column measures) and ib_write_lat on rail 1.
#
# Do not put "ib_write_bw" in a pkill pattern here: pkill -f matches the
# remote shell's own command line and kills the server it just launched.
set -uo pipefail

PEER1=${1:-192.168.100.11}
PEER2=${2:-192.168.101.11}
DEV1=${DEV1:-rocep1s0f1}
DEV2=${DEV2:-roceP2p1s0f1}
GID=${GID:-3}
QP=${QP:-4}
SIZES=${SIZES:-"2048 8192 32768 131072 524288 1048576 2097152 4194304"}

bw_one() {  # dev peer port size -> Gb/s
  ssh -n -o BatchMode=yes "$2" \
    "ib_write_bw -d $1 -x $GID -F --report_gbits -p $3 -q $QP -s $4 -D 3" >/dev/null 2>&1 &
  local srv=$!
  sleep 2
  ib_write_bw -d "$1" -x $GID -F --report_gbits -p "$3" -q $QP -s "$4" -D 3 "$2" 2>/dev/null \
    | awk -v s="$4" '$1==s {print $4}'
  wait $srv 2>/dev/null
}

echo "== ib_write_bw, both rails concurrently, ${QP} QP each =="
printf '%10s %12s %12s %14s\n' bytes rail1_Gb/s rail2_Gb/s aggregate_GB/s
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for s in $SIZES; do
  bw_one "$DEV1" "$PEER1" 18701 "$s" >"$tmp/r1" 2>/dev/null &
  j1=$!
  bw_one "$DEV2" "$PEER2" 18702 "$s" >"$tmp/r2" 2>/dev/null &
  j2=$!
  wait $j1 $j2
  a=$(tr -d ' \n' <"$tmp/r1")
  b=$(tr -d ' \n' <"$tmp/r2")
  printf '%10s %12s %12s %14s\n' "$s" "${a:-na}" "${b:-na}" \
    "$(awk -v x="${a:-0}" -v y="${b:-0}" 'BEGIN{printf "%.2f", (x+y)/8}')"
done

echo
echo "== ib_write_lat, rail 1, t_typical, 20000 iterations =="
printf '%10s %10s\n' bytes us
for s in $SIZES; do
  ssh -n -o BatchMode=yes "$PEER1" \
    "ib_write_lat -d $DEV1 -x $GID -F -p 18711 -s $s -n 20000" >/dev/null 2>&1 &
  srv=$!
  sleep 2
  ib_write_lat -d "$DEV1" -x $GID -F -p 18711 -s "$s" -n 20000 "$PEER1" 2>/dev/null \
    | awk -v s="$s" '$1==s {printf "%10s %10s\n", s, $6}'
  wait $srv 2>/dev/null
done
