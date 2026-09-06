#!/usr/bin/env bash
# RoCE bandwidth and latency between two GB10 nodes (head, peer).
#   ./fabric-bw.sh <peer-rail1-ip> <peer-rail2-ip>
# Assumes perftest on both nodes and passwordless ssh to the peer.
#
# Do not put "ib_write_bw" in a pkill pattern here: pkill -f matches the
# remote shell's own command line and kills the server it just launched.
set -uo pipefail

PEER1=${1:-192.168.100.11}
PEER2=${2:-192.168.101.11}
DEV1=${DEV1:-rocep1s0f1}
DEV2=${DEV2:-roceP2p1s0f1}
GID=${GID:-3}          # RoCE v2, IPv4. Confirm with the gid_attrs dump below.
SIZE=${SIZE:-1048576}
SECS=${SECS:-5}

echo "== RoCE v2 GID indices on $DEV1 =="
for i in $(seq 0 3); do
  t=$(cat "/sys/class/infiniband/$DEV1/ports/1/gid_attrs/types/$i" 2>/dev/null) || continue
  echo "  idx=$i type=$t gid=$(cat "/sys/class/infiniband/$DEV1/ports/1/gids/$i")"
done

echo "== port and host link =="
ibv_devinfo -d "$DEV1" | grep -E 'state:|active_mtu|max_mtu'
for b in $(lspci | grep -i connectx | cut -d' ' -f1); do
  echo "  $b $(lspci -s "$b" -vv 2>/dev/null | grep -E 'LnkSta:' | tr -s ' ')"
done

run_bw() {  # dev peer port qp label
  ssh -n -o BatchMode=yes "$2" \
    "ib_write_bw -d $1 -x $GID -F --report_gbits -p $3 -q $4 -s $SIZE -D $SECS" \
    >/dev/null 2>&1 &
  local srv=$!
  sleep 3
  ib_write_bw -d "$1" -x $GID -F --report_gbits -p "$3" -q "$4" -s $SIZE -D $SECS "$2" \
    2>/dev/null | awk -v l="$5" '/^ '"$SIZE"'/{print l, $4, "Gb/s"}'
  wait $srv 2>/dev/null
}

echo "== single rail =="
for q in 1 4; do run_bw "$DEV1" "$PEER1" 18601 "$q" "rail1 qp=$q"; done

echo "== both rails concurrently =="
run_bw "$DEV1" "$PEER1" 18611 4 "rail1 both" &
run_bw "$DEV2" "$PEER2" 18612 4 "rail2 both" &
wait

echo "== write latency, t_typical =="
for s in 1024 8192 16384; do
  ssh -n -o BatchMode=yes "$PEER1" \
    "ib_write_lat -d $DEV1 -x $GID -F -p 1862${s} -s $s -n 20000" >/dev/null 2>&1 &
  srv=$!
  sleep 3
  ib_write_lat -d "$DEV1" -x $GID -F -p "1862${s}" -s "$s" -n 20000 "$PEER1" \
    2>/dev/null | awk -v s="$s" '$1==s {print "  " s " B  " $6 " us"}'
  wait $srv 2>/dev/null
done
