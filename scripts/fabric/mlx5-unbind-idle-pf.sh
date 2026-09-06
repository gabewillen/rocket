#!/usr/bin/env bash
# Unbind the two idle ConnectX PFs (f0 port on each card). The DGX Spark
# wires only the f1 ports into the RoCE fabric; the f0 PFs still hold
# ~142 MiB of firmware pages each at probe. Unbinding both returned
# 444 MiB (peer) / 393 MiB (head) of MemAvailable with the fabric ACTIVE.
# Refuses to touch a PF whose RDMA link is not DOWN/DISABLED.
# Install as a boot service:
#   sudo ./mlx5-unbind-idle-pf.sh install
set -euo pipefail

IDLE_PFS="0000:01:00.0 0002:01:00.0"

unbind() {
  for pf in $IDLE_PFS; do
    [ -e "/sys/bus/pci/drivers/mlx5_core/$pf" ] || continue
    if rdma link 2>/dev/null | grep -A0 "$(basename /sys/bus/pci/devices/$pf/net/* 2>/dev/null)" | grep -q ACTIVE; then
      echo "refusing: $pf has an ACTIVE rdma link" >&2; exit 1
    fi
    echo "$pf" > /sys/bus/pci/drivers/mlx5_core/unbind
    echo "unbound $pf"
  done
}

install() {
  local self unit=/etc/systemd/system/mlx5-unbind-idle-pf.service
  self=$(readlink -f "$0")
  sudo install -m 0755 "$self" /usr/local/sbin/mlx5-unbind-idle-pf.sh
  sudo tee "$unit" >/dev/null <<UNIT
[Unit]
Description=Unbind idle mlx5 PFs to reclaim firmware pages
After=network-pre.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/mlx5-unbind-idle-pf.sh unbind

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable mlx5-unbind-idle-pf.service
}

"${1:-unbind}"
