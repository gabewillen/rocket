#!/usr/bin/env bash
# Rebuild the Ubuntu NVIDIA 64K kernel without CONFIG_RANDOM_KMALLOC_CACHES
# and with CONFIG_PAGE_OWNER, to measure the slab tax and attribute the
# unaccounted ~1.3 GiB. Run on the target spark. Stage 1: build debs.
#
#   ./build-kernel-norndslab.sh build     # ~1.7G source + long compile
#   ./build-kernel-norndslab.sh install   # installs debs, edits grub
#
#   ./build-kernel-norndslab.sh sign      # sign vmlinuz with the node MOK
#
# Secure Boot is enabled on these nodes. sign creates a machine-owner key in
# /var/lib/rocket-mok on first use and queues it for enrollment; the first
# reboot after that shows MokManager once on the physical console (password
# rocket64k). Enrollment persists, later reboots and later signed kernels
# boot with no prompt. Never commit MOK.key anywhere.
#
# The NVIDIA prebuilt module packages (linux-modules-nvidia-580-open-*) are
# ABI-locked and will not load; rebuild the open modules from the source in
# /usr/src/nvidia-580.173.02 against the new headers after install.
set -euo pipefail

KBUILD_DIR=${KBUILD_DIR:-$HOME/kbuild}
STOCK=6.17.0-1031-nvidia-64k     # config donor and headers source package
LOCALVER=-rocket64k

build() {
  # deb-src is Types: deb in stock ubuntu.sources
  sudo sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
  sudo apt-get update -qq
  sudo apt-get -y install build-essential flex bison libssl-dev libelf-dev \
    bc debhelper rsync kmod cpio dwarves libdw-dev python3-setuptools
  mkdir -p "$KBUILD_DIR" && cd "$KBUILD_DIR"
  # apt-get source on the image package returns the signing wrapper, and on
  # linux-nvidia-6.17 returns the meta package. The headers package maps to
  # the real tree.
  apt-get source "linux-nvidia-6.17-headers-${STOCK%-nvidia-64k}"
  cd linux-nvidia-6.17-*/
  cp "/boot/config-$STOCK" .config
  scripts/config \
    --disable RANDOM_KMALLOC_CACHES \
    --enable PAGE_EXTENSION --enable PAGE_OWNER \
    --set-str SYSTEM_TRUSTED_KEYS "" --set-str SYSTEM_REVOCATION_KEYS "" \
    --disable DEBUG_INFO --enable DEBUG_INFO_NONE --disable DEBUG_INFO_BTF \
    --set-str LOCALVERSION "$LOCALVER"
  make olddefconfig
  grep -E 'RANDOM_KMALLOC|PAGE_OWNER=|LOCALVERSION=' .config
  make -j"$(nproc)" bindeb-pkg
  ls -la ../*.deb
}

install() {
  cd "$KBUILD_DIR"
  sudo dpkg -i linux-image-*"$LOCALVER"*.deb linux-headers-*"$LOCALVER"*.deb
  # page_owner=on makes /sys/kernel/debug/page_owner attribute allocations
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 page_owner=on"/' /etc/default/grub
  sudo update-grub
  echo "Reboot to the $LOCALVER kernel, then rebuild the NVIDIA open modules:"
  echo "  cd /usr/src/nvidia-580.173.02 && sudo make -j\$(nproc) modules && sudo make modules_install && sudo depmod"
}

sign() {
  local kver
  kver=$(ls /boot/vmlinuz-*"$LOCALVER" | sed 's|/boot/vmlinuz-||' | head -1)
  sudo apt-get -y install sbsigntool mokutil
  sudo mkdir -p /var/lib/rocket-mok && cd /var/lib/rocket-mok
  [ -f MOK.key ] || sudo openssl req -new -x509 -newkey rsa:2048 -keyout MOK.key \
    -out MOK.crt -outform PEM -nodes -days 36500 -subj "/CN=rocket kernel signing/"
  sudo chmod 600 MOK.key
  sudo openssl x509 -in MOK.crt -outform DER -out MOK.der
  sudo sbsign --key MOK.key --cert MOK.crt \
    --output "/boot/vmlinuz-$kver.signed" "/boot/vmlinuz-$kver"
  sudo mv "/boot/vmlinuz-$kver.signed" "/boot/vmlinuz-$kver"
  sbverify --cert MOK.crt "/boot/vmlinuz-$kver"
  # no-op if this MOK is already enrolled
  mokutil --test-key MOK.der || printf 'rocket64k\nrocket64k\n' | sudo mokutil --import MOK.der
}

"${1:-build}"
