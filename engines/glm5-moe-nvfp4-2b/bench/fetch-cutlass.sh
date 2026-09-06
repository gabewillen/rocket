#!/usr/bin/env bash
# Fetch the exact CUTLASS commit the nvfp4-grouped-gemm bench was measured against.
# Runnable as-is from anywhere.
set -euo pipefail

SHA="59e3a3338d516ca6ce0e073af8da65289678a35c"   # CUTLASS 4.8.0
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party"

mkdir -p "$DEST"
if [ -f "$DEST/cutlass/ROCKET_PINNED_SHA" ] && \
   [ "$(cat "$DEST/cutlass/ROCKET_PINNED_SHA")" = "$SHA" ]; then
  echo "cutlass already at $SHA"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "fetching CUTLASS $SHA"
curl -sSL -o "$tmp/cutlass.tar.gz" "https://codeload.github.com/NVIDIA/cutlass/tar.gz/$SHA"
tar xzf "$tmp/cutlass.tar.gz" -C "$tmp"
mv "$tmp/cutlass-$SHA" "$DEST/cutlass"
echo "$SHA" > "$DEST/cutlass/ROCKET_PINNED_SHA"
echo "cutlass at $DEST/cutlass ($SHA)"
