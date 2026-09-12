#!/usr/bin/env bash
# Materialize an NVFP4 overlay object for a comma-separated family of BF16
# checkpoint tensors whose loader-side consumer is one row-concatenated matrix
# (today: the KDA q/k/v projection triple, one object per layer).
#
# usage: materialize-nvfp4-family.sh SNAPSHOT OUTPUT_DIR SNAPSHOT_KEY T0,T1,[T2]
# The SNAPSHOT must have checksums.blake3 for the inventory; per-tensor
# SHA-256s come from scripts/numerics/nvfp4-overlay-inventory.py.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
engine="$repo_root/engines/glm5-moe-nvfp4-2b"
build="$engine/build"

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SNAPSHOT OUTPUT_DIR SNAPSHOT_KEY T0,T1[,T2]" >&2
  exit 2
fi
snapshot=$1
output=$2
snapshot_key=$3
names=$4

read -r -a tensors <<<"$(echo "$names" | tr ',' ' ')"

shas=$(python3 "$repo_root/scripts/numerics/nvfp4-overlay-inventory.py" "$snapshot" |
  python3 - "$names" <<'PY'
import json, sys
x = json.load(sys.stdin)
want = sys.argv[1].split(',')
by_name = {t["source_tensor"]: t for t in x["candidate_tensors"]}
missing = [n for n in want if n not in by_name]
if missing:
    sys.exit(f"inventory lacks candidates: {missing}")
print(",".join(f'{n}={by_name[n]["source_sha256"]}' for n in want))
PY
)

cmake --build "$build" --target rocket_fuel -j"$(nproc)" >/dev/null
c++ -std=c++20 -O3 -I"$engine/src" \
  "$repo_root/scripts/numerics/materialize-nvfp4-overlay.cc" \
  "$build/librocket_fuel.a" -o "$build/materialize-nvfp4-overlay"
exec "$build/materialize-nvfp4-overlay" "$snapshot" "$output" "$snapshot_key" "$shas"
