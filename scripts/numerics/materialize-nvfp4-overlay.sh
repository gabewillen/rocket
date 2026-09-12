#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
engine="$repo_root/engines/glm5-moe-nvfp4-2b"
build="$engine/build"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SNAPSHOT OUTPUT KDA_LAYER" >&2
  exit 2
fi
snapshot=$1
output=$2
layer=$3
tensor="model.language_model.layers.$layer.self_attn.q_proj.weight"
read -r snapshot_key source_sha256 < <(
  python3 "$repo_root/scripts/numerics/nvfp4-overlay-inventory.py" "$snapshot" |
    python3 -c 'import json,sys; name=sys.argv[1]; x=json.load(sys.stdin); t=next(t for t in x["candidate_tensors"] if t["source_tensor"]==name); print(x["source"]["snapshot_key"], t["source_sha256"])' "$tensor"
)

cmake --build "$build" --target rocket_fuel -j"$(nproc)"
c++ -std=c++20 -O3 -I"$engine/src" \
  "$repo_root/scripts/numerics/materialize-nvfp4-overlay.cc" \
  "$build/librocket_fuel.a" -o "$build/materialize-nvfp4-overlay"
exec "$build/materialize-nvfp4-overlay" "$snapshot" "$output" "$snapshot_key" "$source_sha256" "$tensor"
