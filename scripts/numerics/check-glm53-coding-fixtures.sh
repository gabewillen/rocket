#!/usr/bin/env bash
# Compile or execute isolated public coding fixtures used by quality gates.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FIX="$ROOT/scripts/runtime/fixtures/coding"
python3 -m py_compile "$ROOT/scripts/runtime/glm53_coding_workload.py"
nvcc -std=c++17 -x cu -c "$ROOT/engines/glm5-moe-nvfp4-2b/bench/alloc_latency.cu" -o /tmp/glm53-fixture-cuda.o
c++ -std=c++17 -c "$ROOT/engines/glm5-moe-nvfp4-2b/src/kv/page_pool.cc" \
  -I"$ROOT/engines/glm5-moe-nvfp4-2b/src" -o /tmp/glm53-fixture-cpp.o
rustc --test "$FIX/rust.rs" -o /tmp/glm53-fixture-rust
/tmp/glm53-fixture-rust
node "$FIX/javascript.js"
bash -n "$ROOT/scripts/runtime/glm53-coding-bench-pair.sh"
rm -f /tmp/glm53-fixture-cuda.o /tmp/glm53-fixture-cpp.o /tmp/glm53-fixture-rust
printf 'coding fixtures: PASS\n'
