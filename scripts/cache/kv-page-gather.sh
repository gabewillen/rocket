#!/usr/bin/env bash
# Page-size sweep for the KV pool, one page size per process.
#
# The bench allocates an 8.25 GiB arena per configuration; running several in
# one process makes the later ones up to 30% slower as the allocator recycles
# that much unified memory, so each page size gets its own invocation.
#
#   scripts/cache/kv-page-gather.sh [scatter|dense]
#
# Cites: engines/glm5-moe-nvfp4-2b/bench/kv_page_gather.cu
set -euo pipefail

SEL="${1:-scatter}"
ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../engines/glm5-moe-nvfp4-2b" && pwd)"
BIN="$ENGINE/build/bench/bench-kv-page-gather"

if [[ ! -x "$BIN" ]]; then
  cmake -S "$ENGINE" -B "$ENGINE/build" >/dev/null
  cmake --build "$ENGINE/build" --target bench-kv-page-gather -j8 >/dev/null
fi

echo "ctx=65536 streams=8 iters=20 sel=$SEL, layer 3 of 11, synthetic KV, best of 3"
printf '%10s %10s %8s %7s %12s %12s\n' page_tok page_KiB pages layout scan_ms gather_ms
for PT in 128 256 512 1024 2048; do
  "$BIN" --page "$PT" --sel "$SEL" | awk '{printf "%10s %10s %8s %7s %12s %12s\n", $1,$2,$3,$4,$5,$6}'
done
