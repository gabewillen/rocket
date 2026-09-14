#!/usr/bin/env bash
# Corrupt one rank-local prefix payload. When REFERENCE_LOG is set, run the
# production pair launcher and prove both ranks fall back to recompute with the
# same generated tokens as the reference.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to a populated prefix cache}
SEGMENT=${SEGMENT:-$CACHE_DIR/segment-00000000.bin}
OFFSET=${OFFSET:-65536}
FAULT_LOG=${FAULT_LOG:-/tmp/glm53-prefix-cache-fault.log}
python3 - "$SEGMENT" "$OFFSET" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1]); off = int(sys.argv[2])
with p.open("r+b", buffering=0) as f:
    f.seek(off); b = f.read(1)
    if not b: raise SystemExit(f"no byte at {p}:{off}")
    f.seek(off); f.write(bytes([b[0] ^ 0x5A])); f.flush()
print(f"corrupted {p}:{off}")
PY

if [[ -z ${REFERENCE_LOG:-} ]]; then
  exit 0
fi
: "${PORT:?set PORT for the fallback pair run}"
"$REPO/scripts/runtime/glm53-nvfp4-spec-bench.sh" >"$FAULT_LOG" 2>&1
peer_log=${PEER_LOG:-/tmp/spec-rank1.log}
grep -q 'prefix    restored 0/' "$FAULT_LOG"
grep -q 'prefix    restored 0/' "$peer_log"
python3 - "$REFERENCE_LOG" "$FAULT_LOG" "$REPO/scripts/runtime/glm53-prefix-cache-bench.py" <<'PY'
import importlib.util, pathlib, sys
script = pathlib.Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("prefix_bench", script)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ref = mod.summarize(pathlib.Path(sys.argv[1]))
got = mod.summarize(pathlib.Path(sys.argv[2]))
if not ref["token_ids"] or ref["token_ids"] != got["token_ids"]:
    raise SystemExit("fault fallback token IDs differ from reference")
print("PASS: both ranks fell back to zero and token IDs match the reference")
PY
