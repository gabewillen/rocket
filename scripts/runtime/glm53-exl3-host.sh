#!/usr/bin/env bash
# Host-native two-node launcher for MiaAI GLM-5.3-Flash EXL3 on 64 KiB GB10.
#
# This is intentionally Docker-free.  The model source and the patched vLLM
# source are copied into an explicit Rocket cache, then both ranks use the
# same host venv and local safetensors files.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

ACTION=${1:-status}
RECIPE_DIR=${RECIPE_DIR:-/home/glwillen/spark-stack/GLM-5.3-Flash-EXL3-2x-DGX-Sparks}
VLLM_SOURCE=${VLLM_SOURCE:-/home/glwillen/vllm-glm53}
WORK_DIR=${WORK_DIR:-$HOME/.cache/rocket-glm53-exl3-host}
PATCHED_SOURCE=${PATCHED_SOURCE:-$WORK_DIR/vllm}
VENV=${VENV:-$WORK_DIR/venv}
MODEL_DIR=${MODEL_DIR:-$WORK_DIR/models/glm-5.3-flash-exl3-tr3-4bpw}
DFLASH_DIR=${DFLASH_DIR:-$WORK_DIR/models/glm-5.3-flash-dflash2}
LOG_DIR=${LOG_DIR:-$WORK_DIR/logs}
MANIFEST=${MANIFEST:-$WORK_DIR/manifest.json}
REMOTE_HOST=${REMOTE_HOST:-glwillen@192.168.100.11}
REMOTE_WORK_DIR=${REMOTE_WORK_DIR:-$WORK_DIR}
REMOTE_RECIPE_DIR=${REMOTE_RECIPE_DIR:-$REMOTE_WORK_DIR/recipe}
REMOTE_VENV=${REMOTE_VENV:-$REMOTE_WORK_DIR/venv}
HEAD_IP=${HEAD_IP:-192.168.100.10}
WORKER_IP=${WORKER_IP:-192.168.100.11}
HEAD_IFACE=${HEAD_IFACE:-enp1s0f1np1}
WORKER_IFACE=${WORKER_IFACE:-enp1s0f1np1}
HEAD_HCA=${HEAD_HCA:-rocep1s0f1}
WORKER_HCA=${WORKER_HCA:-rocep1s0f1}
MASTER_PORT=${MASTER_PORT:-29521}
PORT=${PORT:-8888}

MODEL_ID=${MODEL_ID:-Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw}
MODEL_REVISION=${MODEL_REVISION:-25a44fdbf16862a46b7cc9921142c6c81350af2f}
DFLASH_ID=${DFLASH_ID:-incoai/GLM-5.3-Flash-DFlash2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.84}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
CUDAGRAPH_CAPTURE_SIZES=${CUDAGRAPH_CAPTURE_SIZES:-'1 2 4 8 16 24 27 32 40 48 56 64 72 96'}
# Empty = no torch profiler. Set to 1 to enable with the baked config.
PROFILER_ENABLE=${PROFILER_ENABLE:-}
DFLASH_TOKENS=${DFLASH_TOKENS:-7}
DFLASH_DRAFT_TP=${DFLASH_DRAFT_TP:-1}
DFLASH_SCHEDULE=${DFLASH_SCHEDULE:-'[[1,8,7],[9,24,2],[25,32,1]]'}
# Recipe serves throughput with the mixed-prefill decode floor off; the
# overlay default (skip) serializes new prefills behind running decodes.
GLM53_MIXED_PREFILL_CHUNK=${GLM53_MIXED_PREFILL_CHUNK:-off}
MIN_AVAILABLE_GIB=${MIN_AVAILABLE_GIB:-96}
ROCKET_VLLM_META_INIT=${ROCKET_VLLM_META_INIT:-1}
# The cudaHostRegister file-binding path stalled load at shard 0/120 and was
# removed from the patch set. The loader copies straight from the safetensors
# file view. Keep this at 0.
# The cudaHostRegister file-binding path stalled load at shard 0/120 and was
# removed from the patch set. The loader copies straight from the safetensors
# file view. Keep this at 0.
ROCKET_VLLM_UVA_WEIGHTS=${ROCKET_VLLM_UVA_WEIGHTS:-0}
SAFETENSORS_STRATEGY=${SAFETENSORS_STRATEGY:-prefetch}
DROP_CACHES=${DROP_CACHES:-0}
INSTALL_CUDA_DEPS=${INSTALL_CUDA_DEPS:-1}
INSTALL_INSTANTTENSOR=${INSTALL_INSTANTTENSOR:-0}
EXLLAMAV3_COMMIT=${EXLLAMAV3_COMMIT:-c5d9c657966ffeeaa9353f0cc899f18629da4a13}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '[glm53-exl3-host] %s\n' "$*"; }
remote() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" "$@"; }

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}"
  printf '\nUsage: %s {preflight|prepare|install|download|launch|stop|status}\n' "${BASH_SOURCE[0]}"
}

require_path() { [[ -e "$1" ]] || die "missing: $1"; }

check_node() {
  local label=$1 host=$2
  if [[ -z "$host" ]]; then
    [[ "$(getconf PAGESIZE)" == 65536 ]] || die "$label is not using 64 KiB pages"
    [[ "$(stat -f -c '%T' "$WORK_DIR")" == ext2/ext3 ]] || die "$label work cache is not EXT4"
    local avail
    avail=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    (( avail >= MIN_AVAILABLE_GIB * 1024 * 1024 )) || die "$label has less than ${MIN_AVAILABLE_GIB} GiB available RAM"
  else
    remote "[[ \"\$(getconf PAGESIZE)\" == 65536 ]] || exit 41; [[ \"\$(stat -f -c '%T' '$REMOTE_WORK_DIR')\" == ext2/ext3 ]] || exit 42; awk '/MemAvailable:/ {if (\$2 < ${MIN_AVAILABLE_GIB} * 1024 * 1024) exit 43}' /proc/meminfo"
  fi
}

preflight() {
  require_path "$RECIPE_DIR/overlay/exl3.py"
  require_path "$RECIPE_DIR/overlay/lazyspec/speculative.py"
  require_path "$VLLM_SOURCE/vllm/config/model.py"
  command -v ssh >/dev/null || die "ssh is required"
  command -v rsync >/dev/null || die "rsync is required"
  command -v curl >/dev/null || die "curl is required"
  [[ ! -x "$(command -v docker 2>/dev/null || true)" ]] || log "Docker is installed, but this launcher does not use it"
  mkdir -p "$WORK_DIR" "$LOG_DIR"
  check_node head ""
  remote "mkdir -p '$REMOTE_WORK_DIR' '$REMOTE_RECIPE_DIR' '$REMOTE_WORK_DIR/logs'"
  check_node worker "$REMOTE_HOST"
  remote "command -v python3 >/dev/null && command -v rsync >/dev/null"
  log "preflight OK: pages=65536, head=$HEAD_IP, worker=$WORKER_IP, memory floor=${MIN_AVAILABLE_GIB}GiB"
}

sync_runtime() {
  remote "mkdir -p '$REPO_ROOT/scripts/runtime'"
  scp -q \
    "$SCRIPT_DIR/glm53-exl3-host.sh" \
    "$SCRIPT_DIR/patch-vllm-glm53-exl3-host.py" \
    "$SCRIPT_DIR/patch-vllm-glm53-nope-sm120.py" \
    "$SCRIPT_DIR/patch-vllm-kernel-warmup.py" \
    "$SCRIPT_DIR/patch-vllm-uvm-loader.py" \
    "$REMOTE_HOST:$REPO_ROOT/scripts/runtime/"
}

sync_recipe() {
  remote "mkdir -p '$REMOTE_RECIPE_DIR/overlay'"
  rsync -a --delete "$RECIPE_DIR/overlay/" "$REMOTE_HOST:$REMOTE_RECIPE_DIR/overlay/"
}

prepare_source_copy() {
  mkdir -p "$PATCHED_SOURCE"
  rsync -a --delete --exclude '.git' --exclude '__pycache__' "$VLLM_SOURCE/" "$PATCHED_SOURCE/"
  python3 "$SCRIPT_DIR/patch-vllm-glm53-exl3-host.py" \
    --source "$PATCHED_SOURCE" \
    --recipe "$RECIPE_DIR" \
    --rocket-root "$REPO_ROOT" \
    --manifest "$MANIFEST"
}

prepare() {
  preflight
  prepare_source_copy
  sync_runtime
  sync_recipe
  rsync -a --delete --exclude '.git' --exclude '__pycache__' "$PATCHED_SOURCE/" "$REMOTE_HOST:$REMOTE_WORK_DIR/vllm/"
  scp -q "$MANIFEST" "$REMOTE_HOST:$REMOTE_WORK_DIR/manifest.json"
  log "prepared isolated source: $PATCHED_SOURCE"
  log "features: UVM loader, meta init, 64 KiB SM120 NoPE/page fixes, EXL3, DFlash2, lazy draft"
}

venv_python() { printf '%s/bin/python' "$VENV"; }

install_local() {
  local recipe=${RECIPE_DIR}
  mkdir -p "$WORK_DIR"
  if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --upgrade pip 'setuptools>=77,<81' wheel
  if [[ "$INSTALL_CUDA_DEPS" == 1 ]]; then
    "$VENV/bin/python" -m pip install -r "$PATCHED_SOURCE/requirements/common.txt"
    # instanttensor has no aarch64 wheel and its source build requires
    # python3-dev headers absent from the Spark base OS. It is an optional
    # loader accelerator; Rocket's safetensors UVM path remains enabled.
    sed '/^[[:space:]]*instanttensor[[:space:]]/d; /^-r[[:space:]]\+common\.txt$/d' \
      "$PATCHED_SOURCE/requirements/cuda.txt" \
      | "$VENV/bin/python" -m pip install -r /dev/stdin
    if [[ "$INSTALL_INSTANTTENSOR" == 1 ]]; then
      "$VENV/bin/python" -m pip install 'instanttensor>=0.1.9'
    fi
  fi
  "$VENV/bin/python" -m pip install 'setuptools-rust>=1.9.0' 'setuptools-scm>=8.0'
  VLLM_USE_PRECOMPILED=1 VLLM_VERSION_OVERRIDE=0.1.dev0 \
    "$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$PATCHED_SOURCE"

  local site_packages prebuilt_ext exllamav3_src
  site_packages=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  prebuilt_ext="$recipe/overlay/trellis/exllamav3_ext.cpython-312-aarch64-linux-gnu.so"
  if ! "$VENV/bin/python" -c 'import torch, exllamav3_ext; assert hasattr(exllamav3_ext, "exl3_moe")' 2>/dev/null; then
    require_path "$prebuilt_ext"
    install -m 0644 "$prebuilt_ext" "$site_packages/"
  fi
  # The EXL3 overlay needs the exllamav3 Python package (LinearEXL3) around
  # the prebuilt ext. The overlay stubs the FlashAttention __init__, so only
  # the source tree must be present. No ext rebuild: the trellis .so stays.
  exllamav3_src="$WORK_DIR/exllamav3-src"
  if [[ ! -f "$exllamav3_src/exllamav3/modules/quant/exl3.py" ]]; then
    require_path "$exllamav3_src/exllamav3/modules/quant/exl3.py"
  fi
  rm -rf "$site_packages/exllamav3"
  cp -r "$exllamav3_src/exllamav3" "$site_packages/"
  "$VENV/bin/python" -c 'import vllm, exllamav3_ext; assert hasattr(exllamav3_ext, "exl3_moe"); print(vllm.__file__, exllamav3_ext.__file__)'
}

install() {
  prepare
  install_local
  remote "RECIPE_DIR='$REMOTE_RECIPE_DIR' VLLM_SOURCE='$REMOTE_WORK_DIR/vllm' PATCHED_SOURCE='$REMOTE_WORK_DIR/vllm' VENV='$REMOTE_VENV' WORK_DIR='$REMOTE_WORK_DIR' INSTALL_CUDA_DEPS='$INSTALL_CUDA_DEPS' INSTALL_INSTANTTENSOR='$INSTALL_INSTANTTENSOR' EXLLAMAV3_COMMIT='$EXLLAMAV3_COMMIT' bash '$REPO_ROOT/scripts/runtime/glm53-exl3-host.sh' install-local"
  log "host venvs installed on both nodes"
}

download_one() {
  local id=$1 revision=$2 out=$3
  require_path "$out"
  command -v hf >/dev/null || die "hf CLI missing; install it before download"
  if [[ ! -f "$out/config.json" ]]; then
    hf download "$id" --revision "$revision" --local-dir "$out"
  fi
  [[ -f "$out/config.json" ]] || die "download did not produce $out/config.json"
}

download() {
  preflight
  mkdir -p "$MODEL_DIR" "$DFLASH_DIR"
  download_one "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR"
  download_one "$DFLASH_ID" "main" "$DFLASH_DIR"
  remote "mkdir -p '$REMOTE_WORK_DIR/models/glm-5.3-flash-exl3-tr3-4bpw' '$REMOTE_WORK_DIR/models/glm-5.3-flash-dflash2'"
  rsync -a --delete --info=progress2 "$MODEL_DIR/" "$REMOTE_HOST:$REMOTE_WORK_DIR/models/glm-5.3-flash-exl3-tr3-4bpw/"
  rsync -a --delete --info=progress2 "$DFLASH_DIR/" "$REMOTE_HOST:$REMOTE_WORK_DIR/models/glm-5.3-flash-dflash2/"
  log "model and DFlash2 copied to worker"
}

write_node_script() {
  local path=$1 rank=$2 ip=$3 iface=$4 hca=$5 headless=$6 venv=$7 node_work=$8 node_log=$9 node_model=${10} node_dflash=${11}
  local eager=''
  [[ "$ENFORCE_EAGER" == 1 ]] && eager='--enforce-eager'
  cat > "$path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="$venv/bin:$PATH"
export HF_HOME="$node_work/hf"
export VLLM_CACHE_ROOT="$node_work/vllm-cache"
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_CACHE_PATH="$node_work/cuda-cache"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.80,max_split_size_mb:128"
export GLOO_SOCKET_IFNAME="$iface"
export NCCL_SOCKET_IFNAME="$iface"
export NCCL_IB_HCA="$hca"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_DISABLE="0"
export NCCL_CUMEM_ENABLE="0"
export NCCL_DEBUG="WARN"
export VLLM_HOST_IP="$ip"
export ROCKET_VLLM_META_INIT="$ROCKET_VLLM_META_INIT"
export ROCKET_VLLM_UVA_WEIGHTS="$ROCKET_VLLM_UVA_WEIGHTS"
export SAFETENSORS_STRATEGY="$SAFETENSORS_STRATEGY"
export ROCKET_SKIP_VLLM_KERNEL_WARMUP=1
export VLLM_USE_V1=1
export VLLM_TORCH_PROFILER_DIR="$node_work/profile"
# Qwen38-cluster transfer (same GB10 hardware): pin the 10 fast X925 cores
# (5-9,15-19 @ 3.9GHz; 0-4,10-14 are 2.8GHz A725) and size OMP to the pin.
# Recipe measured +2-3% at every concurrency from the pin alone.
# CPUSET="" disables the pin (applies to this rank's server process).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CPUSET="${CPUSET:-5-9,15-19}"
mkdir -p "$node_work/cuda-cache" "$node_work/vllm-cache" "$node_log"
export DFLASH_DIR="$node_dflash"
export DFLASH_TOKENS="$DFLASH_TOKENS"
export DFLASH_DRAFT_TP="$DFLASH_DRAFT_TP"
export DFLASH_SCHEDULE='$DFLASH_SCHEDULE'
export GLM53_MIXED_PREFILL_CHUNK="$GLM53_MIXED_PREFILL_CHUNK"
export SPEC_ENABLE="${SPEC_ENABLE:-1}"
export GLM53_DENSE_FP8="${GLM53_DENSE_FP8:-off}"
SPEC_JSON=\$(python3 -c 'import json,os; print(json.dumps({"method":"dflash","model":os.environ["DFLASH_DIR"],"num_speculative_tokens":int(os.environ["DFLASH_TOKENS"]),"draft_tensor_parallel_size":int(os.environ["DFLASH_DRAFT_TP"]),"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","lazy_draft":True,"num_speculative_tokens_per_batch_size":json.loads(os.environ["DFLASH_SCHEDULE"])}))' )
ARGS=(
  "$node_model"
  --served-model-name GLM-5.3-Flash-EXL3
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size 2 --nnodes 2 --node-rank "$rank"
  --master-addr "$HEAD_IP" --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --quantization exl3
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --load-format safetensors --safetensors-load-strategy "$SAFETENSORS_STRATEGY"
  --max-parallel-loading-workers 1
  --no-enable-flashinfer-autotune
  ${PREFIX_CACHING_FLAG:---enable-prefix-caching}
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  $eager
)
if [[ "${SPEC_ENABLE:-1}" == 1 ]]; then ARGS+=(--speculative-config "\$SPEC_JSON"); fi
if [[ "$ENFORCE_EAGER" != 1 ]]; then ARGS+=(--cudagraph-capture-sizes $CUDAGRAPH_CAPTURE_SIZES); fi
if [[ -n "$PROFILER_ENABLE" ]]; then ARGS+=(--profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$node_work/profile\", \"torch_profiler_with_stack\": true, \"torch_profiler_record_shapes\": true}"); fi
if [[ "$headless" == 1 ]]; then ARGS+=(--headless); fi
if [[ "${ASYNC_SCHEDULING:-0}" == 1 ]]; then ARGS+=(--async-scheduling); fi
if [[ -n "${CPUSET:-}" ]]; then
  exec taskset -c "${CPUSET:-5-9,15-19}" "$venv/bin/vllm" serve "\${ARGS[@]}"
else
  exec "$venv/bin/vllm" serve "\${ARGS[@]}"
fi
EOF
  chmod +x "$path"
}

launch() {
  [[ "${GLM53_ALLOW_LAUNCH:-0}" == 1 ]] || die "launch is guarded; set GLM53_ALLOW_LAUNCH=1 after reviewing the boot profile"
  preflight
  require_path "$VENV/bin/vllm"
  require_path "$MODEL_DIR/config.json"
  require_path "$DFLASH_DIR/config.json"
  prepare
  if [[ "$DROP_CACHES" == 1 ]]; then
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || die "DROP_CACHES=1 needs passwordless sudo"
    remote "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" || die "worker cache drop failed"
  fi
  write_node_script "$WORK_DIR/worker.sh" 1 "$WORKER_IP" "$WORKER_IFACE" "$WORKER_HCA" 1 "$REMOTE_VENV" "$REMOTE_WORK_DIR" "$REMOTE_WORK_DIR/logs" "$REMOTE_WORK_DIR/models/glm-5.3-flash-exl3-tr3-4bpw" "$REMOTE_WORK_DIR/models/glm-5.3-flash-dflash2"
  write_node_script "$WORK_DIR/head.sh" 0 "$HEAD_IP" "$HEAD_IFACE" "$HEAD_HCA" 0 "$VENV" "$WORK_DIR" "$LOG_DIR" "$MODEL_DIR" "$DFLASH_DIR"
  mkdir -p "$LOG_DIR"
  scp -q "$WORK_DIR/worker.sh" "$REMOTE_HOST:$REMOTE_WORK_DIR/worker.sh"
  remote "nohup '$REMOTE_WORK_DIR/worker.sh' >'$REMOTE_WORK_DIR/logs/worker.log' 2>&1 & echo \$! >'$REMOTE_WORK_DIR/worker.pid'"
  nohup "$WORK_DIR/head.sh" >"$LOG_DIR/head.log" 2>&1 & echo $! >"$WORK_DIR/head.pid"
  log "started head and worker without Docker"
  log "head log: $LOG_DIR/head.log"
  log "worker log: $REMOTE_HOST:$REMOTE_WORK_DIR/logs/worker.log"
}

stop() {
  if [[ -f "$WORK_DIR/head.pid" ]]; then kill "$(<"$WORK_DIR/head.pid")" 2>/dev/null || true; fi
  remote "if [[ -f '$REMOTE_WORK_DIR/worker.pid' ]]; then kill \"\$(cat '$REMOTE_WORK_DIR/worker.pid')\" 2>/dev/null || true; fi"
  log "stop requested for Rocket-owned ranks"
}

status() {
  if [[ -f "$WORK_DIR/head.pid" ]] && kill -0 "$(<"$WORK_DIR/head.pid")" 2>/dev/null; then
    log "head running pid=$(<"$WORK_DIR/head.pid")"
  else
    log "head stopped"
  fi
  remote "if [[ -f '$REMOTE_WORK_DIR/worker.pid' ]] && kill -0 \"\$(cat '$REMOTE_WORK_DIR/worker.pid')\" 2>/dev/null; then echo '[glm53-exl3-host] worker running'; else echo '[glm53-exl3-host] worker stopped'; fi"
  curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null || true
}

case "$ACTION" in
  preflight) preflight ;;
  prepare) prepare ;;
  install-local) install_local ;;
  install) install ;;
  download) download ;;
  launch) launch ;;
  stop) stop ;;
  status) status ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
