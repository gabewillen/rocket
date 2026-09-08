#!/usr/bin/env bash
# Two-node Qwen3.8 expanded-attention calibration launcher.
#
# The default action prepares and verifies every overlay. --launch is required
# before either GPU or an existing container can be touched.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

IMAGE_TAG="vllm/vllm-openai:qwen38-flash-next"
IMAGE_ID="sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a"
IMAGE_REPO_DIGEST="vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
MIA_REPOSITORY="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks.git"
MIA_COMMIT="c2325b22602b51a5faf55fc2bebccc34f3f80b9f"
MODEL_ID="nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_REVISION="fc694b54fb0174e0913e6adf86691ef85a4ead47"
MODEL_CACHE_NAME="models--nvidia--Qwen3.8-Flash-Next-NVFP4"
CONTAINER_MODEL_DIR="/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia"
CONTAINER_VLLM_DIR="/usr/local/lib/python3.12/dist-packages/vllm"

HEAD_IP="192.168.100.10"
WORKER_IP="192.168.100.11"
WORKER_USER="glwillen"
HEAD_IFACE="enp1s0f1np1"
WORKER_IFACE="enp1s0f1np1"
HEAD_HCA="rocep1s0f1"
WORKER_HCA="rocep1s0f1"
GID_INDEX="3"
MASTER_PORT="50000"
API_PORT="8888"
HEAD_CONTAINER="rocket-qwen38-calibration-head"
WORKER_CONTAINER="rocket-qwen38-calibration-worker"
HF_CACHE="${HOME}/.cache/huggingface"
WORKER_HF_VOLUME="vllm-fn-hf"
WORKER_HF_CACHE=""
WORKER_CACHE_KIND="docker_volume"
WORKER_CACHE_MOUNT="$WORKER_HF_VOLUME"
CHECKPOINT_MANIFEST_SHA256=""
CHECKPOINT_SHARD_COUNT=0
HEAD_CACHE_FILESYSTEM=""
WORKER_CACHE_FILESYSTEM=""
WORKER_SNAPSHOT=""
HEAD_RUNTIME_CACHE_MOUNT="$HF_CACHE"
WORKER_RUNTIME_CACHE_MOUNT="$WORKER_HF_VOLUME"
USE_IMMUTABLE_CACHE_VIEW=false
OUTPUT_DIR=""
MIA_SOURCE=""
FP8_ARTIFACT_DIR=""
NVFP4_ARTIFACT_DIR=""
NVFP4_OVERLAY_FILE=""
NVFP4_EXPECTED_COUNT=""
NVFP4_FAMILIES=()
NVFP4_FAMILIES_CSV=""
NVFP4_HAS_BASE_ROUTERS=false
LAUNCH=false
KEEP_RUNNING=false
PRODUCTION=false
ORACLE_K0=false
TWO_NODE_PREFLIGHT=false
STARTUP_TIMEOUT_SECONDS=3600
GPU_MEMORY_UTILIZATION="0.835"
MTP_DEPTH="3"

usage() {
    cat <<'EOF'
Usage:
  scripts/numerics/qwen38-expanded-calibration.sh --output-dir ABSOLUTE_PATH [options]

Default: prepare and verify all pinned overlays without launching the model.

Required:
  --output-dir DIR       Empty, durable output directory (must be absolute)

Options:
  --launch               Launch both nodes, run the corpus, reduce telemetry
  --two-node-preflight   Transfer and verify both nodes without launching
  --production           Launch without telemetry/eager mode and benchmark throughput
  --production-preflight Prepare/deploy production launch scripts without launching
  --oracle-k0            Capture one target-only K0 coding-prompt oracle
  --keep-running         Leave containers running after a successful calibration
  --mia-source DIR       Existing checkout at the pinned MiaAI-Lab commit
  --fp8-artifact-dir DIR Immutable linear-attention FP8 artifact directory
  --nvfp4-artifact-dir DIR Immutable manifest-driven NVFP4 artifact directory
  --worker USER@HOST     Worker SSH destination (default: glwillen@192.168.100.11)
  --head-ip IP           Head fabric address (default: 192.168.100.10)
  --worker-ip IP         Worker fabric address (default: 192.168.100.11)
  --head-iface NAME      Head fabric interface (default: enp1s0f1np1)
  --worker-iface NAME    Worker fabric interface (default: enp1s0f1np1)
  --head-hca NAME        Head RDMA HCA (default: rocep1s0f1)
  --worker-hca NAME      Worker RDMA HCA (default: rocep1s0f1)
  --hf-cache DIR         Head Hugging Face cache
  --worker-hf-volume V   Worker read-only HF volume (default: vllm-fn-hf)
  --worker-hf-cache DIR  Worker host HF cache; require matching ext4 checkpoint
  --startup-timeout-seconds N
                         Health deadline in seconds (default: 3600)
  --gpu-memory-utilization F
                         vLLM device-memory fraction in (0,1] (default: 0.835)
  --mtp-depth K          Fixed MTP depth 1..7 (default: 3); K0 fails closed
  --help                 Show this help
EOF
}

fail() {
    write_oracle_failure "${CURRENT_PHASE:-launcher}" "$*"
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

write_oracle_failure() {
    local phase=${1:-launcher} reason=${2:-unknown}
    if [[ "${ORACLE_K0:-false}" == true && -n "${OUTPUT_DIR:-}" && -d "${OUTPUT_DIR:-}" ]]; then
        python3 - "$OUTPUT_DIR/oracle-failure.json" "$phase" "$reason" <<'PY' || true
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.exists():
    raise SystemExit(0)
identity_path = path.with_name("oracle-identity.json")
identity = json.loads(identity_path.read_text()) if identity_path.is_file() else {}
capture = path.with_name("capture")
record = {
    "schema": "rocket.qwen38.k0-target-oracle-failure.v1",
    "valid": False,
    "complete": False,
    "phase": sys.argv[2][:128],
    "reason": sys.argv[3][:1024],
    "identity": identity,
    "completed": sorted(item.name for item in capture.glob("*.bin"))[:51] if capture.is_dir() else [],
}
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
    fi
}

while (($#)); do
    case "$1" in
        --output-dir) OUTPUT_DIR=${2:?missing value}; shift 2 ;;
        --launch) LAUNCH=true; shift ;;
        --two-node-preflight) TWO_NODE_PREFLIGHT=true; shift ;;
        --production) PRODUCTION=true; LAUNCH=true; shift ;;
        --oracle-k0) ORACLE_K0=true; MTP_DEPTH=0; shift ;;
        --production-preflight)
            PRODUCTION=true
            TWO_NODE_PREFLIGHT=true
            LAUNCH=false
            shift
            ;;
        --keep-running) KEEP_RUNNING=true; shift ;;
        --mia-source) MIA_SOURCE=${2:?missing value}; shift 2 ;;
        --fp8-artifact-dir) FP8_ARTIFACT_DIR=${2:?missing value}; shift 2 ;;
        --nvfp4-artifact-dir) NVFP4_ARTIFACT_DIR=${2:?missing value}; shift 2 ;;
        --worker)
            worker_arg=${2:?missing value}
            [[ "$worker_arg" == *@* ]] || fail "--worker must be USER@HOST"
            WORKER_USER=${worker_arg%@*}
            WORKER_IP=${worker_arg#*@}
            shift 2
            ;;
        --head-ip) HEAD_IP=${2:?missing value}; shift 2 ;;
        --worker-ip) WORKER_IP=${2:?missing value}; shift 2 ;;
        --head-iface) HEAD_IFACE=${2:?missing value}; shift 2 ;;
        --worker-iface) WORKER_IFACE=${2:?missing value}; shift 2 ;;
        --head-hca) HEAD_HCA=${2:?missing value}; shift 2 ;;
        --worker-hca) WORKER_HCA=${2:?missing value}; shift 2 ;;
        --hf-cache) HF_CACHE=${2:?missing value}; shift 2 ;;
        --worker-hf-volume) WORKER_HF_VOLUME=${2:?missing value}; shift 2 ;;
        --worker-hf-cache) WORKER_HF_CACHE=${2:?missing value}; shift 2 ;;
        --startup-timeout-seconds)
            STARTUP_TIMEOUT_SECONDS=${2:?missing value}
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION=${2:?missing value}
            shift 2
            ;;
        --mtp-depth) MTP_DEPTH=${2:?missing value}; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ -n "$OUTPUT_DIR" ]] || fail "--output-dir is required"
[[ "$OUTPUT_DIR" == /* ]] || fail "--output-dir must be absolute"
if [[ -n "$WORKER_HF_CACHE" ]]; then
    [[ "$WORKER_HF_CACHE" == /* ]] || fail "--worker-hf-cache must be absolute"
    WORKER_CACHE_KIND="host_ext4"
    WORKER_CACHE_MOUNT="$WORKER_HF_CACHE"
    USE_IMMUTABLE_CACHE_VIEW=true
fi
[[ "$MTP_DEPTH" =~ ^[0-7]$ ]] || fail "--mtp-depth must be one of 0,1,2,3,4,5,6,7"
# Pinned image d464f3b4 declares SpeculativeConfig.num_speculative_tokens with
# Pydantic Field(gt=0). Omitting speculative_config also omits the MTP model and
# its cache path, so neither CLI shape represents K0 with synchronized MTP state.
[[ "$MTP_DEPTH" != 0 || "$ORACLE_K0" == true ]] || fail "K0 is only available through --oracle-k0"
[[ "$ORACLE_K0" != true || "$PRODUCTION" != true ]] || fail "--oracle-k0 and --production are mutually exclusive"
[[ "$ORACLE_K0" != true || -n "$NVFP4_ARTIFACT_DIR" ]] || fail "--oracle-k0 requires the accepted --nvfp4-artifact-dir"
[[ "$ORACLE_K0" != true || "$KEEP_RUNNING" != true ]] || fail "--oracle-k0 requires exact cleanup and rejects --keep-running"
[[ "$ORACLE_K0" != true || -n "$WORKER_HF_CACHE" ]] || fail "--oracle-k0 requires --worker-hf-cache for the read-only rank1 source mount"
# The pinned Qwen path reuses its one MTP layer. Its GDN and PLE cache shapes add
# num_speculative_tokens to their convolution history, while the model weights,
# model revision, and persistent recurrent-state shapes remain unchanged. K7's
# eight-token decode query is the largest shape covered by this launcher.
[[ "$STARTUP_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    fail "--startup-timeout-seconds must be a positive integer"
if ! GPU_MEMORY_UTILIZATION=$(python3 - "$GPU_MEMORY_UTILIZATION" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
if not math.isfinite(value) or not 0.0 < value <= 1.0:
    raise SystemExit(1)
print(format(value, ".15g"))
PY
); then
    fail "--gpu-memory-utilization must be a finite number in (0,1]"
fi
if [[ -n "$FP8_ARTIFACT_DIR" ]]; then
    [[ "$FP8_ARTIFACT_DIR" == /* ]] || fail "--fp8-artifact-dir must be absolute"
    [[ -d "$FP8_ARTIFACT_DIR" ]] || fail "FP8 artifact directory missing: $FP8_ARTIFACT_DIR"
    FP8_ARTIFACT_DIR=$(cd "$FP8_ARTIFACT_DIR" && pwd)
    fp8_artifact_key=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_key"])' \
        "$FP8_ARTIFACT_DIR/manifest.json") || fail "FP8 artifact manifest is unreadable"
    [[ "$(basename "$FP8_ARTIFACT_DIR")" == "$fp8_artifact_key" ]] || \
        fail "FP8 artifact directory is not keyed by its manifest"
fi
if [[ -n "$FP8_ARTIFACT_DIR" && -n "$NVFP4_ARTIFACT_DIR" ]]; then
    fail "choose at most one linear-attention precision artifact"
fi
if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
    [[ "$NVFP4_ARTIFACT_DIR" == /* ]] || fail "--nvfp4-artifact-dir must be absolute"
    [[ -d "$NVFP4_ARTIFACT_DIR" ]] || fail "NVFP4 artifact directory missing: $NVFP4_ARTIFACT_DIR"
    NVFP4_ARTIFACT_DIR=$(cd "$NVFP4_ARTIFACT_DIR" && pwd)
    nvfp4_artifact_key=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_key"])' \
        "$NVFP4_ARTIFACT_DIR/manifest.json") || fail "NVFP4 artifact manifest is unreadable"
    [[ "$(basename "$NVFP4_ARTIFACT_DIR")" == "$nvfp4_artifact_key" ]] || \
        fail "NVFP4 artifact directory is not keyed by its manifest"
    read_nvfp4_manifest() {
        python3 - "$NVFP4_ARTIFACT_DIR/manifest.json" "$1" <<'PY'
import json
import pathlib
import sys

manifest = json.load(open(sys.argv[1]))
schema = manifest.get("schema")
source = manifest.get("source")
overlay = manifest.get("overlay")
if not isinstance(source, dict) or not isinstance(overlay, dict):
    raise SystemExit("NVFP4 manifest source/overlay contract missing")
if schema == "rocket.qwen38.linear-nvfp4-overlay.v1":
    families = ["linear_attention"]
elif schema == "rocket.qwen38.nvfp4-overlay.v2":
    families = source.get("families")
else:
    raise SystemExit("NVFP4 manifest schema mismatch")
contracts = {
    "base_ple": 2,
    "base_routers": 48,
    "full_attention": 48,
    "linear_attention": 180,
}
if not isinstance(families, list) or not families:
    raise SystemExit("NVFP4 manifest family selection is invalid")
if families != sorted(set(families)) or any(family not in contracts for family in families):
    raise SystemExit("NVFP4 manifest family selection is invalid")
expected = sum(contracts[family] for family in families)
if not isinstance(source.get("tensors"), list) or len(source["tensors"]) != expected:
    raise SystemExit("NVFP4 manifest source tensor count mismatch")
filename = overlay.get("file")
if not isinstance(filename, str) or pathlib.Path(filename).name != filename:
    raise SystemExit("NVFP4 manifest overlay file is unsafe")
field = sys.argv[2]
if field == "file":
    print(filename)
elif field == "count":
    print(expected)
elif field == "families":
    print("\n".join(families))
else:
    raise SystemExit("unknown NVFP4 manifest field")
PY
    }
    NVFP4_OVERLAY_FILE=$(read_nvfp4_manifest file) || fail "NVFP4 artifact manifest is invalid"
    NVFP4_EXPECTED_COUNT=$(read_nvfp4_manifest count) || fail "NVFP4 artifact manifest is invalid"
    mapfile -t NVFP4_FAMILIES < <(read_nvfp4_manifest families)
    [[ ${#NVFP4_FAMILIES[@]} -gt 0 ]] || fail "NVFP4 artifact has no selected families"
    NVFP4_FAMILIES_CSV=$(IFS=,; printf '%s' "${NVFP4_FAMILIES[*]}")
    for family in "${NVFP4_FAMILIES[@]}"; do
        if [[ "$family" == base_routers ]]; then
            NVFP4_HAS_BASE_ROUTERS=true
        fi
    done
    [[ -f "$NVFP4_ARTIFACT_DIR/$NVFP4_OVERLAY_FILE" ]] || \
        fail "NVFP4 overlay payload missing: $NVFP4_OVERLAY_FILE"
fi
if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "--output-dir must be empty: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/artifacts" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/work"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
ARTIFACT_DIR="$OUTPUT_DIR/artifacts"
LOG_DIR="$OUTPUT_DIR/logs"
WORK_DIR="$OUTPUT_DIR/work"
SSH_TARGET="${WORKER_USER}@${WORKER_IP}"

head_log_pid=""
worker_log_pid=""
head_workload_log_pid=""
worker_workload_log_pid=""
hardware_monitor_pid=""
head_container_started=false
worker_container_started=false
worker_scratch_created=false
REMOTE_OUTPUT=""
CURRENT_PHASE="cpu_preflight"

cleanup() {
    local status=${1:-0}
    set +e
    [[ -n "$head_log_pid" ]] && kill "$head_log_pid" >/dev/null 2>&1
    [[ -n "$worker_log_pid" ]] && kill "$worker_log_pid" >/dev/null 2>&1
    [[ -n "$head_workload_log_pid" ]] && kill "$head_workload_log_pid" >/dev/null 2>&1
    [[ -n "$worker_workload_log_pid" ]] && kill "$worker_workload_log_pid" >/dev/null 2>&1
    [[ -n "$hardware_monitor_pid" ]] && kill "$hardware_monitor_pid" >/dev/null 2>&1
    [[ -n "$hardware_monitor_pid" ]] && wait "$hardware_monitor_pid" 2>/dev/null
    if [[ "$KEEP_RUNNING" != true ]]; then
        [[ "$head_container_started" == true ]] && docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1
        if [[ "$worker_container_started" == true ]]; then
            ssh -o BatchMode=yes "$SSH_TARGET" \
                "docker rm -f '$WORKER_CONTAINER' >/dev/null 2>&1" >/dev/null 2>&1
        fi
    fi
    if [[ "$worker_scratch_created" == true ]]; then
        ssh -o BatchMode=yes "$SSH_TARGET" \
            "case '$REMOTE_OUTPUT' in /dev/shm/rocket-qwen38-k0-*) find '$REMOTE_OUTPUT' -depth -delete ;; *) exit 91 ;; esac" \
            >/dev/null 2>&1
    fi
    return "$status"
}

on_shell_error() {
    local status=$?
    trap - ERR EXIT INT TERM
    write_oracle_failure "$CURRENT_PHASE" "command failed with status $status"
    cleanup "$status"
    exit "$status"
}

on_shell_exit() {
    local status=$?
    trap - ERR EXIT INT TERM
    if ((status != 0)); then
        write_oracle_failure "$CURRENT_PHASE" "shell exited with status $status"
    fi
    cleanup "$status"
    exit "$status"
}

on_shell_signal() {
    local signal=$1 status
    case "$signal" in
        INT) status=130 ;;
        TERM) status=143 ;;
        *) status=1 ;;
    esac
    trap - ERR EXIT INT TERM
    write_oracle_failure "$CURRENT_PHASE" "received SIG$signal"
    cleanup "$status"
    exit "$status"
}

run_checked() {
    CURRENT_PHASE=$1
    shift
    "$@"
}

set -E
trap on_shell_error ERR
trap on_shell_exit EXIT
trap 'on_shell_signal INT' INT
trap 'on_shell_signal TERM' TERM

command -v docker >/dev/null || fail "docker is required"
command -v python3 >/dev/null || fail "python3 is required"
command -v ssh >/dev/null || fail "ssh is required"
command -v scp >/dev/null || fail "scp is required"
command -v sha256sum >/dev/null || fail "sha256sum is required"

check_port_available() {
    local host=$1 port=$2
    python3 - "$host" "$port" <<'PY'
import socket
import sys

host, raw_port = sys.argv[1:]
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind((host, int(raw_port)))
PY
}

actual_image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG" 2>/dev/null || true)
[[ "$actual_image_id" == "$IMAGE_ID" ]] || fail \
    "head image mismatch: expected $IMAGE_ID, got ${actual_image_id:-missing}"
remote_image_id=$(ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker image inspect --format '{{.Id}}' '$IMAGE_TAG' 2>/dev/null" || true)
[[ "$remote_image_id" == "$IMAGE_ID" ]] || fail \
    "worker image mismatch: expected $IMAGE_ID, got ${remote_image_id:-missing}"
actual_image_digest=$(docker image inspect --format '{{join .RepoDigests "\n"}}' "$IMAGE_TAG" | grep -Fx "$IMAGE_REPO_DIGEST" || true)
[[ "$actual_image_digest" == "$IMAGE_REPO_DIGEST" ]] || fail "head image repo digest mismatch"
remote_image_digest=$(ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker image inspect --format '{{join .RepoDigests \"\\n\"}}' '$IMAGE_TAG'" | grep -Fx "$IMAGE_REPO_DIGEST" || true)
[[ "$remote_image_digest" == "$IMAGE_REPO_DIGEST" ]] || fail "worker image repo digest mismatch"

head_page_size=$(getconf PAGESIZE)
worker_page_size=$(ssh -o BatchMode=yes "$SSH_TARGET" getconf PAGESIZE)
[[ "$head_page_size" == 65536 ]] || fail "head page size is $head_page_size, expected 65536"
[[ "$worker_page_size" == 65536 ]] || fail "worker page size is $worker_page_size, expected 65536"

HEAD_SNAPSHOT="$HF_CACHE/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
[[ -f "$HEAD_SNAPSHOT/config.json" ]] || fail "head checkpoint revision missing: $HEAD_SNAPSHOT"
HEAD_CACHE_FILESYSTEM=$(findmnt -T "$HEAD_SNAPSHOT" -n -o FSTYPE)
if [[ "$WORKER_CACHE_KIND" == host_ext4 ]]; then
    [[ "$HEAD_CACHE_FILESYSTEM" == ext4 ]] || \
        fail "head checkpoint cache must be ext4 for local-cache control"
    WORKER_SNAPSHOT="$WORKER_HF_CACHE/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
    printf -v worker_snapshot_q '%q' "$WORKER_SNAPSHOT"
    WORKER_CACHE_FILESYSTEM=$(ssh -o BatchMode=yes "$SSH_TARGET" \
        "findmnt -T $worker_snapshot_q -n -o FSTYPE" 2>/dev/null || true)
    [[ "$WORKER_CACHE_FILESYSTEM" == ext4 ]] || \
        fail "worker checkpoint cache must exist on ext4, got ${WORKER_CACHE_FILESYSTEM:-missing}"
    head_safetensor_count=$(find "$HEAD_SNAPSHOT" -maxdepth 1 -name '*.safetensors' | wc -l)
    worker_safetensor_count=$(ssh -o BatchMode=yes "$SSH_TARGET" \
        "find $worker_snapshot_q -maxdepth 1 -name '*.safetensors' | wc -l")
    [[ "$head_safetensor_count" == 11 && "$worker_safetensor_count" == 11 ]] || \
        fail "pinned checkpoint requires 11 safetensor shards on each node"
    CHECKPOINT_SHARD_COUNT=11
    manifest_program='import hashlib,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=[f"{x.name}\t{x.resolve().name}\t{x.stat().st_size}\n" for x in sorted(p.glob("*.safetensors"))]; print(hashlib.sha256("".join(rows).encode()).hexdigest())'
    head_manifest=$(python3 -c "$manifest_program" "$HEAD_SNAPSHOT")
    worker_manifest=$(ssh -o BatchMode=yes "$SSH_TARGET" \
        "python3 -c $(printf '%q' "$manifest_program") $worker_snapshot_q")
    [[ -n "$head_manifest" && "$worker_manifest" == "$head_manifest" ]] || \
        fail "worker local checkpoint manifest differs from head"
    CHECKPOINT_MANIFEST_SHA256="$head_manifest"
else
    ssh -o BatchMode=yes "$SSH_TARGET" "docker volume inspect '$WORKER_HF_VOLUME' >/dev/null" || \
        fail "worker Hugging Face volume missing: $WORKER_HF_VOLUME"
    ssh -o BatchMode=yes "$SSH_TARGET" \
        "docker run --rm -v '$WORKER_HF_VOLUME:/cache:ro' --entrypoint /bin/sh '$IMAGE_TAG' -c 'test -f /cache/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/model.safetensors.index.json'" || \
        fail "worker volume $WORKER_HF_VOLUME does not contain pinned checkpoint metadata"
fi

MIA_WORK="$WORK_DIR/mia"
if [[ -n "$MIA_SOURCE" ]]; then
    [[ -d "$MIA_SOURCE/.git" ]] || fail "--mia-source is not a Git checkout: $MIA_SOURCE"
    [[ "$(git -C "$MIA_SOURCE" rev-parse HEAD)" == "$MIA_COMMIT" ]] || \
        fail "--mia-source is not at $MIA_COMMIT"
    mkdir -p "$MIA_WORK"
    git -C "$MIA_SOURCE" archive "$MIA_COMMIT" | tar -x -C "$MIA_WORK"
else
    GIT_TERMINAL_PROMPT=0 git clone --quiet --filter=blob:none "$MIA_REPOSITORY" "$MIA_WORK"
    git -C "$MIA_WORK" checkout --quiet --detach "$MIA_COMMIT"
fi
[[ "$(git -C "$MIA_WORK" rev-parse HEAD 2>/dev/null || printf '%s' "$MIA_COMMIT")" == "$MIA_COMMIT" ]] || \
    fail "prepared MiaAI-Lab source is not at $MIA_COMMIT"

extract_image_file() {
    local container_path=$1 destination=$2 container_id
    container_id=$(docker create "$IMAGE_TAG")
    docker cp "$container_id:$container_path" "$destination"
    docker rm "$container_id" >/dev/null
}

# Generate every runtime overlay from pinned source. The source checkout and HF
# cache remain untouched.
RUNTIME_FILES="$WORK_DIR/runtime/files"
mkdir -p "$RUNTIME_FILES"
for generator in patch_ple_layer.py patch_modelopt_mxfp8.py \
    patch_qsa_fp8_kv.py patch_checkpoint_config.py; do
    cp "$MIA_WORK/files/$generator" "$RUNTIME_FILES/$generator"
done
extract_image_file "$CONTAINER_MODEL_DIR/ple_layer.py" "$RUNTIME_FILES/ple_layer_patched.py.orig"
extract_image_file "$CONTAINER_VLLM_DIR/model_executor/layers/quantization/modelopt.py" \
    "$RUNTIME_FILES/modelopt_patched.py.orig"
extract_image_file "$CONTAINER_MODEL_DIR/ops/qsa.py" "$RUNTIME_FILES/qsa_ops_patched.py.orig"
extract_image_file "$CONTAINER_MODEL_DIR/qsa.py" "$RUNTIME_FILES/qsa_nvidia_patched.py.orig"
extract_image_file "$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py" \
    "$ARTIFACT_DIR/weight_utils_64k.py"
extract_image_file "$CONTAINER_VLLM_DIR/platforms/interface.py" \
    "$ARTIFACT_DIR/platform_qsa_patched.py"
extract_image_file "$CONTAINER_MODEL_DIR/model.py" "$ARTIFACT_DIR/model_router.py"

python3 "$RUNTIME_FILES/patch_ple_layer.py" >/dev/null
python3 "$RUNTIME_FILES/patch_modelopt_mxfp8.py" >/dev/null
python3 "$SCRIPT_DIR/patch-qwen38-modelopt-fp8-block-moe.py" \
    "$RUNTIME_FILES/modelopt_patched.py"
python3 "$RUNTIME_FILES/patch_qsa_fp8_kv.py" >/dev/null
python3 "$RUNTIME_FILES/patch_checkpoint_config.py" "$HEAD_SNAPSHOT" "$RUNTIME_FILES" >/dev/null
python3 "$REPO_ROOT/scripts/runtime/patch-vllm-64k-loader.py" "$ARTIFACT_DIR/weight_utils_64k.py"
python3 "$REPO_ROOT/scripts/runtime/patch-qwen38-qsa-page-alignment.py" \
    "$ARTIFACT_DIR/platform_qsa_patched.py"
weight_utils_64k_sha=$(sha256sum "$ARTIFACT_DIR/weight_utils_64k.py" | cut -d' ' -f1)
[[ "$weight_utils_64k_sha" == "6cbca7f793403b0d169e0d8a60f100a4c721d3ec008404eab0ddfa0b81389c0e" ]] || \
    fail "64 KiB loader checksum mismatch: $weight_utils_64k_sha"
if [[ -n "$FP8_ARTIFACT_DIR" ]]; then
    python3 "$REPO_ROOT/scripts/runtime/patch-qwen38-fp8-overlay-loader.py" \
        "$ARTIFACT_DIR/weight_utils_64k.py"
fi
if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
    python3 "$REPO_ROOT/scripts/runtime/patch-qwen38-nvfp4-overlay-loader.py" \
        "$ARTIFACT_DIR/weight_utils_64k.py"
fi
if [[ "$NVFP4_HAS_BASE_ROUTERS" == true ]]; then
    python3 "$REPO_ROOT/scripts/runtime/patch-qwen38-nvfp4-router.py" \
        "$ARTIFACT_DIR/model_router.py"
fi
cp "$ARTIFACT_DIR/model_router.py" "$ARTIFACT_DIR/model_telemetry.py"
python3 "$SCRIPT_DIR/patch-qwen38-activation-telemetry.py" "$ARTIFACT_DIR/model_telemetry.py"
if [[ "${ORACLE_K0:-false}" == true ]]; then
    cp "$ARTIFACT_DIR/model_router.py" "$ARTIFACT_DIR/model_oracle.py"
    python3 "$REPO_ROOT/scripts/runtime/patch-qwen38-k0-oracle.py" "$ARTIFACT_DIR/model_oracle.py"
fi

for file in ple_layer_patched.py modelopt_patched.py qsa_ops_patched.py \
    qsa_nvidia_patched.py config_patched.json hf_quant_config_patched.json; do
    [[ -f "$RUNTIME_FILES/$file" ]] || fail "runtime generator did not produce $file"
    cp "$RUNTIME_FILES/$file" "$ARTIFACT_DIR/$file"
done
if [[ -n "$FP8_ARTIFACT_DIR" ]]; then
    python3 "$REPO_ROOT/scripts/runtime/qwen38-embed-fp8-config.py" \
        "$ARTIFACT_DIR/config_patched.json" \
        "$FP8_ARTIFACT_DIR/hf_quant_config.json" \
        "$ARTIFACT_DIR/config_fp8_patched.json"
fi
if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
    nvfp4_family_args=()
    for family in "${NVFP4_FAMILIES[@]}"; do
        nvfp4_family_args+=(--family "$family")
    done
    python3 "$REPO_ROOT/scripts/runtime/qwen38-embed-fp8-config.py" \
        "$ARTIFACT_DIR/config_patched.json" \
        "$NVFP4_ARTIFACT_DIR/hf_quant_config.json" \
        "$ARTIFACT_DIR/config_nvfp4_patched.json" --quant-algo NVFP4 \
        "${nvfp4_family_args[@]}"
fi

verify_sha() {
    local expected=$1 file=$2 actual
    actual=$(sha256sum "$file" | cut -d' ' -f1)
    [[ "$actual" == "$expected" ]] || fail \
        "overlay checksum mismatch for $(basename "$file"): expected $expected, got $actual"
}
verify_sha fae9fd5242748e8cdb314445a25ad628a0ce335cf26f794623f8679497a65186 "$ARTIFACT_DIR/ple_layer_patched.py"
verify_sha 6b1a1eb03c66dd51e239001be551f60ec1115681b1804e28d732c19b87b75228 "$ARTIFACT_DIR/modelopt_patched.py"
if [[ -z "$FP8_ARTIFACT_DIR" && -z "$NVFP4_ARTIFACT_DIR" ]]; then
    verify_sha 6cbca7f793403b0d169e0d8a60f100a4c721d3ec008404eab0ddfa0b81389c0e "$ARTIFACT_DIR/weight_utils_64k.py"
fi
if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
    NVFP4_CONTAINER_DIR="/rocket/qwen38-linear-nvfp4"
    docker run --rm \
        -v "$HF_CACHE:/root/.cache/huggingface:ro" \
        -v "$NVFP4_ARTIFACT_DIR:$NVFP4_CONTAINER_DIR:ro" \
        -v "$NVFP4_ARTIFACT_DIR/hf_quant_config.json:/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/hf_quant_config.json:ro" \
        -v "$ARTIFACT_DIR/weight_utils_64k.py:$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py:ro" \
        -e "ROCKET_QWEN38_NVFP4_OVERLAY_MANIFEST=$NVFP4_CONTAINER_DIR/manifest.json" \
        -e "ROCKET_QWEN38_NVFP4_QUANT_CONFIG=$NVFP4_CONTAINER_DIR/hf_quant_config.json" \
        --entrypoint /usr/bin/python3 "$IMAGE_TAG" -c \
        "import glob; from vllm.model_executor.model_loader.weight_utils import _rocket_qwen38_nvfp4_overlay_preflight as check; files=glob.glob('/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/*.safetensors'); result=check(sorted(files), 'lazy'); assert len(result['selected']) == $NVFP4_EXPECTED_COUNT; print('validated NVFP4 overlay: $NVFP4_EXPECTED_COUNT tensors')"
    docker run --rm \
        -v "$ARTIFACT_DIR/modelopt_patched.py:$CONTAINER_VLLM_DIR/model_executor/layers/quantization/modelopt.py:ro" \
        -v "$NVFP4_ARTIFACT_DIR/hf_quant_config.json:/work/hf_quant_config.json:ro" \
        -v "$ARTIFACT_DIR/model_telemetry.py:/work/model.py:ro" \
        -v "$ARTIFACT_DIR/model_router.py:/work/model_router.py:ro" \
        -e "ROCKET_NVFP4_FAMILIES=$NVFP4_FAMILIES_CSV" \
        --entrypoint /usr/bin/python3 "$IMAGE_TAG" -c \
        "import json,os,pathlib; from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig; config=ModelOptMixedPrecisionConfig.from_config(json.load(open('/work/hf_quant_config.json'))); families=set(os.environ['ROCKET_NVFP4_FAMILIES'].split(',')); prefix='mtp.layers.48.mlp.experts'; algo=config._resolve_quant_algo(prefix); block=config._fp8_block_scales_config(prefix); assert algo in ('FP8_BLOCK_SCALES', 'FP8_PB_WO'), algo; assert block.weight_block_size == [128, 128], block.weight_block_size; checks={'base_routers':'model.language_model.model.layers.0.mlp.gate','base_ple':'model.language_model.model.layers.1.ple.key_proj'}; [(_ for _ in ()).throw(AssertionError((family, config._resolve_quant_algo(target)))) for family,target in checks.items() if family in families and config._resolve_quant_algo(target) != 'NVFP4']; telemetry=pathlib.Path('/work/model.py').read_text(); runtime=pathlib.Path('/work/model_router.py').read_text(); selected='base_routers' in families; assert ('ROCKET_QWEN38_NVFP4_ROUTER_V1' in telemetry) == selected; assert ('ROCKET_QWEN38_NVFP4_ROUTER_V1' in runtime) == selected; assert 'ROCKET_NVFP4_TELEMETRY' in telemetry; assert 'ROCKET_NVFP4_TELEMETRY' not in runtime; print(f'validated NVFP4 mounted-model semantics: {sorted(families)}; MTP {algo} {block.weight_block_size}')"
fi
verify_sha 0669d6334f58a624c89c15f3e46c90f28e59b0b913507101dec1c5765e3c3b12 "$ARTIFACT_DIR/qsa_ops_patched.py"
verify_sha ee5de40742ad48a6064ea24b99a285ff69c47d57bbb170f57c4eef71567a1df3 "$ARTIFACT_DIR/qsa_nvidia_patched.py"
verify_sha c3864cf981365bfe6b40deaaaa6d12e8402e60a3cef03d009c29f95b4e995403 "$ARTIFACT_DIR/config_patched.json"
verify_sha dd8727422cafbb0257d11a7163442bda46421f6e67c78eb9acd58669cb6eb5f8 "$ARTIFACT_DIR/hf_quant_config_patched.json"

docker run --rm \
    -v "$ARTIFACT_DIR/model_telemetry.py:/work/model.py:ro" \
    -v "$ARTIFACT_DIR/model_router.py:/work/model_router.py:ro" \
    -v "$ARTIFACT_DIR/platform_qsa_patched.py:/work/platform.py:ro" \
    --entrypoint /usr/bin/python3 "$IMAGE_TAG" -m py_compile \
    /work/model.py /work/model_router.py /work/platform.py
if [[ "${ORACLE_K0:-false}" == true ]]; then
    docker run --rm -v "$ARTIFACT_DIR/model_oracle.py:/work/model_oracle.py:ro" \
        --entrypoint /usr/bin/python3 "$IMAGE_TAG" -m py_compile /work/model_oracle.py
    docker run --rm \
        -v "$HF_CACHE:/root/.cache/huggingface:ro" \
        -v "$OUTPUT_DIR:/rocket/output" \
        -v "$REPO_ROOT/scripts/runtime/qwen38-k0-oracle.py:/rocket/qwen38-k0-oracle.py:ro" \
        --entrypoint /usr/bin/python3 "$IMAGE_TAG" \
        /rocket/qwen38-k0-oracle.py prepare \
        --model-dir "/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION" \
        --output /rocket/output/oracle-request.json
    ORACLE_EXPECTED_IDS=$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["input_token_ids"],separators=(",",":")))' "$OUTPUT_DIR/oracle-request.json")
    oracle_request_sha=$(sha256sum "$OUTPUT_DIR/oracle-request.json" | cut -d' ' -f1)
    nvfp4_manifest_sha=$(sha256sum "$NVFP4_ARTIFACT_DIR/manifest.json" | cut -d' ' -f1)
    nvfp4_payload_sha=$(sha256sum "$NVFP4_ARTIFACT_DIR/$NVFP4_OVERLAY_FILE" | cut -d' ' -f1)
    nvfp4_quant_sha=$(sha256sum "$NVFP4_ARTIFACT_DIR/hf_quant_config.json" | cut -d' ' -f1)
    ORACLE_IDENTITY=$(python3 -c 'import json,sys; print(json.dumps({"image_id":sys.argv[1],"image_repo_digest":sys.argv[2],"model":sys.argv[3],"model_revision":sys.argv[4],"mia_commit":sys.argv[5],"overlay_manifest_sha256":sys.argv[6],"overlay_payload_sha256":sys.argv[7],"overlay_quant_config_sha256":sys.argv[8],"request_sha256":sys.argv[9],"generation_index":0,"speculation":"disabled","tensor_parallel_size":2,"node_count":2},sort_keys=True,separators=(",",":")))' "$IMAGE_ID" "$IMAGE_REPO_DIGEST" "$MODEL_ID" "$MODEL_REVISION" "$MIA_COMMIT" "$nvfp4_manifest_sha" "$nvfp4_payload_sha" "$nvfp4_quant_sha" "$oracle_request_sha")
    python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]),indent=2,sort_keys=True))' \
        "$ORACLE_IDENTITY" > "$OUTPUT_DIR/oracle-identity.json"
    docker run --rm --entrypoint /usr/bin/python3 "$IMAGE_TAG" -c \
        "from vllm.platforms import current_platform; current_platform.device_type='cpu'; from vllm.config.cache import CacheConfig; from vllm.entrypoints.openai.cli_args import make_arg_parser; from vllm.utils.argparse_utils import FlexibleArgumentParser; args=make_arg_parser(FlexibleArgumentParser()).parse_args(['$MODEL_ID']); assert args.enable_prefix_caching is None; assert CacheConfig.__dataclass_fields__['enable_prefix_caching'].default is True; assert args.enable_log_requests is False; assert args.speculative_config is None; print('validated pinned K0 CLI parse/defaults: speculation omitted; prefix caching resolves enabled; request logging disabled')"
fi
if [[ -n "$FP8_ARTIFACT_DIR" ]]; then
    FP8_CONTAINER_DIR="/rocket/qwen38-linear-fp8"
    docker run --rm \
        -v "$HF_CACHE:/root/.cache/huggingface:ro" \
        -v "$FP8_ARTIFACT_DIR:$FP8_CONTAINER_DIR:ro" \
        -v "$FP8_ARTIFACT_DIR/hf_quant_config.json:/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/hf_quant_config.json:ro" \
        -v "$ARTIFACT_DIR/weight_utils_64k.py:$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py:ro" \
        -e "ROCKET_QWEN38_FP8_OVERLAY_MANIFEST=$FP8_CONTAINER_DIR/manifest.json" \
        -e "ROCKET_QWEN38_FP8_QUANT_CONFIG=$FP8_CONTAINER_DIR/hf_quant_config.json" \
        --entrypoint /usr/bin/python3 "$IMAGE_TAG" -c \
        "import glob; from vllm.model_executor.model_loader.weight_utils import _rocket_qwen38_fp8_overlay_preflight as check; files=glob.glob('/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/*.safetensors'); result=check(sorted(files), 'lazy'); assert len(result['selected']) == 180; print('validated FP8 overlay: 180 tensors')"
fi

# A host cache cannot accept child bind mounts beneath a read-only parent. Build
# one immutable cache view per node with hard-linked content-addressed blobs and
# the selected patched metadata, then mount that view once as read-only.
if [[ "$WORKER_CACHE_KIND" == host_ext4 ]]; then
    HEAD_RUNTIME_CACHE_MOUNT="$WORK_DIR/hf-cache-view"
    if [[ "$ORACLE_K0" != true ]]; then
        WORKER_RUNTIME_CACHE_MOUNT="$OUTPUT_DIR/work/hf-cache-view"
    fi
    head_cache_device=$(stat -c %d "$HF_CACHE")
    head_output_device=$(stat -c %d "$WORK_DIR")
    [[ "$head_cache_device" == "$head_output_device" ]] || \
        fail "head immutable cache view must share the checkpoint ext4 filesystem"
    mkdir -p "$HEAD_RUNTIME_CACHE_MOUNT/hub"
    cp -al "$HF_CACHE/hub/$MODEL_CACHE_NAME" "$HEAD_RUNTIME_CACHE_MOUNT/hub/"
    head_view_snapshot="$HEAD_RUNTIME_CACHE_MOUNT/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
    rm "$head_view_snapshot/config.json" "$head_view_snapshot/hf_quant_config.json"
    if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
        cp "$ARTIFACT_DIR/config_nvfp4_patched.json" "$head_view_snapshot/config.json"
        cp "$NVFP4_ARTIFACT_DIR/hf_quant_config.json" "$head_view_snapshot/hf_quant_config.json"
    elif [[ -n "$FP8_ARTIFACT_DIR" ]]; then
        cp "$ARTIFACT_DIR/config_fp8_patched.json" "$head_view_snapshot/config.json"
        cp "$FP8_ARTIFACT_DIR/hf_quant_config.json" "$head_view_snapshot/hf_quant_config.json"
    else
        cp "$ARTIFACT_DIR/config_patched.json" "$head_view_snapshot/config.json"
        cp "$ARTIFACT_DIR/hf_quant_config_patched.json" "$head_view_snapshot/hf_quant_config.json"
    fi
    [[ "$(find -L "$head_view_snapshot" -maxdepth 1 -name '*.safetensors' -type f | wc -l)" == 11 ]] || \
        fail "head immutable cache view has unresolved checkpoint shards"
fi
(
    cd "$ARTIFACT_DIR"
    sha256sum ./* > SHA256SUMS
)
cat > "$OUTPUT_DIR/run.json" <<EOF
{"image_id":"$IMAGE_ID","image_tag":"$IMAGE_TAG","image_repo_digest":"$IMAGE_REPO_DIGEST","mia_commit":"$MIA_COMMIT","model":"$MODEL_ID","model_revision":"$MODEL_REVISION","head_page_size":$head_page_size,"worker_page_size":$worker_page_size,"startup_timeout_seconds":$STARTUP_TIMEOUT_SECONDS,"gpu_memory_utilization":$GPU_MEMORY_UTILIZATION,"mtp_depth":$MTP_DEPTH,"oracle_k0":$ORACLE_K0,"worker_cache_kind":"$WORKER_CACHE_KIND","head_cache_filesystem":"$HEAD_CACHE_FILESYSTEM","worker_cache_filesystem":"$WORKER_CACHE_FILESYSTEM","head_snapshot_path":"$HEAD_SNAPSHOT","worker_snapshot_path":"$WORKER_SNAPSHOT","head_runtime_cache_path":"$HEAD_RUNTIME_CACHE_MOUNT","worker_runtime_cache_path":"$WORKER_RUNTIME_CACHE_MOUNT","checkpoint_manifest_sha256":"$CHECKPOINT_MANIFEST_SHA256","checkpoint_safetensor_shards":$CHECKPOINT_SHARD_COUNT}
EOF

printf 'Prepared and verified pinned calibration artifacts in %s\n' "$ARTIFACT_DIR"
if [[ "$LAUNCH" != true && "$TWO_NODE_PREFLIGHT" != true ]]; then
    printf 'Preparation complete. Re-run with a new empty --output-dir and --launch to execute.\n'
    exit 0
fi

# Launch mode starts only after every immutable input and generated overlay has
# passed its identity check.
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d')" ]]; then
    fail "head GPU has a compute tenant"
fi
if [[ -n "$(ssh -o BatchMode=yes "$SSH_TARGET" \
    "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^\$/d'")" ]]; then
    fail "worker GPU has a compute tenant"
fi
docker container inspect "$HEAD_CONTAINER" >/dev/null 2>&1 && fail "head container already exists: $HEAD_CONTAINER"
ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker container inspect '$WORKER_CONTAINER' >/dev/null 2>&1" && \
    fail "worker container already exists: $WORKER_CONTAINER"
check_port_available "$HEAD_IP" "$MASTER_PORT" || \
    fail "head master port is unavailable: $HEAD_IP:$MASTER_PORT"
check_port_available "0.0.0.0" "$API_PORT" || \
    fail "head API port is unavailable: 0.0.0.0:$API_PORT"

REMOTE_OUTPUT="$OUTPUT_DIR"
if [[ "$ORACLE_K0" == true ]]; then
    scratch_key=$(printf '%s\n%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)" "$OUTPUT_DIR" | sha256sum | cut -c1-20)
    REMOTE_OUTPUT="/dev/shm/rocket-qwen38-k0-$scratch_key"
    [[ "$REMOTE_OUTPUT" =~ ^/dev/shm/rocket-qwen38-k0-[0-9a-f]{20}$ ]] || \
        fail "worker scratch identity is unsafe"
    overlay_bytes=$(stat -c %s "$NVFP4_ARTIFACT_DIR/$NVFP4_OVERLAY_FILE")
    artifact_bytes=$(du -sb "$ARTIFACT_DIR" | cut -f1)
    WORKER_SCRATCH_MIN_BYTES=$((overlay_bytes + artifact_bytes + 134217728))
    worker_tmpfs_available=$(ssh -o BatchMode=yes "$SSH_TARGET" \
        "df -B1 --output=avail /dev/shm | tail -1 | tr -d ' '")
    [[ "$worker_tmpfs_available" =~ ^[0-9]+$ && "$worker_tmpfs_available" -ge "$WORKER_SCRATCH_MIN_BYTES" ]] || \
        fail "worker tmpfs capacity ${worker_tmpfs_available:-unknown} is below required $WORKER_SCRATCH_MIN_BYTES bytes"
    WORKER_RUNTIME_CACHE_MOUNT="$REMOTE_OUTPUT/work/hf-cache-view"
    worker_scratch_created=true
    python3 - "$OUTPUT_DIR/run.json" "$REMOTE_OUTPUT" "$WORKER_RUNTIME_CACHE_MOUNT" "$WORKER_SCRATCH_MIN_BYTES" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
record = json.loads(path.read_text())
record["worker_oracle_scratch"] = sys.argv[2]
record["worker_runtime_cache_path"] = sys.argv[3]
record["worker_scratch_min_bytes"] = int(sys.argv[4])
path.write_text(json.dumps(record, sort_keys=True) + "\n")
PY
fi
run_checked "worker_output_prepare" ssh -o BatchMode=yes "$SSH_TARGET" \
    "mkdir -p '$REMOTE_OUTPUT/artifacts' '$REMOTE_OUTPUT/logs' '$REMOTE_OUTPUT/work'"
run_checked "worker_artifact_transfer" scp -q "$ARTIFACT_DIR"/* "$SSH_TARGET:$REMOTE_OUTPUT/artifacts/"
run_checked "worker_artifact_verify" ssh -o BatchMode=yes "$SSH_TARGET" \
    "cd '$REMOTE_OUTPUT/artifacts' && sha256sum --check SHA256SUMS"

write_launch_script_array() {
    local destination=$1 node_rank=$2 node_ip=$3 iface=$4 hca=$5 cache_mount=$6 artifact_dir=$7 mode=$8 cache_access=$9 fp8_host_dir=${10} nvfp4_host_dir=${11}
    local use_immutable_cache_view=${12:-false} container_name="$WORKER_CONTAINER"
    local config_source="$artifact_dir/config_patched.json" quant_source="$artifact_dir/hf_quant_config_patched.json"
    local -a args mode_args
    [[ "$node_rank" == 0 ]] && container_name="$HEAD_CONTAINER"
    read -r -a mode_args <<< "$mode"
    [[ -n "$fp8_host_dir" ]] && config_source="$artifact_dir/config_fp8_patched.json" && quant_source="$fp8_host_dir/hf_quant_config.json"
    [[ -n "$nvfp4_host_dir" ]] && config_source="$artifact_dir/config_nvfp4_patched.json" && quant_source="$nvfp4_host_dir/hf_quant_config.json"
    args=(run -d --name "$container_name" --gpus all --network host --ipc host
        --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864
        --device /dev/infiniband:/dev/infiniband
        -e "GLOO_SOCKET_IFNAME=$iface" -e "NCCL_SOCKET_IFNAME=$iface" -e "TP_SOCKET_IFNAME=$iface"
        -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$hca" -e "NCCL_IB_GID_INDEX=$GID_INDEX"
        -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1 -e "VLLM_HOST_IP=$node_ip" -e HF_HOME=/root/.cache/huggingface)
    if [[ "$PRODUCTION" != true && "$ORACLE_K0" != true ]]; then
        args+=(-e ROCKET_NVFP4_CALIBRATE=1 -e ROCKET_NVFP4_SAMPLE_ELEMENTS=2048
            -e ROCKET_QWEN38_LOAD_TRACE=1 -e ROCKET_NVFP4_MAX_EMISSIONS=12)
    fi
    [[ -n "$fp8_host_dir" ]] && args+=(-v "$fp8_host_dir:/rocket/qwen38-linear-fp8:ro" -e ROCKET_QWEN38_FP8_OVERLAY_MANIFEST=/rocket/qwen38-linear-fp8/manifest.json -e ROCKET_QWEN38_FP8_QUANT_CONFIG=/rocket/qwen38-linear-fp8/hf_quant_config.json)
    [[ -n "$nvfp4_host_dir" ]] && args+=(-v "$nvfp4_host_dir:/rocket/qwen38-linear-nvfp4:ro" -e ROCKET_QWEN38_NVFP4_OVERLAY_MANIFEST=/rocket/qwen38-linear-nvfp4/manifest.json -e ROCKET_QWEN38_NVFP4_QUANT_CONFIG=/rocket/qwen38-linear-nvfp4/hf_quant_config.json)
    if [[ "$ORACLE_K0" == true && "$node_rank" == 0 ]]; then
        args+=(-e ROCKET_QWEN38_K0_ORACLE=1 -e "ROCKET_QWEN38_K0_EXPECTED_IDS=$ORACLE_EXPECTED_IDS"
            -e "ROCKET_QWEN38_K0_IDENTITY=$ORACLE_IDENTITY" -e ROCKET_QWEN38_K0_ORACLE_DIR=/rocket/oracle-root/capture
            -v "$OUTPUT_DIR:/rocket/oracle-root")
    elif [[ "$ORACLE_K0" == true ]]; then
        args+=(-v "$WORKER_HF_CACHE:/rocket/source-hf:ro")
    fi
    args+=(-v "$artifact_dir/ple_layer_patched.py:$CONTAINER_MODEL_DIR/ple_layer.py:ro"
        -v "$artifact_dir/modelopt_patched.py:$CONTAINER_VLLM_DIR/model_executor/layers/quantization/modelopt.py:ro"
        -v "$artifact_dir/weight_utils_64k.py:$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py:ro"
        -v "$artifact_dir/platform_qsa_patched.py:$CONTAINER_VLLM_DIR/platforms/interface.py:ro")
    if [[ "$ORACLE_K0" == true ]]; then
        args+=(-v "$artifact_dir/model_oracle.py:$CONTAINER_MODEL_DIR/model.py:ro")
    elif [[ "$PRODUCTION" == true && "$NVFP4_HAS_BASE_ROUTERS" == true ]]; then
        args+=(-v "$artifact_dir/model_router.py:$CONTAINER_MODEL_DIR/model.py:ro")
    elif [[ "$PRODUCTION" != true ]]; then
        args+=(-v "$artifact_dir/model_telemetry.py:$CONTAINER_MODEL_DIR/model.py:ro")
    fi
    args+=(-v "$artifact_dir/qsa_ops_patched.py:$CONTAINER_MODEL_DIR/ops/qsa.py:ro" -v "$artifact_dir/qsa_nvidia_patched.py:$CONTAINER_MODEL_DIR/qsa.py:ro")
    if [[ "$use_immutable_cache_view" != true ]]; then
        args+=(-v "$config_source:/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/config.json:ro"
            -v "$quant_source:/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/hf_quant_config.json:ro")
    fi
    args+=(-v "$cache_mount:/root/.cache/huggingface:$cache_access" -v "$HOME/.cache/vllm:/root/.cache/vllm"
        "$IMAGE_TAG" "$MODEL_ID" --revision "$MODEL_REVISION" --served-model-name qwen3.8-flash-next
        --tensor-parallel-size 2 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-seqs 16
        --max-num-batched-tokens 8192 --max-model-len 262144 --kv-cache-dtype fp8 --load-format safetensors
        --safetensors-load-strategy lazy --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice
        --tool-call-parser qwen3_coder --distributed-executor-backend mp --mm-encoder-tp-mode data
        --nnodes 2 --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" --enable-expert-parallel
        --all2all-backend allgather_reducescatter)
    [[ "$ORACLE_K0" == true ]] || args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH}")
    args+=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
        --hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}')
    [[ "$PRODUCTION" == true ]] || args+=(--enforce-eager)
    args+=(--node-rank "$node_rank" "${mode_args[@]}")
    {
        printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' 'docker_args=('
        printf '  %q\n' "${args[@]}"
        printf '%s\n' ')' 'if [[ "${1:-}" == "--print-argv" ]]; then' \
            '  exec python3 -c '\''import json,sys; print(json.dumps(sys.argv[1:],separators=(",",":")))'\'' "${docker_args[@]}"' \
            'fi' 'exec docker "${docker_args[@]}"'
    } > "$destination"
    python3 - "$destination.expected-argv.json" "${args[@]}" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], separators=(",", ":")) + "\n")
PY
    chmod +x "$destination"
}

REMOTE_FP8_ARTIFACT=""
if [[ -n "$FP8_ARTIFACT_DIR" ]]; then
    REMOTE_FP8_ARTIFACT="$REMOTE_OUTPUT/fp8-artifact"
    ssh -o BatchMode=yes "$SSH_TARGET" "mkdir -p '$REMOTE_FP8_ARTIFACT'"
    (
        cd "$FP8_ARTIFACT_DIR"
        sha256sum manifest.json hf_quant_config.json linear-attention-fp8.safetensors \
            > "$WORK_DIR/fp8-artifact-SHA256SUMS"
    )
    scp -q "$FP8_ARTIFACT_DIR/manifest.json" "$FP8_ARTIFACT_DIR/hf_quant_config.json" \
        "$FP8_ARTIFACT_DIR/linear-attention-fp8.safetensors" \
        "$WORK_DIR/fp8-artifact-SHA256SUMS" \
        "$SSH_TARGET:$REMOTE_FP8_ARTIFACT/"
    ssh -o BatchMode=yes "$SSH_TARGET" \
        "cd '$REMOTE_FP8_ARTIFACT' && sha256sum --check fp8-artifact-SHA256SUMS"
fi
REMOTE_NVFP4_ARTIFACT=""
if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
    REMOTE_NVFP4_ARTIFACT="$REMOTE_OUTPUT/nvfp4-artifact"
    run_checked "worker_overlay_prepare" ssh -o BatchMode=yes "$SSH_TARGET" "mkdir -p '$REMOTE_NVFP4_ARTIFACT'"
    (
        cd "$NVFP4_ARTIFACT_DIR"
        sha256sum manifest.json hf_quant_config.json "$NVFP4_OVERLAY_FILE" \
            > "$WORK_DIR/nvfp4-artifact-SHA256SUMS"
    )
    run_checked "worker_overlay_transfer" scp -q "$NVFP4_ARTIFACT_DIR/manifest.json" "$NVFP4_ARTIFACT_DIR/hf_quant_config.json" \
        "$NVFP4_ARTIFACT_DIR/$NVFP4_OVERLAY_FILE" \
        "$WORK_DIR/nvfp4-artifact-SHA256SUMS" \
        "$SSH_TARGET:$REMOTE_NVFP4_ARTIFACT/"
    run_checked "worker_overlay_verify" ssh -o BatchMode=yes "$SSH_TARGET" \
        "cd '$REMOTE_NVFP4_ARTIFACT' && sha256sum --check nvfp4-artifact-SHA256SUMS"
    run_checked "worker_overlay_preflight" ssh -o BatchMode=yes "$SSH_TARGET" \
        "docker run --rm \
        -v '$WORKER_CACHE_MOUNT:/root/.cache/huggingface:ro' \
        -v '$REMOTE_NVFP4_ARTIFACT:/rocket/qwen38-linear-nvfp4:ro' \
        -v '$REMOTE_OUTPUT/artifacts/weight_utils_64k.py:$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py:ro' \
        -e ROCKET_QWEN38_NVFP4_OVERLAY_MANIFEST=/rocket/qwen38-linear-nvfp4/manifest.json \
        -e ROCKET_QWEN38_NVFP4_QUANT_CONFIG=/rocket/qwen38-linear-nvfp4/hf_quant_config.json \
        --entrypoint /usr/bin/python3 '$IMAGE_TAG' -c \
        \"import glob; from vllm.model_executor.model_loader.weight_utils import _rocket_qwen38_nvfp4_overlay_preflight as check; files=glob.glob('/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/*.safetensors'); result=check(sorted(files), 'lazy'); assert len(result['selected']) == $NVFP4_EXPECTED_COUNT; print('validated worker NVFP4 overlay: $NVFP4_EXPECTED_COUNT tensors')\""
    run_checked "worker_overlay_semantics" ssh -o BatchMode=yes "$SSH_TARGET" \
        "docker run --rm \
        -v '$REMOTE_OUTPUT/artifacts/modelopt_patched.py:$CONTAINER_VLLM_DIR/model_executor/layers/quantization/modelopt.py:ro' \
        -v '$REMOTE_NVFP4_ARTIFACT/hf_quant_config.json:/work/hf_quant_config.json:ro' \
        -v '$REMOTE_OUTPUT/artifacts/model_telemetry.py:/work/model.py:ro' \
        -v '$REMOTE_OUTPUT/artifacts/model_router.py:/work/model_router.py:ro' \
        -e ROCKET_NVFP4_FAMILIES='$NVFP4_FAMILIES_CSV' \
        --entrypoint /usr/bin/python3 '$IMAGE_TAG' -c \
        \"import json,os,pathlib; from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig; config=ModelOptMixedPrecisionConfig.from_config(json.load(open('/work/hf_quant_config.json'))); families=set(os.environ['ROCKET_NVFP4_FAMILIES'].split(',')); checks={'base_routers':'model.language_model.model.layers.0.mlp.gate','base_ple':'model.language_model.model.layers.1.ple.value_proj'}; [(_ for _ in ()).throw(AssertionError((family, config._resolve_quant_algo(target)))) for family,target in checks.items() if family in families and config._resolve_quant_algo(target) != 'NVFP4']; telemetry=pathlib.Path('/work/model.py').read_text(); runtime=pathlib.Path('/work/model_router.py').read_text(); selected='base_routers' in families; assert ('ROCKET_QWEN38_NVFP4_ROUTER_V1' in telemetry) == selected; assert ('ROCKET_QWEN38_NVFP4_ROUTER_V1' in runtime) == selected; assert 'ROCKET_NVFP4_TELEMETRY' in telemetry; assert 'ROCKET_NVFP4_TELEMETRY' not in runtime; print(f'validated worker NVFP4 mounted-model semantics: {sorted(families)}')\""
fi
if [[ "$WORKER_CACHE_KIND" == host_ext4 ]]; then
    printf -v worker_cache_q '%q' "$WORKER_HF_CACHE"
    printf -v remote_work_q '%q' "$REMOTE_OUTPUT/work"
    printf -v remote_view_q '%q' "$WORKER_RUNTIME_CACHE_MOUNT"
    printf -v remote_model_q '%q' "$WORKER_HF_CACHE/hub/$MODEL_CACHE_NAME"
    printf -v remote_view_snapshot_q '%q' "$WORKER_RUNTIME_CACHE_MOUNT/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
    if [[ "$ORACLE_K0" == true ]]; then
        printf -v remote_source_snapshot_q '%q' "/rocket/source-hf/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
        run_checked "worker_cache_view" ssh -o BatchMode=yes "$SSH_TARGET" \
            "mkdir -p $remote_view_snapshot_q && for source in $worker_snapshot_q/*; do name=\${source##*/}; case \"\$name\" in config.json|hf_quant_config.json) continue ;; esac; ln -s $remote_source_snapshot_q/\"\$name\" $remote_view_snapshot_q/\"\$name\"; done"
    else
        worker_cache_device=$(ssh -o BatchMode=yes "$SSH_TARGET" "stat -c %d $worker_cache_q")
        worker_output_device=$(ssh -o BatchMode=yes "$SSH_TARGET" "stat -c %d $remote_work_q")
        [[ "$worker_cache_device" == "$worker_output_device" ]] || \
            fail "worker immutable cache view must share the checkpoint ext4 filesystem"
        ssh -o BatchMode=yes "$SSH_TARGET" \
            "mkdir -p $remote_view_q/hub && cp -al $remote_model_q $remote_view_q/hub/ && rm $remote_view_snapshot_q/config.json $remote_view_snapshot_q/hf_quant_config.json"
    fi
    if [[ -n "$NVFP4_ARTIFACT_DIR" ]]; then
        run_checked "worker_cache_metadata" ssh -o BatchMode=yes "$SSH_TARGET" \
            "cp '$REMOTE_OUTPUT/artifacts/config_nvfp4_patched.json' $remote_view_snapshot_q/config.json && cp '$REMOTE_NVFP4_ARTIFACT/hf_quant_config.json' $remote_view_snapshot_q/hf_quant_config.json"
    elif [[ -n "$FP8_ARTIFACT_DIR" ]]; then
        ssh -o BatchMode=yes "$SSH_TARGET" \
            "cp '$REMOTE_OUTPUT/artifacts/config_fp8_patched.json' $remote_view_snapshot_q/config.json && cp '$REMOTE_FP8_ARTIFACT/hf_quant_config.json' $remote_view_snapshot_q/hf_quant_config.json"
    else
        ssh -o BatchMode=yes "$SSH_TARGET" \
            "cp '$REMOTE_OUTPUT/artifacts/config_patched.json' $remote_view_snapshot_q/config.json && cp '$REMOTE_OUTPUT/artifacts/hf_quant_config_patched.json' $remote_view_snapshot_q/hf_quant_config.json"
    fi
    if [[ "$ORACLE_K0" == true ]]; then
        resolved_worker_shards=$(ssh -o BatchMode=yes "$SSH_TARGET" \
            "find $remote_view_snapshot_q -maxdepth 1 -name '*.safetensors' -type l | wc -l")
    else
        resolved_worker_shards=$(ssh -o BatchMode=yes "$SSH_TARGET" \
            "find -L $remote_view_snapshot_q -maxdepth 1 -name '*.safetensors' -type f | wc -l")
    fi
    [[ "$resolved_worker_shards" == 11 ]] || \
        fail "worker immutable cache view has unresolved checkpoint shards"
fi
write_launch_script_array "$OUTPUT_DIR/launch-worker.sh" 1 "$WORKER_IP" "$WORKER_IFACE" \
    "$WORKER_HCA" "$WORKER_RUNTIME_CACHE_MOUNT" "$REMOTE_OUTPUT/artifacts" "--headless" "ro" "$REMOTE_FP8_ARTIFACT" "$REMOTE_NVFP4_ARTIFACT" "$USE_IMMUTABLE_CACHE_VIEW"
write_launch_script_array "$OUTPUT_DIR/launch-head.sh" 0 "$HEAD_IP" "$HEAD_IFACE" \
    "$HEAD_HCA" "$HEAD_RUNTIME_CACHE_MOUNT" "$ARTIFACT_DIR" "--host 0.0.0.0 --port $API_PORT" "ro" "$FP8_ARTIFACT_DIR" "$NVFP4_ARTIFACT_DIR" "$USE_IMMUTABLE_CACHE_VIEW"
for launch_script in "$OUTPUT_DIR/launch-head.sh" "$OUTPUT_DIR/launch-worker.sh"; do
    run_checked "launch_argv_identity" bash "$launch_script" --print-argv > "$launch_script.actual-argv.json"
    cmp "$launch_script.expected-argv.json" "$launch_script.actual-argv.json" || \
        fail "generated launch argv differs from expected vector: $launch_script"
done
run_checked "worker_launcher_transfer" scp -q "$OUTPUT_DIR/launch-worker.sh" \
    "$SSH_TARGET:$REMOTE_OUTPUT/launch-worker.sh"
run_checked "worker_launch_argv_identity" ssh -o BatchMode=yes "$SSH_TARGET" \
    "bash '$REMOTE_OUTPUT/launch-worker.sh' --print-argv" \
    > "$OUTPUT_DIR/launch-worker.sh.remote-actual-argv.json"
cmp "$OUTPUT_DIR/launch-worker.sh.expected-argv.json" \
    "$OUTPUT_DIR/launch-worker.sh.remote-actual-argv.json" || \
    fail "remote launch argv differs from expected worker vector"

if [[ "$LAUNCH" != true ]]; then
    printf 'Two-node preflight complete. Re-run with a new empty --output-dir and --launch to execute.\n'
    exit 0
fi

run_checked "worker_launch" ssh -o BatchMode=yes "$SSH_TARGET" "bash '$REMOTE_OUTPUT/launch-worker.sh'"
worker_container_started=true
sleep 15
run_checked "head_launch" bash "$OUTPUT_DIR/launch-head.sh"
head_container_started=true
docker logs --timestamps -f "$HEAD_CONTAINER" >"$LOG_DIR/head.log" 2>&1 &
head_log_pid=$!
ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker logs --timestamps -f '$WORKER_CONTAINER'" >"$LOG_DIR/worker.log" 2>&1 &
worker_log_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
CURRENT_PHASE="startup_health"
until curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; do
    ((SECONDS < deadline)) || fail \
        "server did not become healthy within ${STARTUP_TIMEOUT_SECONDS} seconds (startup timeout budget exhausted)"
    docker inspect -f '{{.State.Running}}' "$HEAD_CONTAINER" 2>/dev/null | grep -qx true || \
        fail "head container exited during startup"
    ssh -o BatchMode=yes "$SSH_TARGET" \
        "docker inspect -f '{{.State.Running}}' '$WORKER_CONTAINER'" | grep -qx true || \
        fail "worker container exited during startup"
    sleep 10
done

if [[ "${ORACLE_K0:-false}" == true ]]; then
    CURRENT_PHASE="oracle_request"
    if ! python3 "$REPO_ROOT/scripts/runtime/qwen38-k0-oracle.py" invoke \
        --endpoint "http://127.0.0.1:$API_PORT" \
        --request "$OUTPUT_DIR/oracle-request.json" \
        --output "$OUTPUT_DIR/oracle-response.json" \
        --arm "$OUTPUT_DIR/ARMED"; then
        fail "oracle request invocation failed"
    fi
    CURRENT_PHASE="oracle_validation"
    if ! python3 "$REPO_ROOT/scripts/runtime/qwen38-k0-oracle.py" validate \
        --request "$OUTPUT_DIR/oracle-request.json" \
        --capture-dir "$OUTPUT_DIR/capture" \
        --response "$OUTPUT_DIR/oracle-response.json" \
        --output "$OUTPUT_DIR/oracle-result.json"; then
        fail "oracle artifact validation failed"
    fi
    cat "$OUTPUT_DIR/oracle-result.json"
    printf 'K0 target oracle complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

if [[ "$PRODUCTION" == true ]]; then
    # Keep startup records out of the fixed-depth acceptance sample.
    sleep 1
    head_benchmark_since=$(date --iso-8601=seconds)
    python3 "$REPO_ROOT/scripts/hardware/qwen38-two-node-monitor.py" collect \
        --worker "$SSH_TARGET" --output "$OUTPUT_DIR/hardware-samples.jsonl" &
    hardware_monitor_pid=$!
    python3 "$REPO_ROOT/scripts/baseline/openai-forked-prefix.py" \
        --endpoint "http://127.0.0.1:$API_PORT" --concurrency 1,2,4,8,16 \
        --decode 256 --json > "$OUTPUT_DIR/throughput.json"
    python3 "$REPO_ROOT/scripts/baseline/openai-forked-prefix.py" \
        --endpoint "http://127.0.0.1:$API_PORT" --concurrency 1,2 \
        --decode 256 \
        --user-prompt "implement a lock-free bounded ring buffer in C++20; emit code and invariants." \
        --json > "$OUTPUT_DIR/coding-throughput.json"
    kill -INT "$hardware_monitor_pid"
    wait "$hardware_monitor_pid" || fail "two-node hardware monitor failed"
    hardware_monitor_pid=""
    python3 "$REPO_ROOT/scripts/hardware/qwen38-two-node-monitor.py" summarize \
        --samples "$OUTPUT_DIR/hardware-samples.jsonl" \
        --benchmark "$OUTPUT_DIR/throughput.json" \
        --output "$OUTPUT_DIR/hardware.json"
    python3 "$REPO_ROOT/scripts/hardware/qwen38-two-node-monitor.py" summarize \
        --samples "$OUTPUT_DIR/hardware-samples.jsonl" \
        --benchmark "$OUTPUT_DIR/coding-throughput.json" \
        --output "$OUTPUT_DIR/coding-hardware.json"
    kill "$head_log_pid" "$worker_log_pid" >/dev/null 2>&1 || true
    wait "$head_log_pid" "$worker_log_pid" 2>/dev/null || true
    head_log_pid=""
    worker_log_pid=""
    python3 "$SCRIPT_DIR/qwen38-mtp-runtime-evidence.py" \
        --log "$LOG_DIR/head.log" \
        --not-before "$head_benchmark_since" \
        --min-records 2 --positions "$MTP_DEPTH" \
        > "$OUTPUT_DIR/mtp-runtime-evidence.json"
    cat "$OUTPUT_DIR/throughput.json"
    cat "$OUTPUT_DIR/coding-throughput.json"
    cat "$OUTPUT_DIR/hardware.json"
    cat "$OUTPUT_DIR/coding-hardware.json"
    cat "$OUTPUT_DIR/mtp-runtime-evidence.json"
    printf 'Production benchmark complete: %s\n' "$OUTPUT_DIR"
    exit 0
fi

# Move past the health-check second before opening the workload-only log window.
# This prevents a final startup/profiling record with the same timestamp second
# from entering the calibration gate.
sleep 1
head_workload_since=$(date --iso-8601=seconds)
worker_workload_since=$(ssh -o BatchMode=yes "$SSH_TARGET" date --iso-8601=seconds)
printf 'head=%s\nworker=%s\n' "$head_workload_since" "$worker_workload_since" \
    > "$LOG_DIR/workload-since.txt"
docker logs --timestamps --since "$head_workload_since" -f "$HEAD_CONTAINER" \
    >"$LOG_DIR/head-workload.log" 2>&1 &
head_workload_log_pid=$!
ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker logs --timestamps --since '$worker_workload_since' -f '$WORKER_CONTAINER'" \
    >"$LOG_DIR/worker-workload.log" 2>&1 &
worker_workload_log_pid=$!

python3 "$SCRIPT_DIR/qwen38-attention-calibration.py" \
    --endpoint "http://127.0.0.1:$API_PORT" --mtp --concurrent-streams 4 \
    --long-decode-tokens 2048 --mtp-tokens 512 \
    --out "$OUTPUT_DIR/attention-calibration.json"
sleep 5

# Stop followers so all buffered telemetry is visible before reduction.
kill "$head_log_pid" "$worker_log_pid" "$head_workload_log_pid" \
    "$worker_workload_log_pid" >/dev/null 2>&1 || true
wait "$head_log_pid" "$worker_log_pid" "$head_workload_log_pid" \
    "$worker_workload_log_pid" 2>/dev/null || true
head_log_pid=""
worker_log_pid=""
head_workload_log_pid=""
worker_workload_log_pid=""
cat "$LOG_DIR/head.log" "$LOG_DIR/worker.log" > "$LOG_DIR/combined.log"
cat "$LOG_DIR/head-workload.log" "$LOG_DIR/worker-workload.log" \
    > "$LOG_DIR/combined-workload.log"

# Public model metadata can omit speculative_config. Runtime interval metrics
# from the workload window are the execution proof and are required to proceed.
python3 "$SCRIPT_DIR/qwen38-mtp-runtime-evidence.py" \
    --log "$LOG_DIR/head-workload.log" \
    --not-before "$head_workload_since" \
    --min-records 2 --positions "$MTP_DEPTH" \
    > "$OUTPUT_DIR/mtp-runtime-evidence.json"

for node in head worker combined; do
    python3 "$SCRIPT_DIR/qwen38-activation-maxima.py" \
        --require-expanded --expanded-v2-only --min-emission-call 8 \
        --layers 36 --full-attention-layers 12 \
        --ple-layers 1 --router-layers 48 --recurrent-state-layers 36 \
        < "$LOG_DIR/$node-workload.log" \
        > "$OUTPUT_DIR/$node-activation-summary.json"
done

printf 'Expanded calibration complete: %s\n' "$OUTPUT_DIR"
