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
  --startup-timeout-seconds N
                         Health deadline in seconds (default: 3600)
  --gpu-memory-utilization F
                         vLLM device-memory fraction in (0,1] (default: 0.835)
  --mtp-depth K          Fixed MTP depth 1..3 (default: 3); K0 fails closed
  --help                 Show this help
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --output-dir) OUTPUT_DIR=${2:?missing value}; shift 2 ;;
        --launch) LAUNCH=true; shift ;;
        --two-node-preflight) TWO_NODE_PREFLIGHT=true; shift ;;
        --production) PRODUCTION=true; LAUNCH=true; shift ;;
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
[[ "$MTP_DEPTH" =~ ^[0-3]$ ]] || fail "--mtp-depth must be one of 0,1,2,3"
# Pinned image d464f3b4 declares SpeculativeConfig.num_speculative_tokens with
# Pydantic Field(gt=0). Omitting speculative_config also omits the MTP model and
# its cache path, so neither CLI shape represents K0 with synchronized MTP state.
[[ "$MTP_DEPTH" != 0 ]] || fail "pinned vLLM requires num_speculative_tokens > 0; true K0 with loaded MTP state is unavailable"
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

head_page_size=$(getconf PAGESIZE)
worker_page_size=$(ssh -o BatchMode=yes "$SSH_TARGET" getconf PAGESIZE)
[[ "$head_page_size" == 65536 ]] || fail "head page size is $head_page_size, expected 65536"
[[ "$worker_page_size" == 65536 ]] || fail "worker page size is $worker_page_size, expected 65536"

HEAD_SNAPSHOT="$HF_CACHE/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION"
[[ -f "$HEAD_SNAPSHOT/config.json" ]] || fail "head checkpoint revision missing: $HEAD_SNAPSHOT"
ssh -o BatchMode=yes "$SSH_TARGET" "docker volume inspect '$WORKER_HF_VOLUME' >/dev/null" || \
    fail "worker Hugging Face volume missing: $WORKER_HF_VOLUME"
ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker run --rm -v '$WORKER_HF_VOLUME:/cache:ro' --entrypoint /bin/sh '$IMAGE_TAG' -c 'test -f /cache/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/model.safetensors.index.json'" || \
    fail "worker volume $WORKER_HF_VOLUME does not contain pinned checkpoint metadata"

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
extract_image_file "$CONTAINER_MODEL_DIR/model.py" "$ARTIFACT_DIR/model_router.py"

python3 "$RUNTIME_FILES/patch_ple_layer.py" >/dev/null
python3 "$RUNTIME_FILES/patch_modelopt_mxfp8.py" >/dev/null
python3 "$SCRIPT_DIR/patch-qwen38-modelopt-fp8-block-moe.py" \
    "$RUNTIME_FILES/modelopt_patched.py"
python3 "$RUNTIME_FILES/patch_qsa_fp8_kv.py" >/dev/null
python3 "$RUNTIME_FILES/patch_checkpoint_config.py" "$HEAD_SNAPSHOT" "$RUNTIME_FILES" >/dev/null
python3 "$REPO_ROOT/scripts/runtime/patch-vllm-64k-loader.py" "$ARTIFACT_DIR/weight_utils_64k.py"
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
    --entrypoint /usr/bin/python3 "$IMAGE_TAG" -m py_compile \
    /work/model.py /work/model_router.py
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
(
    cd "$ARTIFACT_DIR"
    sha256sum ./* > SHA256SUMS
)
cat > "$OUTPUT_DIR/run.json" <<EOF
{"image_id":"$IMAGE_ID","image_tag":"$IMAGE_TAG","mia_commit":"$MIA_COMMIT","model":"$MODEL_ID","model_revision":"$MODEL_REVISION","head_page_size":$head_page_size,"worker_page_size":$worker_page_size,"startup_timeout_seconds":$STARTUP_TIMEOUT_SECONDS,"gpu_memory_utilization":$GPU_MEMORY_UTILIZATION,"mtp_depth":$MTP_DEPTH}
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
ssh -o BatchMode=yes "$SSH_TARGET" "mkdir -p '$REMOTE_OUTPUT/artifacts' '$REMOTE_OUTPUT/logs'"
scp -q "$ARTIFACT_DIR"/* "$SSH_TARGET:$REMOTE_OUTPUT/artifacts/"
ssh -o BatchMode=yes "$SSH_TARGET" \
    "cd '$REMOTE_OUTPUT/artifacts' && sha256sum --check SHA256SUMS"

write_launch_script() {
    local destination=$1 node_rank=$2 node_ip=$3 iface=$4 hca=$5 cache_mount=$6 artifact_dir=$7 mode=$8 cache_access=$9 fp8_host_dir=${10} nvfp4_host_dir=${11}
    local fp8_options="" nvfp4_options="" quant_config_source="$artifact_dir/hf_quant_config_patched.json" config_source="$artifact_dir/config_patched.json"
    if [[ -n "$fp8_host_dir" ]]; then
        quant_config_source="$fp8_host_dir/hf_quant_config.json"
        config_source="$artifact_dir/config_fp8_patched.json"
        fp8_options="
  -v $(printf '%q' "$fp8_host_dir"):/rocket/qwen38-linear-fp8:ro \\
  -e ROCKET_QWEN38_FP8_OVERLAY_MANIFEST=/rocket/qwen38-linear-fp8/manifest.json \\
  -e ROCKET_QWEN38_FP8_QUANT_CONFIG=/rocket/qwen38-linear-fp8/hf_quant_config.json \\"
    fi
    if [[ -n "$nvfp4_host_dir" ]]; then
        quant_config_source="$nvfp4_host_dir/hf_quant_config.json"
        config_source="$artifact_dir/config_nvfp4_patched.json"
        nvfp4_options="
  -v $(printf '%q' "$nvfp4_host_dir"):/rocket/qwen38-linear-nvfp4:ro \\
  -e ROCKET_QWEN38_NVFP4_OVERLAY_MANIFEST=/rocket/qwen38-linear-nvfp4/manifest.json \\
  -e ROCKET_QWEN38_NVFP4_QUANT_CONFIG=/rocket/qwen38-linear-nvfp4/hf_quant_config.json \\"
    fi
    cat > "$destination" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec docker run -d --name $(if [[ "$node_rank" == 0 ]]; then printf '%q' "$HEAD_CONTAINER"; else printf '%q' "$WORKER_CONTAINER"; fi) \\
  --gpus all --network host --ipc host \\
  --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \\
  --device /dev/infiniband:/dev/infiniband \\
  -e GLOO_SOCKET_IFNAME=$(printf '%q' "$iface") \\
  -e NCCL_SOCKET_IFNAME=$(printf '%q' "$iface") \\
  -e TP_SOCKET_IFNAME=$(printf '%q' "$iface") \\
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$(printf '%q' "$hca") \\
  -e NCCL_IB_GID_INDEX=$GID_INDEX -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN \\
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \\
  -e VLLM_HOST_IP=$(printf '%q' "$node_ip") -e HF_HOME=/root/.cache/huggingface \\
  -e ROCKET_NVFP4_CALIBRATE=1 -e ROCKET_NVFP4_SAMPLE_ELEMENTS=2048 \\
  -e ROCKET_QWEN38_LOAD_TRACE=1 \\
  -e ROCKET_NVFP4_MAX_EMISSIONS=12 \\$fp8_options$nvfp4_options
  -v $(printf '%q' "$artifact_dir/ple_layer_patched.py"):$CONTAINER_MODEL_DIR/ple_layer.py:ro \\
  -v $(printf '%q' "$artifact_dir/modelopt_patched.py"):$CONTAINER_VLLM_DIR/model_executor/layers/quantization/modelopt.py:ro \\
  -v $(printf '%q' "$artifact_dir/weight_utils_64k.py"):$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py:ro \\
  -v $(printf '%q' "$artifact_dir/model_telemetry.py"):$CONTAINER_MODEL_DIR/model.py:ro \\
  -v $(printf '%q' "$artifact_dir/qsa_ops_patched.py"):$CONTAINER_MODEL_DIR/ops/qsa.py:ro \\
  -v $(printf '%q' "$artifact_dir/qsa_nvidia_patched.py"):$CONTAINER_MODEL_DIR/qsa.py:ro \\
  -v $(printf '%q' "$config_source"):/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/config.json:ro \\
  -v $(printf '%q' "$quant_config_source"):/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/hf_quant_config.json:ro \\
  -v $(printf '%q' "$cache_mount"):/root/.cache/huggingface:$cache_access \\
  -v \$HOME/.cache/vllm:/root/.cache/vllm \\
  $IMAGE_TAG $MODEL_ID \\
  --revision $MODEL_REVISION --served-model-name qwen3.8-flash-next \\
  --tensor-parallel-size 2 --gpu-memory-utilization $(printf '%q' "$GPU_MEMORY_UTILIZATION") \\
  --max-num-seqs 16 --max-num-batched-tokens 8192 --max-model-len 262144 \\
  --kv-cache-dtype fp8 --load-format safetensors --safetensors-load-strategy lazy \\
  --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice \\
  --tool-call-parser qwen3_coder --distributed-executor-backend mp \\
  --mm-encoder-tp-mode data --nnodes 2 --master-addr $HEAD_IP --master-port $MASTER_PORT \\
  --enable-expert-parallel --all2all-backend allgather_reducescatter \\
  --speculative-config '{"method":"mtp","num_speculative_tokens":$MTP_DEPTH}' \\
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \\
  --hf-overrides '{"text_config":{"ple_embedding_dtype":"float8_e4m3fn"}}' \\
  --enforce-eager --node-rank $node_rank $mode
EOF
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
    ssh -o BatchMode=yes "$SSH_TARGET" "mkdir -p '$REMOTE_NVFP4_ARTIFACT'"
    (
        cd "$NVFP4_ARTIFACT_DIR"
        sha256sum manifest.json hf_quant_config.json "$NVFP4_OVERLAY_FILE" \
            > "$WORK_DIR/nvfp4-artifact-SHA256SUMS"
    )
    scp -q "$NVFP4_ARTIFACT_DIR/manifest.json" "$NVFP4_ARTIFACT_DIR/hf_quant_config.json" \
        "$NVFP4_ARTIFACT_DIR/$NVFP4_OVERLAY_FILE" \
        "$WORK_DIR/nvfp4-artifact-SHA256SUMS" \
        "$SSH_TARGET:$REMOTE_NVFP4_ARTIFACT/"
    ssh -o BatchMode=yes "$SSH_TARGET" \
        "cd '$REMOTE_NVFP4_ARTIFACT' && sha256sum --check nvfp4-artifact-SHA256SUMS"
    ssh -o BatchMode=yes "$SSH_TARGET" \
        "docker run --rm \
        -v '$WORKER_HF_VOLUME:/root/.cache/huggingface:ro' \
        -v '$REMOTE_NVFP4_ARTIFACT:/rocket/qwen38-linear-nvfp4:ro' \
        -v '$REMOTE_OUTPUT/artifacts/weight_utils_64k.py:$CONTAINER_VLLM_DIR/model_executor/model_loader/weight_utils.py:ro' \
        -e ROCKET_QWEN38_NVFP4_OVERLAY_MANIFEST=/rocket/qwen38-linear-nvfp4/manifest.json \
        -e ROCKET_QWEN38_NVFP4_QUANT_CONFIG=/rocket/qwen38-linear-nvfp4/hf_quant_config.json \
        --entrypoint /usr/bin/python3 '$IMAGE_TAG' -c \
        \"import glob; from vllm.model_executor.model_loader.weight_utils import _rocket_qwen38_nvfp4_overlay_preflight as check; files=glob.glob('/root/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$MODEL_REVISION/*.safetensors'); result=check(sorted(files), 'lazy'); assert len(result['selected']) == $NVFP4_EXPECTED_COUNT; print('validated worker NVFP4 overlay: $NVFP4_EXPECTED_COUNT tensors')\""
    ssh -o BatchMode=yes "$SSH_TARGET" \
        "docker run --rm \
        -v '$REMOTE_OUTPUT/artifacts/modelopt_patched.py:$CONTAINER_VLLM_DIR/model_executor/layers/quantization/modelopt.py:ro' \
        -v '$REMOTE_NVFP4_ARTIFACT/hf_quant_config.json:/work/hf_quant_config.json:ro' \
        -v '$REMOTE_OUTPUT/artifacts/model_telemetry.py:/work/model.py:ro' \
        -v '$REMOTE_OUTPUT/artifacts/model_router.py:/work/model_router.py:ro' \
        -e ROCKET_NVFP4_FAMILIES='$NVFP4_FAMILIES_CSV' \
        --entrypoint /usr/bin/python3 '$IMAGE_TAG' -c \
        \"import json,os,pathlib; from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig; config=ModelOptMixedPrecisionConfig.from_config(json.load(open('/work/hf_quant_config.json'))); families=set(os.environ['ROCKET_NVFP4_FAMILIES'].split(',')); checks={'base_routers':'model.language_model.model.layers.0.mlp.gate','base_ple':'model.language_model.model.layers.1.ple.value_proj'}; [(_ for _ in ()).throw(AssertionError((family, config._resolve_quant_algo(target)))) for family,target in checks.items() if family in families and config._resolve_quant_algo(target) != 'NVFP4']; telemetry=pathlib.Path('/work/model.py').read_text(); runtime=pathlib.Path('/work/model_router.py').read_text(); selected='base_routers' in families; assert ('ROCKET_QWEN38_NVFP4_ROUTER_V1' in telemetry) == selected; assert ('ROCKET_QWEN38_NVFP4_ROUTER_V1' in runtime) == selected; assert 'ROCKET_NVFP4_TELEMETRY' in telemetry; assert 'ROCKET_NVFP4_TELEMETRY' not in runtime; print(f'validated worker NVFP4 mounted-model semantics: {sorted(families)}')\""
fi
write_launch_script "$OUTPUT_DIR/launch-worker.sh" 1 "$WORKER_IP" "$WORKER_IFACE" \
    "$WORKER_HCA" "$WORKER_HF_VOLUME" "$REMOTE_OUTPUT/artifacts" "--headless" "ro" "$REMOTE_FP8_ARTIFACT" "$REMOTE_NVFP4_ARTIFACT"
write_launch_script "$OUTPUT_DIR/launch-head.sh" 0 "$HEAD_IP" "$HEAD_IFACE" \
    "$HEAD_HCA" "$HF_CACHE" "$ARTIFACT_DIR" "--host 0.0.0.0 --port $API_PORT" "rw" "$FP8_ARTIFACT_DIR" "$NVFP4_ARTIFACT_DIR"
if [[ "$PRODUCTION" == true ]]; then
    for launch_script in "$OUTPUT_DIR/launch-head.sh" "$OUTPUT_DIR/launch-worker.sh"; do
        sed -i \
            -e '/ROCKET_NVFP4_CALIBRATE=/d' \
            -e '/ROCKET_NVFP4_SAMPLE_ELEMENTS=/d' \
            -e '/ROCKET_NVFP4_MAX_EMISSIONS=/d' \
            -e '/ROCKET_QWEN38_LOAD_TRACE=/d' \
            -e 's/--enforce-eager //' \
            "$launch_script"
        if [[ "$NVFP4_HAS_BASE_ROUTERS" == true ]]; then
            sed -i 's/model_telemetry.py:/model_router.py:/' "$launch_script"
        else
            sed -i '/model_telemetry.py:.*\/model.py:ro/d' "$launch_script"
        fi
    done
fi
scp -q "$OUTPUT_DIR/launch-worker.sh" "$SSH_TARGET:$REMOTE_OUTPUT/launch-worker.sh"

if [[ "$LAUNCH" != true ]]; then
    printf 'Two-node preflight complete. Re-run with a new empty --output-dir and --launch to execute.\n'
    exit 0
fi

head_log_pid=""
worker_log_pid=""
head_workload_log_pid=""
worker_workload_log_pid=""
hardware_monitor_pid=""
cleanup() {
    [[ -n "$head_log_pid" ]] && kill "$head_log_pid" >/dev/null 2>&1 || true
    [[ -n "$worker_log_pid" ]] && kill "$worker_log_pid" >/dev/null 2>&1 || true
    [[ -n "$head_workload_log_pid" ]] && kill "$head_workload_log_pid" >/dev/null 2>&1 || true
    [[ -n "$worker_workload_log_pid" ]] && kill "$worker_workload_log_pid" >/dev/null 2>&1 || true
    [[ -n "$hardware_monitor_pid" ]] && kill "$hardware_monitor_pid" >/dev/null 2>&1 || true
    [[ -n "$hardware_monitor_pid" ]] && wait "$hardware_monitor_pid" 2>/dev/null || true
    if [[ "$KEEP_RUNNING" != true ]]; then
        docker rm -f "$HEAD_CONTAINER" >/dev/null 2>&1 || true
        ssh -o BatchMode=yes "$SSH_TARGET" \
            "docker rm -f '$WORKER_CONTAINER' >/dev/null 2>&1 || true" || true
    fi
}
trap cleanup EXIT INT TERM

ssh -o BatchMode=yes "$SSH_TARGET" "bash '$REMOTE_OUTPUT/launch-worker.sh'"
sleep 15
bash "$OUTPUT_DIR/launch-head.sh"
docker logs --timestamps -f "$HEAD_CONTAINER" >"$LOG_DIR/head.log" 2>&1 &
head_log_pid=$!
ssh -o BatchMode=yes "$SSH_TARGET" \
    "docker logs --timestamps -f '$WORKER_CONTAINER'" >"$LOG_DIR/worker.log" 2>&1 &
worker_log_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
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

if [[ "$PRODUCTION" == true ]]; then
    python3 "$REPO_ROOT/scripts/hardware/qwen38-two-node-monitor.py" collect \
        --worker "$SSH_TARGET" --output "$OUTPUT_DIR/hardware-samples.jsonl" &
    hardware_monitor_pid=$!
    python3 "$REPO_ROOT/scripts/baseline/openai-forked-prefix.py" \
        --endpoint "http://127.0.0.1:$API_PORT" --concurrency 1,2,4,8,16 \
        --decode 256 --json > "$OUTPUT_DIR/throughput.json"
    kill -INT "$hardware_monitor_pid"
    wait "$hardware_monitor_pid" || fail "two-node hardware monitor failed"
    hardware_monitor_pid=""
    python3 "$REPO_ROOT/scripts/hardware/qwen38-two-node-monitor.py" summarize \
        --samples "$OUTPUT_DIR/hardware-samples.jsonl" \
        --benchmark "$OUTPUT_DIR/throughput.json" \
        --output "$OUTPUT_DIR/hardware.json"
    cat "$OUTPUT_DIR/throughput.json"
    cat "$OUTPUT_DIR/hardware.json"
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
    --min-records 2 --positions 3 \
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
