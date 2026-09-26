#!/usr/bin/env bash
# Regenerate one shard of the PerfectBlend training split with the target model.
#
# Starts one SGLang server per listed GPU, verifies the input shard against its
# manifest, and calls scripts/data/generate_train_data.py with the target's sampling
# settings (Qwen3-4B: temperature 0.7, top-p 0.8, top-k 20, thinking disabled,
# max 4096 tokens per assistant turn).  Every assistant turn is regenerated in
# order.  Resume is by exact source id: rerun until the shard is complete.
#
# Usage (shards from scripts/data/shard_jsonl.py):
#   bash scripts/data/launch_rollout_shard.sh qwen3_4b 0 4 "0 1 2 3"
set -euo pipefail

MODEL_KEY=${1:?Usage: $0 MODEL_KEY SHARD_INDEX NUM_SHARDS "GPU_IDS"}
SHARD_INDEX=${2:?Usage: $0 MODEL_KEY SHARD_INDEX NUM_SHARDS "GPU_IDS"}
NUM_SHARDS=${3:?Usage: $0 MODEL_KEY SHARD_INDEX NUM_SHARDS "GPU_IDS"}
GPUS=${4:-0}

case "${MODEL_KEY}" in
  qwen3_4b)
    DEFAULT_MODEL_ID=Qwen/Qwen3-4B
    DEFAULT_CONCURRENCY_PER_GPU=32
    SAMPLING_TAG=temp07
    TEMPERATURE=0.7
    TOP_P=0.8
    TOP_K=20
    ;;
  *)
    echo "Unknown MODEL_KEY=${MODEL_KEY}; expected qwen3_4b" >&2
    exit 2
    ;;
esac

if ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ && "${NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SHARD_INDEX and NUM_SHARDS must be non-negative/positive integers" >&2
  exit 2
fi
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "SHARD_INDEX=${SHARD_INDEX} must be smaller than NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
SGLANG_BIN=${SGLANG_BIN:-sglang}
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_ID}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${DEFAULT_MODEL_ID}}
HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
TMPDIR=${TMPDIR:-/tmp}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-1349860}
PORT_BASE=${PORT_BASE:-30000}
MEM_FRACTION=${MEM_FRACTION:-0.85}
CONCURRENCY_PER_GPU=${CONCURRENCY_PER_GPU:-${DEFAULT_CONCURRENCY_PER_GPU}}
MAX_SHARD_ITEMS=${MAX_SHARD_ITEMS:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

NAMESPACE=${MODEL_KEY}_official_pb95_multiturn_${SAMPLING_TAG}_max4096
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/refine_data/rollouts/${NAMESPACE}}
SHARD_STEM=shard_$(printf '%05d' "${SHARD_INDEX}")_of_$(printf '%05d' "${NUM_SHARDS}")
SHARD_INPUT_DIR=${SHARD_INPUT_DIR:-${REPO_ROOT}/cache/dataset/perfectblend_train_shards_${NUM_SHARDS}}
SHARD_INPUT=${SHARD_INPUT:-${SHARD_INPUT_DIR}/${SHARD_STEM}.jsonl}
SHARD_MANIFEST=${SHARD_MANIFEST:-${SHARD_INPUT_DIR}/manifest.json}
EXPECTED_SOURCE_SHA256=${EXPECTED_SOURCE_SHA256:-}
OUT=${OUT:-${OUT_DIR}/${SHARD_STEM}.jsonl}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/rollout_${NAMESPACE}/shard_${SHARD_INDEX}}

if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* ]]; then
  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Local MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
  fi
fi
VERIFY_ARGS=(
  --manifest "${SHARD_MANIFEST}"
  --shard "${SHARD_INPUT}"
  --shard-index "${SHARD_INDEX}"
  --num-shards "${NUM_SHARDS}"
  --expected-total "${EXPECTED_TRAIN_ROWS}"
)
if [[ -n "${EXPECTED_SOURCE_SHA256}" ]]; then
  VERIFY_ARGS+=(--expected-source-sha256 "${EXPECTED_SOURCE_SHA256}")
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/data/verify_jsonl_shard.py" "${VERIFY_ARGS[@]}"

if (( PREFLIGHT_ONLY == 1 )); then
  echo "[rollout] PREFLIGHT OK model_key=${MODEL_KEY} model_path=${MODEL_PATH} served_model=${SERVED_MODEL_NAME} temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} namespace=${NAMESPACE}"
  exit 0
elif (( PREFLIGHT_ONLY != 0 )); then
  echo "PREFLIGHT_ONLY must be 0 or 1, got ${PREFLIGHT_ONLY}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TMPDIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME TMPDIR

read -r -a GPU_ARRAY <<< "${GPUS}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "No GPU IDs supplied" >&2
  exit 2
fi

SERVER_PIDS=()
SERVERS=()
cleanup() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  wait "${SERVER_PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[rollout] OFFICIAL protocol: 95% split, all user turns, max_tokens=4096"
echo "[rollout] model_id=${DEFAULT_MODEL_ID} model_path=${MODEL_PATH} served_model=${SERVED_MODEL_NAME}"
echo "[rollout] shard=${SHARD_INDEX}/${NUM_SHARDS} GPUs=${GPUS}"
echo "[rollout] input=${SHARD_INPUT} output=${OUT}"

for worker_index in "${!GPU_ARRAY[@]}"; do
  gpu_id=${GPU_ARRAY[${worker_index}]}
  port=$((PORT_BASE + worker_index))
  server=http://127.0.0.1:${port}
  server_address=127.0.0.1:${port}
  log_file=${LOG_DIR}/server_gpu${gpu_id}_port${port}.log
  if curl --silent --fail --max-time 2 "${server}/health" >/dev/null 2>&1; then
    echo "[rollout] refusing to reuse an existing server on ${server}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=${gpu_id} "${SGLANG_BIN}" serve \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype bfloat16 \
    --mem-fraction-static "${MEM_FRACTION}" \
    > "${log_file}" 2>&1 &
  SERVER_PIDS+=("$!")
  SERVERS+=("${server_address}")
  echo "[rollout] launched GPU=${gpu_id} pid=$! server=${server} log=${log_file}"
done

for worker_index in "${!SERVERS[@]}"; do
  server=http://${SERVERS[${worker_index}]}
  server_pid=${SERVER_PIDS[${worker_index}]}
  ready=0
  for _ in $(seq 1 240); do
    if curl --silent --fail --max-time 3 "${server}/health" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "[rollout] launched server died before becoming ready: ${server}" >&2
      exit 1
    fi
    sleep 5
  done
  if (( ready == 0 )); then
    echo "[rollout] server failed health check: ${server}" >&2
    exit 1
  fi
  echo "[rollout] ready ${server}"
done

if (( MAX_SHARD_ITEMS > 0 )); then
  read -r SHARD_ROWS _ < <(wc -l "${SHARD_INPUT}")
  if (( MAX_SHARD_ITEMS > SHARD_ROWS )); then
    echo "MAX_SHARD_ITEMS=${MAX_SHARD_ITEMS} exceeds shard rows=${SHARD_ROWS}" >&2
    exit 1
  fi
  SMOKE_OUT=${OUT_DIR}/smoke_${SHARD_STEM}_first_${MAX_SHARD_ITEMS}.jsonl
  echo "[rollout] exact-ID smoke: first ${MAX_SHARD_ITEMS} conversation(s) -> ${SMOKE_OUT}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/data/generate_train_data.py" \
    --model "${SERVED_MODEL_NAME}" \
    --server-address "${SERVERS[@]}" \
    --input-file-path "${SHARD_INPUT}" \
    --output-file-path "${SMOKE_OUT}" \
    --concurrency "${CONCURRENCY_PER_GPU}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --num-samples "${MAX_SHARD_ITEMS}" \
    --resume \
    --resume-by-id
  sha256sum "${SMOKE_OUT}"
  echo "[rollout] SMOKE COMPLETE; formal output remains untouched"
  exit 0
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/data/generate_train_data.py" \
  --model "${SERVED_MODEL_NAME}" \
  --server-address "${SERVERS[@]}" \
  --input-file-path "${SHARD_INPUT}" \
  --output-file-path "${OUT}" \
  --concurrency "${CONCURRENCY_PER_GPU}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --min-p 0 \
  --max-tokens 4096 \
  --disable-thinking \
  --resume \
  --resume-by-id

sha256sum "${OUT}"
echo "[rollout] COMPLETE model=${MODEL_KEY} shard=${SHARD_INDEX}/${NUM_SHARDS}"
