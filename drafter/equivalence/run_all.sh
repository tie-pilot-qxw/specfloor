#!/usr/bin/env bash
# Equivalence of the released overlay against the original research tree, for every
# paper configuration with a checkpoint.  For each configuration:
#
#   1. run_step.py on the ORIGINAL tree with a fresh torch.compile cache;
#   2. run_step.py on upstream DeepSpec + overlay with a COPY of that cache, so both
#      use the same compiled kernels (inductor autotunes reduction kernels by timing,
#      so two independent compilations can pick different block configs and differ
#      in the last bit of compiled reductions);
#   3. compare.py: init, strict checkpoint load, per-micro-batch losses, forward
#      outputs, metrics and gradients must be bit-identical.
#
# For the final configuration it also runs the overlay with its own independent
# compile cache, as a control that shows the size of compile-level differences.
#
#   ORIG=<original tree> OVL=<upstream+overlay> CKPT=<checkpoint root> \
#   DATA=<online_train.jsonl> OUT=<dir> GPU=0 PY=python \
#   [SPECULATORS_PATH=<colon-separated import path>] bash run_all.sh
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${ORIG:?}" "${OVL:?}" "${CKPT:?}" "${DATA:?}" "${OUT:?}"
GPU=${GPU:-0}; PY=${PY:-python}
mkdir -p "$OUT/cache"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIGS=(
  "attnconv_qwen3_4b_b7_10ep attnconv_b7_qwen3_4b_10ep/step_26160"
  "dspark_qwen3_4b_b7_1ep dspark_b7_qwen3_4b_1ep/step_2616"
  "dspark_qwen3_4b_b7_1ep_shortconv dspark_b7_qwen3_4b_1ep_shortconv/step_2616"
  "slotembed_qwen3_4b_b7 slotembed_b7_qwen3_4b/step_2616"
  "attnhead_qwen3_4b_b7 attnhead_b7_qwen3_4b/step_2616"
  "attnconv_qwen3_4b_b7 attnconv_b7_qwen3_4b/step_2616"
  "official_dflash2_qwen3_4b_b8_1ep official_dflash2_b8_qwen3_4b_1ep/step_2616"
)

step () {   # $1 tree  $2 config  $3 checkpoint  $4 out.json  $5 cache dir  $6 port  [$7 extra PYTHONPATH]
  TORCHINDUCTOR_CACHE_DIR="$5/inductor" TRITON_CACHE_DIR="$5/triton" \
  PYTHONPATH="${7:-}" CUDA_VISIBLE_DEVICES=$GPU \
    "$PY" "$HERE/run_step.py" --repo "$1" --config "$2" --checkpoint "$3" \
      --data "$DATA" --out "$4" --port "$6" > "$4.log" 2>&1
  local rc=$?
  echo "  $(basename "$4"): exit $rc"
  return $rc
}

port=29640
for entry in "${CONFIGS[@]}"; do
  read -r name ckpt <<< "$entry"
  extra=""
  if [[ $name == official_dflash2* ]]; then
    if [ -z "${SPECULATORS_PATH:-}" ]; then
      echo "## $name: skipped (set SPECULATORS_PATH to run the DFlash2 reproduction)"
      continue
    fi
    extra=$SPECULATORS_PATH
  fi
  echo "## $name"
  port=$((port + 1))
  rm -rf "$OUT/cache/$name.orig" "$OUT/cache/$name.shared"
  step "$ORIG" "$ORIG/config/dspark/$name.py" "$CKPT/$ckpt" "$OUT/$name.orig.json" \
       "$OUT/cache/$name.orig" $port "$extra" || continue
  cp -r "$OUT/cache/$name.orig" "$OUT/cache/$name.shared"
  port=$((port + 1))
  step "$OVL" "$OVL/config/dspark/$name.py" "$CKPT/$ckpt" "$OUT/$name.overlay.json" \
       "$OUT/cache/$name.shared" $port "$extra" || continue
  "$PY" "$HERE/compare.py" "$OUT/$name.orig.json" "$OUT/$name.overlay.json" \
      --json "$OUT/$name.compare.json" | sed 's/^/  /'
done

# Control: an independent compilation of the overlay for the final configuration.
name=attnconv_qwen3_4b_b7_10ep
echo "## $name (independent compile cache, control)"
rm -rf "$OUT/cache/$name.indep"
port=$((port + 1))
step "$OVL" "$OVL/config/dspark/$name.py" "$CKPT/attnconv_b7_qwen3_4b_10ep/step_26160" \
     "$OUT/$name.overlay_indep.json" "$OUT/cache/$name.indep" $port "" && \
"$PY" "$HERE/compare.py" "$OUT/$name.orig.json" "$OUT/$name.overlay_indep.json" \
    --json "$OUT/$name.compare_indep.json" | sed 's/^/  /'
