#!/bin/bash
# Gemma-4-12B re-anchored. See PREDICTION.md.
#
# The corpus is REUSED from scale/gemma12b/C0: the bug was in anchors.py, which
# selects positions inside an already-written response, so the responses
# themselves are unaffected and regenerating them would only burn a GPU-hour and
# change the text under comparison. Everything downstream of anchors is redone
# because everything downstream consumes the anchor set.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/specfloor
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec
export SPECFLOOR_EVAL_ROOT=/workspace/DeepSpec/eval_datasets
PY=/workspace/sglang017-env/bin/python

TARGET=google/gemma-4-12B-it
DFLASH=deepseek-ai/dflash_gemma4_12b_block7
DSPARK=deepseek-ai/dspark_gemma4_12b_block7
SRC=/workspace/measurement_runs/scale/gemma12b/C0        # corpus, reused
D=/workspace/measurement_runs/scale/gemma12b_fix/C0      # anchors + ladder, new
OUT=/workspace/measurement_runs/scale/gemma12b_fix
mkdir -p $D
QUIET='Loading|Fetching|it/s\]|UserWarning|SOLUTION|flex_attention|_warn_once|debug your score_mod|This will allow|Capturing|Multi-thread'

for dom in gsm8k mbpp alpaca arena8k; do
  echo ""; echo "######## anchors $dom"
  [ -s $D/$dom.anchors.jsonl ] || $PY -m specfloor.anchors \
      --corpus-file $SRC/$dom.jsonl --out $D/$dom.anchors.jsonl --budget 256 \
      --target $TARGET 2>&1 | grep -avE "$QUIET" | tail -4
  echo "######## ladder $dom"
  [ -s $D/$dom.ladder.jsonl ] || $PY -m specfloor.probe_cheap --corpus C0 \
      --corpus-file $SRC/$dom.jsonl --anchors $D/$dom.anchors.jsonl \
      --out $D/$dom.ladder.jsonl --target $TARGET --m-base 512 --m-max 512 \
      2>&1 | grep -avE "$QUIET" | tail -4
done

for dom in gsm8k mbpp alpaca arena8k; do
  echo ""; echo "######## tk $dom"
  [ -s $OUT/$dom.t01.jsonl ] || $PY -m specfloor.probe_tk --corpus C0 \
      --corpus-file $SRC/$dom.jsonl --cheap $D/$dom.ladder.jsonl \
      --out $OUT/$dom.t01.jsonl --target $TARGET \
      --anchors 96 --paths 1024 --top-k 256 --rungs 0,1 --split \
      2>&1 | grep -avE "$QUIET" | tail -6
done

for dom in gsm8k mbpp alpaca arena8k; do
  echo ""; echo "######## rpre $dom order 0"
  [ -s $OUT/$dom.srv0.jsonl ] || $PY -m specfloor.probe_rpre --corpus C0 \
      --corpus-file $SRC/$dom.jsonl --cheap $D/$dom.ladder.jsonl \
      --drafter "$DFLASH" --order 0 --cond both --out $OUT/$dom.srv0.jsonl \
      --target $TARGET --anchors 96 --paths 256 --split --kv-budget-gib 12 \
      2>&1 | grep -avE "$QUIET" | tail -6
  echo "######## rpre $dom order 1"
  [ -s $OUT/$dom.srv1.jsonl ] || $PY -m specfloor.probe_rpre --corpus C0 \
      --corpus-file $SRC/$dom.jsonl --cheap $D/$dom.ladder.jsonl \
      --drafter "$DSPARK" --order 1 --cond both --out $OUT/$dom.srv1.jsonl \
      --target $TARGET --anchors 96 --paths 256 --split --kv-budget-gib 12 \
      2>&1 | grep -avE "$QUIET" | tail -6
done
echo ""; echo "===== GEMMA12B RE-ANCHORED ALL DONE ====="
