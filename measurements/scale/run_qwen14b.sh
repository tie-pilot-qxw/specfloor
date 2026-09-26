#!/bin/bash
# Qwen3-14B: the third point INSIDE the Qwen family.
#
# The scale section's own limitation is that size and family are confounded --
# 4B and 8B are Qwen, 12B is Gemma, so the objection "the gap closes as the
# drafter grows" can only be answered on three points spanning two families.
# 14B removes the confound: 4B -> 8B -> 14B, one family, one drafter recipe.
#
# Same pipeline as run_target.sh, on specfloor rather than the in-tree
# measurement package, and with the target read from a local path because
# Qwen3-14B was fetched as a plain directory rather than into the HF cache.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/specfloor
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec
export SPECFLOOR_EVAL_ROOT=/workspace/DeepSpec/eval_datasets
PY=/workspace/sglang017-env/bin/python

# 14B in bf16 is 28 GiB, and sglang's auto mem-fraction left no room for a KV
# cache on the first attempt ("minimum viable = 0.5261"). Pin it rather than
# let it be derived from whatever the card happened to have free, so the run
# does not depend on a neighbour.
MEMF=0.72

TARGET=/workspace/models/Qwen3-14B
DFLASH=deepseek-ai/dflash_qwen3_14b_block7
DSPARK=deepseek-ai/dspark_qwen3_14b_block7
D=/workspace/measurement_runs/scale/qwen14b/C0
OUT=/workspace/measurement_runs/scale/qwen14b
mkdir -p $D

QUIET='Loading|Fetching|it/s\]|UserWarning|SOLUTION|flex_attention|_warn_once|debug your score_mod|This will allow|Capturing|Multi-thread'

# 1-3: corpus, anchors, and the M=512 ladder the probes select anchors from.
# arena8k is arena-hard-v2 regenerated at an 8192 cap; the file keeps the
# original name.
for dom in gsm8k mbpp alpaca arena8k; do
  case $dom in arena8k) src=arena-hard-v2;; *) src=$dom;; esac
  echo ""; echo "######## corpus $dom"
  [ -s $D/$dom.jsonl ] || $PY -m specfloor.corpus --corpus C0 --domain $src \
      --out $D/$dom.jsonl --target $TARGET --prompts 96 --max-new-tokens 8192 \
      --mem-fraction $MEMF \
      2>&1 | grep -avE "$QUIET" | tail -4
  [ -s $D/$dom.anchors.jsonl ] || $PY -m specfloor.anchors \
      --corpus-file $D/$dom.jsonl --out $D/$dom.anchors.jsonl --budget 256 \
      2>&1 | tail -3
  [ -s $D/$dom.ladder.jsonl ] || $PY -m specfloor.probe_cheap --corpus C0 \
      --corpus-file $D/$dom.jsonl --anchors $D/$dom.anchors.jsonl \
      --out $D/$dom.ladder.jsonl --target $TARGET --m-base 512 --m-max 512 \
      --mem-fraction $MEMF \
      2>&1 | grep -avE "$QUIET" | tail -4
done

# 4: the floor, both orders, on the serving engine.
for dom in gsm8k mbpp alpaca arena8k; do
  echo ""; echo "######## tk $dom"
  [ -s $OUT/$dom.t01.jsonl ] || $PY -m specfloor.probe_tk --corpus C0 \
      --corpus-file $D/$dom.jsonl --cheap $D/$dom.ladder.jsonl \
      --out $OUT/$dom.t01.jsonl --target $TARGET \
      --anchors 96 --paths 1024 --top-k 256 --rungs 0,1 --split --mem-fraction $MEMF \
      2>&1 | grep -avE "$QUIET" | tail -6
done

# 5-6: the gaps. Local HF forward, full vocabulary, exact TV.
for dom in gsm8k mbpp alpaca arena8k; do
  echo ""; echo "######## rpre $dom order 0"
  [ -s $OUT/$dom.srv0.jsonl ] || $PY -m specfloor.probe_rpre --corpus C0 \
      --corpus-file $D/$dom.jsonl --cheap $D/$dom.ladder.jsonl \
      --drafter "$DFLASH" --order 0 --out $OUT/$dom.srv0.jsonl --target $TARGET \
      --anchors 96 --paths 256 --split --kv-budget-gib 12 \
      2>&1 | grep -avE "$QUIET" | tail -8
  echo "######## rpre $dom order 1"
  [ -s $OUT/$dom.srv1.jsonl ] || $PY -m specfloor.probe_rpre --corpus C0 \
      --corpus-file $D/$dom.jsonl --cheap $D/$dom.ladder.jsonl \
      --drafter "$DSPARK" --order 1 --cond both --out $OUT/$dom.srv1.jsonl \
      --target $TARGET --anchors 96 --paths 256 --split --kv-budget-gib 12 \
      2>&1 | grep -avE "$QUIET" | tail -8
done

echo ""; echo "===== QWEN3-14B ALL DONE ====="
