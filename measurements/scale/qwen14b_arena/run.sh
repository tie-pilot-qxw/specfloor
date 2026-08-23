#!/bin/bash
# Qwen3-14B, arena-hard only, regenerated at a cap the censor gate passes.
#
# At --max-new-tokens 8192 this domain censored 1.0417% of responses against a
# protocol limit of 0.50%. Right-censoring truncates the longest responses,
# which carry the deep context buckets and the late relative-position stratum,
# so both the block weights and the stratification move. The cap is raised to
# SAFETY_MAX_NEW_TOKENS and the whole domain is regenerated; the other three
# domains censored 0.0000% and are untouched.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/specfloor
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec
export SPECFLOOR_EVAL_ROOT=/workspace/DeepSpec/eval_datasets
PY=/workspace/sglang017-env/bin/python
MEMF=0.72
TARGET=/workspace/models/Qwen3-14B
DFLASH=deepseek-ai/dflash_qwen3_14b_block7
DSPARK=deepseek-ai/dspark_qwen3_14b_block7
D=/workspace/measurement_runs/scale/qwen14b_arena/C0
OUT=/workspace/measurement_runs/scale/qwen14b_arena
mkdir -p $D
QUIET='Loading|Fetching|it/s\]|UserWarning|SOLUTION|flex_attention|_warn_once|debug your score_mod|This will allow|Capturing|Multi-thread'

# Every stage feeds the next, so a stage that produced nothing must stop the
# script. Piping a command into grep hands the pipeline grep's exit status, so
# the guard has to be on the OUTPUT FILE, not on $?.
need () { [ -s "$1" ] || { echo "!! stage produced nothing: $1"; exit 1; }; }

# corpus.py now exits nonzero if the gate fails, so `set -e` semantics are
# restored by checking explicitly rather than by trusting the exit status of a
# pipeline whose last element is grep.
echo "######## corpus arena8k @ 16384"
if [ ! -s $D/arena8k.jsonl ]; then
  $PY -m specfloor.corpus --corpus C0 --domain arena-hard-v2 \
      --out $D/arena8k.jsonl --target $TARGET --prompts 96 \
      --max-new-tokens 16384 --mem-fraction $MEMF > $D/corpus.log 2>&1
  rc=$?
  grep -E "censor rate|!!" $D/corpus.log | tail -3
  [ $rc -ne 0 ] && { echo "GATE FAILED -- stopping"; exit 1; }
fi

need $D/arena8k.jsonl
echo "######## anchors + ladder"
[ -s $D/arena8k.anchors.jsonl ] || $PY -m specfloor.anchors \
    --corpus-file $D/arena8k.jsonl --out $D/arena8k.anchors.jsonl --budget 256 \
    --target $TARGET 2>&1 | tail -3
[ -s $D/arena8k.ladder.jsonl ] || $PY -m specfloor.probe_cheap --corpus C0 \
    --corpus-file $D/arena8k.jsonl --anchors $D/arena8k.anchors.jsonl \
    --out $D/arena8k.ladder.jsonl --target $TARGET --m-base 512 --m-max 512 \
    --mem-fraction $MEMF 2>&1 | grep -avE "$QUIET" | tail -4
need $D/arena8k.anchors.jsonl
need $D/arena8k.ladder.jsonl

echo "######## tk"
[ -s $OUT/arena8k.t01.jsonl ] || $PY -m specfloor.probe_tk --corpus C0 \
    --corpus-file $D/arena8k.jsonl --cheap $D/arena8k.ladder.jsonl \
    --out $OUT/arena8k.t01.jsonl --target $TARGET \
    --anchors 96 --paths 1024 --top-k 256 --rungs 0,1 --split --mem-fraction $MEMF \
    2>&1 | grep -avE "$QUIET" | tail -6
need $OUT/arena8k.t01.jsonl

echo "######## rpre order 0 / order 1"
[ -s $OUT/arena8k.srv0.jsonl ] || $PY -m specfloor.probe_rpre --corpus C0 \
    --corpus-file $D/arena8k.jsonl --cheap $D/arena8k.ladder.jsonl \
    --drafter "$DFLASH" --order 0 --cond both --out $OUT/arena8k.srv0.jsonl \
    --target $TARGET --anchors 96 --paths 256 --split --kv-budget-gib 12 \
    2>&1 | grep -avE "$QUIET" | tail -6
[ -s $OUT/arena8k.srv1.jsonl ] || $PY -m specfloor.probe_rpre --corpus C0 \
    --corpus-file $D/arena8k.jsonl --cheap $D/arena8k.ladder.jsonl \
    --drafter "$DSPARK" --order 1 --cond both --out $OUT/arena8k.srv1.jsonl \
    --target $TARGET --anchors 96 --paths 256 --split --kv-budget-gib 12 \
    2>&1 | grep -avE "$QUIET" | tail -6
need $OUT/arena8k.srv0.jsonl
need $OUT/arena8k.srv1.jsonl
echo ""; echo "===== QWEN14B ARENA REGENERATED ====="
