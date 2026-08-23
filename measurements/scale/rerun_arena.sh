#!/bin/bash
# Re-run the gap probes on arena-hard for the two larger targets.
#
# The original scale run lost anchors to OOM on this domain only -- 39 of 96 on
# Gemma-4-12B, 9 of 96 on Qwen3-8B -- because the hidden-state pass in
# probe_rpre.main() was outside no_grad and retained ~16 MiB per token of
# autograd graph. With that fixed the whole context range fits, so the domain
# can be measured rather than dropped.
#
# Writes beside the originals with a .new suffix. The per-anchor random stream
# is seeded from (prompt_id, t, SEED) and the chunking is a deterministic
# function of (context, K, budget), so every anchor the old run DID keep must
# reproduce exactly. compare_rerun.py checks that before anything is replaced.
#
# The budget therefore has to MATCH the run being replaced, and the two targets
# do not share one: gemma12b's arena8k came from the main sweep at 12 GiB,
# qwen8b's from a separate rerun at 24 GiB (arena8b_rerun.log names it in its
# own headers). Every recorded chunk value confirms it. Passing one budget to
# both halves the chunk on one of them, which redraws the paths and moves every
# number -- a real difference, but not the one this rerun is testing.
set -uo pipefail
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export HF_HUB_OFFLINE=1
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/workspace/specfloor
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec
PY=/workspace/sglang017-env/bin/python

run () {           # tag budget-gib target dflash dspark
  tag=$1; gib=$2; target=$3; dflash=$4; dspark=$5
  D=/workspace/measurement_runs/scale/$tag/C0
  OUT=/workspace/measurement_runs/scale/$tag
  echo ""; echo "######## $tag  order 0 (DFlash) @${gib}GiB"
  $PY -m specfloor.probe_rpre --corpus C0 --corpus-file $D/arena8k.jsonl \
      --cheap $D/arena8k.ladder.jsonl --out $OUT/arena8k.srv0.jsonl.new \
      --target "$target" --drafter "$dflash" --order 0 \
      --anchors 96 --paths 256 --split --kv-budget-gib $gib 2>&1 \
    | grep -avE "Loading|Fetching|it/s\]|UserWarning|SOLUTION|flex_attention|_warn_once|debug your score_mod|This will allow" | tail -12
  echo "-- rows: $(wc -l < $OUT/arena8k.srv0.jsonl.new)"

  echo ""; echo "######## $tag  order 1 (DSpark) @${gib}GiB"
  $PY -m specfloor.probe_rpre --corpus C0 --corpus-file $D/arena8k.jsonl \
      --cheap $D/arena8k.ladder.jsonl --out $OUT/arena8k.srv1.jsonl.new \
      --target "$target" --drafter "$dspark" --order 1 --cond both \
      --anchors 96 --paths 256 --split --kv-budget-gib $gib 2>&1 \
    | grep -avE "Loading|Fetching|it/s\]|UserWarning|SOLUTION|flex_attention|_warn_once|debug your score_mod|This will allow" | tail -12
  echo "-- rows: $(wc -l < $OUT/arena8k.srv1.jsonl.new)"
}

run qwen8b   24 Qwen/Qwen3-8B          deepseek-ai/dflash_qwen3_8b_block7   deepseek-ai/dspark_qwen3_8b_block7
run gemma12b 12 google/gemma-4-12B-it  deepseek-ai/dflash_gemma4_12b_block7 deepseek-ai/dspark_gemma4_12b_block7
echo ""; echo "===== ARENA RERUN DONE ====="
