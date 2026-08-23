set -uo pipefail
# The whole 4B pipeline, parameterised by target and drafter pair, for a second
# scale (Qwen3-8B) and a second architecture family (Gemma-4-12B).
#
#   bash run_target.sh qwen8b   Qwen/Qwen3-8B      deepseek-ai/dflash_qwen3_8b_block7  deepseek-ai/dspark_qwen3_8b_block7   5
#   bash run_target.sh gemma12b google/gemma-4-12B-it deepseek-ai/dflash_gemma4_12b_block7 deepseek-ai/dspark_gemma4_12b_block7 5
#
# google/gemma-4-12b is a config-only stub with no weights; the served checkpoint
# is google/gemma-4-12B-it, which is also the one carrying a chat template, and
# C0 is a chat-mode protocol.
#
# THE CORPUS CANNOT BE SHARED ACROSS TARGETS. A floor lives at the positions the
# target itself would occupy, so anchoring one model inside another model's text
# measures its uncertainty about continuing someone else's writing -- out of
# distribution, and not the quantity that governs its own speculative decoding.
# Prompts are shared, so the domain axis stays comparable; responses are not.
#
# Stages, in order, because each consumes the last:
#   1 corpus      the target writes its own responses under C0
#   2 anchors     stratified (context bucket x relative position), known pi
#   3 cheap       the M=512 ladder file the probes select anchors from
#   4 tk          T^(0) and T^(1), sglang backend, top-256 SNIS, M=1024
#   5 rpre        R and G for the product-measure drafter, full vocabulary
#   6 rpre order1 R, T^(1) and exposure for the markov head, plus serving weights
set -x
NAME=$1; TARGET=$2; DFLASH=$3; DSPARK=$4; GPU=${5:-5}
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/scale/$NAME/C0
OUT=/workspace/measurement_runs/scale/$NAME
mkdir -p $D; cd /workspace/DeepSpec

# arena8k is arena-hard-v2's prompts regenerated at an 8192 cap: at 2048 that
# domain came back 14.6% right-censored, which biases its anchors toward the
# early part of long answers. The dataset file keeps the original name.
for dom in gsm8k mbpp alpaca arena8k; do
  case $dom in arena8k) src=arena-hard-v2;; *) src=$dom;; esac
  [ -s $D/$dom.jsonl ] || $PY -m measurement.corpus --corpus C0 --domain $src \
      --out $D/$dom.jsonl --target $TARGET --prompts 96 --max-new-tokens 8192
  [ -s $D/$dom.anchors.jsonl ] || $PY -m measurement.anchors \
      --corpus-file $D/$dom.jsonl --out $D/$dom.anchors.jsonl --budget 256
  [ -s $D/$dom.ladder.jsonl ] || $PY -m measurement.probe_cheap --corpus C0 \
      --corpus-file $D/$dom.jsonl --anchors $D/$dom.anchors.jsonl \
      --out $D/$dom.ladder.jsonl --target $TARGET --m-base 512 --m-max 512
done

for dom in gsm8k mbpp alpaca arena8k; do
  $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$dom.jsonl \
      --cheap $D/$dom.ladder.jsonl --out $OUT/$dom.t01.jsonl --target $TARGET \
      --anchors 96 --paths 1024 --top-k 256 --rungs 0,1 --split
done

for dom in gsm8k mbpp alpaca arena8k; do
  $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/$dom.jsonl \
      --cheap $D/$dom.ladder.jsonl --drafter "$DFLASH" --order 0 --cond both \
      --out $OUT/$dom.srv0.jsonl --target $TARGET \
      --anchors 96 --paths 256 --split --kv-budget-gib 12
  $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/$dom.jsonl \
      --cheap $D/$dom.ladder.jsonl --drafter "$DSPARK" --order 1 --cond both \
      --out $OUT/$dom.srv1.jsonl --target $TARGET \
      --anchors 96 --paths 256 --split --kv-budget-gib 12
done
set +x
echo "===== $NAME ALL DONE ====="
