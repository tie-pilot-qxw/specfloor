set -uo pipefail
# Per-slot rejection-risk breakdown for OUR 10-epoch attnconv drafter, against the
# recorded DSpark order-1 run in ../rpre_o1.
#
# EVERY argument that touches the target's random stream -- --corpus, --corpus-file,
# --cheap, --paths, --kv-budget-gib -- is copied verbatim from ../rpre_o1/run.sh, and
# the drafter's chain draws from its own generator, so the 256 rollout paths per anchor
# are bit-identical to the DSpark run and T^(1) is literally the same number. Every
# difference between the two files is then the drafter and nothing else.
#
# GATE FIRST. This runs specfloor's copy of the probe; the recorded DSpark numbers came
# from DeepSpec/measurement, which specfloor later refactored (imports behind _deepspec,
# one added no_grad). `--gate` re-runs DSPARK itself on gsm8k through THIS code and the
# reported numbers are compared against the recorded ones before anything is believed.
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPECFLOOR_DEEPSPEC=/workspace/DeepSpec PYTHONPATH=/workspace/specfloor
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/rpre_o1_ours
DRAFT=${DRAFT:-/workspace/checkpoints/deepspec/attnconv_b7_qwen3_4b_10ep/step_26160}
mkdir -p $OUT; cd /workspace/specfloor
run () {
  echo ""; echo "############ $1 ($4) ############"
  $PY -m specfloor.probe_rpre --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --drafter "$5" --order 1 --cond both --out $OUT/$1.rpre1.jsonl \
      --anchors 96 --paths 256 --split --kv-budget-gib 4 2>&1 \
      | grep -avE "Loading weights" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.rpre1.jsonl)"
}
if [ "${1:-}" = "--gate" ]; then
  run gate_dspark_gsm8k gsm8k.jsonl ladder.M512.jsonl \
      "DSpark through specfloor -- must reproduce ../rpre_o1/gsm8k" \
      deepseek-ai/dspark_qwen3_4b_block7
  exit 0
fi
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl        "ours" "$DRAFT"
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl   "ours" "$DRAFT"
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl "ours" "$DRAFT"
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl "ours" "$DRAFT"
echo ""; echo "===== RPRE ORDER-1 (OURS) ALL DONE ====="
