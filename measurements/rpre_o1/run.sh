set -uo pipefail
# Order-1 companion to measurement_runs/rpre/run_rpre.sh. Every argument that
# touches the target's random stream (--paths, --kv-budget-gib, the corpus and
# anchor files) is IDENTICAL to that script, and the drafter chain draws from a
# separate generator, so the 256 rollout paths per anchor are bit-identical to
# the DFlash run. T^(0), R_DFlash, R_DSpark and T^(1) are therefore all read off
# one set of paths and every difference between them is paired.
export CUDA_VISIBLE_DEVICES=3
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/rpre_o1
DRAFT=deepseek-ai/dspark_qwen3_4b_block7
mkdir -p $OUT; cd /workspace/DeepSpec
run () {
  echo ""; echo "############ $1 (R, DSpark order 1) ############"
  $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --drafter $DRAFT --order 1 --cond both --out $OUT/$1.rpre1.jsonl \
      --anchors 96 --paths 256 --split --kv-budget-gib 4 2>&1 | grep -avE "Loading weights" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.rpre1.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== RPRE ORDER-1 ALL DONE ====="
