set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/rpre
mkdir -p $OUT; cd /workspace/DeepSpec
run () {
  echo ""; echo "############ $1 (R_pre, DFlash rung 0) ############"
  $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --drafter /workspace/dflash_sgl --out $OUT/$1.rpre.jsonl \
      --anchors 96 --paths 256 --split --kv-budget-gib 4 2>&1 | grep -avE "Loading weights" | tail -5
  echo "-- $1 rows: $(wc -l < $OUT/$1.rpre.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== RPRE ALL DONE ====="
