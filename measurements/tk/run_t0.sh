set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/tk
mkdir -p $OUT; cd /workspace/DeepSpec
run () {
  echo ""; echo "############ $1 (rung 0) ############"
  $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.t0.jsonl --anchors 96 --paths 256 --top-k 256 \
      --rungs 0 --split 2>&1 | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.t0.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== T0 ALL DONE ====="
