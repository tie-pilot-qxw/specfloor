set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/t1
while pgrep -f "measurement.probe_tk" > /dev/null; do sleep 20; done
cd /workspace/DeepSpec
# The M-sensitivity arm. Identical anchors and seeds to the M=1024 run, so
# ESS_256 -> ESS_1024 and T1_256 -> T1_1024 are paired comparisons rather than
# two independent estimates. This is the arm that answers whether M=1024 was
# necessary or merely expensive -- the unique-path count cannot answer it,
# because repeated draws still inform the mass even after the unique count
# saturates.
run () {
  echo ""; echo "############ $1 (orders 0,1, M=256) ############"
  TK_PROFILE=1 $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.t01.m256.jsonl --anchors 96 --paths 256 --top-k 256 \
      --rungs 0,1 --split 2>&1 | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -12
  echo "-- $1 rows: $(wc -l < $OUT/$1.t01.m256.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== M256 ARM DONE ====="
