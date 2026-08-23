set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/t1
cd /workspace/DeepSpec
# mbpp/alpaca/arena8k died on an IndentationError: probe_tk.py was edited while
# the loop was running and each domain starts a fresh interpreter, so the three
# that had not launched yet picked up a half-applied edit. gsm8k predates it and
# is intact. Do not edit a module while a loop that re-imports it is in flight.
run () {
  echo ""; echo "############ $1 (orders 0,1, M=1024) ############"
  TK_PROFILE=1 $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.t01.jsonl --anchors 96 --paths 1024 --top-k 256 \
      --rungs 0,1 --split 2>&1 | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -12
  echo "-- $1 rows: $(wc -l < $OUT/$1.t01.jsonl)"
}
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== 1024 RERUN DONE ====="
