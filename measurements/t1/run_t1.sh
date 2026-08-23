set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/t1
mkdir -p $OUT; cd /workspace/DeepSpec
# M=1024 so the ESS gate stops selecting a subpopulation: at M=256 it dropped 43%
# of eligible cells at order 1, and the dropped cells are exactly the ones where
# the revealed token is most informative -- which would bias T^(0)-T^(1) low.
# rungs 0,1 in ONE run so the difference is paired on identical paths.
# top-k 256 matches the T^(0) protocol exactly.
run () {
  echo ""; echo "############ $1 (orders 0 and 1, M=1024) ############"
  TK_PROFILE=1 $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.t01.jsonl --anchors 96 --paths 1024 --top-k 256 \
      --rungs 0,1 --split 2>&1 | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -14
  echo "-- $1 rows: $(wc -l < $OUT/$1.t01.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== T1 ALL DONE ====="
