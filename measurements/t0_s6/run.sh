set -uo pipefail
# Rung 0 at M=1024, covering ALL K slots including the last one.
#
# Rung m>=1 scores slots m..K-1, so slot K-1 already carries a T^(1) value in
# measurement_runs/t1/*.t01.jsonl. Rung 0 is what was one slot short: a path is
# K-1 tokens, so prefix+path creates next-token positions for slots 0..K-2 only.
# probe_tk now appends one trailing token, which manufactures the position for
# slot K-1 without any row being conditioned on it, so slots 0..K-2 are scored
# from byte-identical inputs and slot K-1 is new. Everything else -- anchors,
# M, top-k, seeds -- matches measurement_runs/t1/run_t1.sh exactly, so the
# earlier slots double as a reproducibility check on the path draw: if they come
# back identical, T^(0)_6 here pairs with T^(1)_6 there on the same paths.
export CUDA_VISIBLE_DEVICES=3
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/t0_s6
mkdir -p $OUT; cd /workspace/DeepSpec
run () {
  echo ""; echo "############ $1 (rung 0, M=1024, all slots) ############"
  $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.t0.jsonl --anchors 96 --paths 1024 --top-k 256 \
      --rungs 0 --split 2>&1 | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -6
  echo "-- $1 rows: $(wc -l < $OUT/$1.t0.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== T0 SLOT-6 ALL DONE ====="
