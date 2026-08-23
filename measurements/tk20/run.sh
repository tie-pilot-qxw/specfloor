set -uo pipefail
# 5.7 calibration: how much does a top-20 read cost?
#
# The V4-Pro floor is measured through an API that caps top_logprobs at 20, so
# every p_Z there is a top-20 truncation renormalised over the reported masses.
# Locally we can read the same anchors at top-256, which 4.x establishes is
# already indistinguishable from the full vocabulary. This arm re-reads the
# EXACT anchors, corpus files and --paths of measurement_runs/t1/run_t1_m256.sh
# with --top-k 20 and nothing else changed, so T^(0)_20 vs T^(0)_256 is a paired
# within-anchor difference and identifies the coarsening alone.
#
# Expected direction: truncation drops tail disagreements, so a top-20 read
# should bias T DOWN. That is the confound the frontier-scale comparison needs
# ruled out, since it reports the API floor as SMALLER than the 4B one.
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/tk20
mkdir -p $OUT; cd /workspace/DeepSpec

# Shared box: take a card only when one is genuinely free. Never crowd a
# neighbour -- a contended run produces timings nobody can interpret and,
# worse here, can OOM mid-corpus and silently drop anchors.
pick_gpu () {
  while :; do
    for g in $(seq 0 7); do
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ "$used" -lt 8000 ]; then echo $g; return; fi
    done
    sleep 120
  done
}
G=$(pick_gpu)
export CUDA_VISIBLE_DEVICES=$G
echo "== top-20 calibration arm on GPU $G"

run () {
  echo ""; echo "######## $1 (orders 0,1, M=256, top-20)"
  TK_PROFILE=1 $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/$2 --cheap $D/$3 \
      --out $OUT/$1.t01.tk20.jsonl --anchors 96 --paths 256 --top-k 20 \
      --rungs 0,1 --split 2>&1 | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -12
  echo "-- $1 rows: $(wc -l < $OUT/$1.t01.tk20.jsonl)"
}
run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""; echo "===== TOP-20 CALIBRATION DONE ====="
