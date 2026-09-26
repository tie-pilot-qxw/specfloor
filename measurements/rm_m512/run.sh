set -uo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/rm_m512
mkdir -p $OUT
cd /workspace/DeepSpec

# domain : corpus file : cheap source (the M=512 ladder arm -- 247-252 anchors,
# 50-60 prompts survive the eps_R=0.05 selection, vs 10 in the pilot's cheap file)
run () {
  dom=$1; corp=$2; cheap=$3
  echo ""
  echo "############ $dom ############"
  $PY -m measurement.probe_rm --corpus C0 \
      --corpus-file $D/$corp --cheap $D/$cheap \
      --out $OUT/$dom.rm.jsonl --budget 256 --mixed-paths 512 2>&1 \
    | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -12
  echo "-- $dom rows: $(wc -l < $OUT/$dom.rm.jsonl)"
}

run gsm8k   gsm8k.jsonl   ladder.M512.jsonl
run mbpp    mbpp.jsonl    mbpp.ladder.M512.jsonl
run alpaca  alpaca.jsonl  alpaca.ladder.M512.jsonl
run arena8k arena8k.jsonl arena8k.ladder.M512.jsonl
echo ""
echo "===== ALL DOMAINS DONE ====="
