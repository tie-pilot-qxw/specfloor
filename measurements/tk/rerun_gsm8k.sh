set -uo pipefail
# gsm8k's first pass shared its output file with a still-live process from the
# previous (dense-barycentre) run: two writers, independent file offsets, so
# records got spliced at overlapping byte ranges. Two lines failed to parse and
# an unknown number of the rest are silent splices, so the domain is re-run
# single-writer rather than filtered. The damaged file is kept as
# gsm8k.t0.jsonl.CORRUPT-two-writers.
while pgrep -f "measurement.probe_tk" > /dev/null; do sleep 20; done
export CUDA_VISIBLE_DEVICES=4
cd /workspace/DeepSpec
/workspace/sglang017-env/bin/python -m measurement.probe_tk --corpus C0 \
  --corpus-file /workspace/measurement_runs/calib_20260818/C0/gsm8k.jsonl \
  --cheap /workspace/measurement_runs/calib_20260818/C0/ladder.M512.jsonl \
  --out /workspace/measurement_runs/tk/gsm8k.t0.jsonl \
  --anchors 96 --paths 256 --top-k 256 --rungs 0 --split 2>&1 \
  | grep -avE "Capturing|Multi-thread|it/s\]$" | tail -6
echo "-- gsm8k rows: $(wc -l < /workspace/measurement_runs/tk/gsm8k.t0.jsonl)"
echo "===== GSM8K RERUN DONE ====="
