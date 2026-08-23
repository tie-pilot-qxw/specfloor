set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/calib_20260818/C0
OUT=/workspace/measurement_runs/rm_snis
mkdir -p $OUT
cd /workspace/DeepSpec
echo "===== R_m with BOTH estimators, gsm8k C0, 128 anchors, 64 mixed paths ====="
$PY -m measurement.probe_rm --corpus C0 \
    --corpus-file $D/gsm8k.jsonl --cheap $D/gsm8k.cheap.jsonl \
    --out $OUT/gsm8k.rm_snis.jsonl --budget 128 --mixed-paths 64 2>&1 | tail -25
echo "===== done ====="
