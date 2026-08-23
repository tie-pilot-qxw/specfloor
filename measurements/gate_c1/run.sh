set -euo pipefail
export CUDA_VISIBLE_DEVICES=4
PY=/workspace/sglang017-env/bin/python
OUT=/workspace/measurement_runs/gate_c1
cd /workspace/DeepSpec
mkdir -p $OUT

echo "===== [1/4] C1 corpus, gsm8k, 96 prompts ====="
$PY -m measurement.corpus --corpus C1 --domain gsm8k \
    --out $OUT/gsm8k.C1.jsonl --prompts 96 --max-new-tokens 2048 2>&1 | tail -20

echo "===== [2/4] verify hf (transformers) ====="
$PY -m measurement.verify_backend --phase hf  --corpus-file $OUT/gsm8k.C1.jsonl --out $OUT/v_c1 2>&1 | tail -15

echo "===== [3/4] verify sgl (sglang) ====="
$PY -m measurement.verify_backend --phase sgl --corpus-file $OUT/gsm8k.C1.jsonl --out $OUT/v_c1 2>&1 | tail -15

echo "===== [4/4] compare ====="
$PY -m measurement.verify_backend --phase cmp --out $OUT/v_c1 2>&1 | tail -30
echo "===== GATE EXIT: $? ====="
