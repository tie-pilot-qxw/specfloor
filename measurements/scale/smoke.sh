set -uo pipefail
# Whole run_target.sh chain at toy size, to fail on a bad flag in minutes rather
# than after the corpus stage has burned an afternoon. Numbers from this are
# meaningless -- 2 prompts, 64 paths -- only the exit codes matter.
NAME=$1; TARGET=$2; DFLASH=$3; DSPARK=$4; GPU=${5:-5}
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/sglang017-env/bin/python
D=/workspace/measurement_runs/scale/_smoke/$NAME
mkdir -p $D; cd /workspace/DeepSpec
set -e
step () { echo ""; echo "######## $1"; shift; "$@"; }
step "1 corpus" $PY -m measurement.corpus --corpus C0 --domain gsm8k \
    --out $D/gsm8k.jsonl --target $TARGET --prompts 2 --max-new-tokens 512
step "2 anchors" $PY -m measurement.anchors --corpus-file $D/gsm8k.jsonl \
    --out $D/gsm8k.anchors.jsonl --budget 8
step "3 cheap" $PY -m measurement.probe_cheap --corpus C0 --corpus-file $D/gsm8k.jsonl \
    --anchors $D/gsm8k.anchors.jsonl --out $D/gsm8k.ladder.jsonl \
    --target $TARGET --m-base 64 --m-max 64
step "4 tk" $PY -m measurement.probe_tk --corpus C0 --corpus-file $D/gsm8k.jsonl \
    --cheap $D/gsm8k.ladder.jsonl --out $D/gsm8k.t01.jsonl --target $TARGET \
    --anchors 4 --paths 64 --top-k 256 --rungs 0,1 --split
step "5 rpre order0" $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/gsm8k.jsonl \
    --cheap $D/gsm8k.ladder.jsonl --drafter "$DFLASH" --order 0 --cond both \
    --out $D/gsm8k.srv0.jsonl --target $TARGET --anchors 4 --paths 32 --split --kv-budget-gib 12
step "6 rpre order1" $PY -m measurement.probe_rpre --corpus C0 --corpus-file $D/gsm8k.jsonl \
    --cheap $D/gsm8k.ladder.jsonl --drafter "$DSPARK" --order 1 --cond both \
    --out $D/gsm8k.srv1.jsonl --target $TARGET --anchors 4 --paths 32 --split --kv-budget-gib 12
step "7 reports" $PY -m measurement.tk_report --tk "$D/gsm8k.t01.jsonl"
$PY -m measurement.srv_report --rpre "$D/gsm8k.srv0.jsonl"
echo ""; echo "===== $NAME SMOKE OK ====="
