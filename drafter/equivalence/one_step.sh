#!/usr/bin/env bash
# One full optimizer step through train.py (BaseTrainer: FSDP, torch.compile,
# BF16Optimizer, window normalisation, checkpoint save) in the original tree and in
# upstream DeepSpec + overlay, then a bitwise comparison of the saved weights and of
# the logged loss and gradient norm.
#
#   ORIG=<tree> OVL=<tree> DATA=<online_train.jsonl> OUT=<scratch dir> GPU=0 \
#   PY=python bash one_step.sh [config-name]
#
# Tiny overrides keep it on one shared GPU: 2 sequences, global batch 2 (two
# micro-batches of one sequence), 64 anchors, max length 1024, one step on a
# one-step schedule (so the step has a non-zero learning rate).  HOME is pointed
# at OUT so the configs' ~/checkpoints and ~/tensorboard resolve inside OUT.
# Both trees use the same compiled kernels (see run_all.sh).
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${ORIG:?}" "${OVL:?}" "${DATA:?}" "${OUT:?}"
NAME=${1:-attnconv_qwen3_4b_b7_10ep}
GPU=${GPU:-0}; PY=${PY:-python}
mkdir -p "$OUT"
head -n 2 "$DATA" > "$OUT/two_rows.jsonl"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

OPTS=(--opts "exp_name=onestep_${NAME}"
      --opts "train.max_train_steps=1" --opts "train.global_batch_size=2"
      --opts "model.num_anchors=64" --opts "data.max_length=1024"
      --opts "data.num_workers=1" --opts "logging.checkpointing_steps=1")
if grep -q "lr_schedule_steps" "$ORIG/config/dspark/$NAME.py"; then
  OPTS+=(--opts "train.lr_schedule_steps=1")
fi

run () {   # $1 tag  $2 tree  $3 cache  $4 port
  local home="$OUT/$1.home"
  rm -rf "$home"; mkdir -p "$home"
  ( cd "$2" && HOME="$home" DSPARK_TRAIN_DATA="$OUT/two_rows.jsonl" \
    TORCHINDUCTOR_CACHE_DIR="$3/inductor" TRITON_CACHE_DIR="$3/triton" \
    CUDA_VISIBLE_DEVICES=$GPU MASTER_ADDR=127.0.0.1 MASTER_PORT=$4 RANK=0 WORLD_SIZE=1 \
    "$PY" train.py --config "config/dspark/$NAME.py" "${OPTS[@]}" \
      --opts "data.train_data_paths=$OUT/two_rows.jsonl" > "$OUT/$1.log" 2>&1 )
  echo "$1: exit $?"
}

rm -rf "$OUT/cache.orig" "$OUT/cache.shared"
run orig "$ORIG" "$OUT/cache.orig" 29681
cp -r "$OUT/cache.orig" "$OUT/cache.shared"
run overlay "$OVL" "$OUT/cache.shared" 29682

"$PY" - "$OUT" "$NAME" <<'PY'
import hashlib, json, sys
from pathlib import Path
from safetensors.torch import load_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

out, name = Path(sys.argv[1]), sys.argv[2]
res = {"config": name}
sd, scalars = {}, {}
for tag in ("orig", "overlay"):
    home = out / f"{tag}.home"
    ck = home / f"checkpoints/deepspec/onestep_{name}/step_1/model.safetensors"
    sd[tag] = load_file(str(ck))
    ea = EventAccumulator(str(home / f"tensorboard/deepspec/onestep_{name}"),
                          size_guidance={"scalars": 0})
    ea.Reload()
    scalars[tag] = {k: [e.value for e in ea.Scalars(k)] for k in ea.Tags()["scalars"]}
    res[tag] = {"tensors": len(sd[tag]), "sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
                "loss": scalars[tag].get("train/loss"),
                "grad_norm": scalars[tag].get("train/grad_norm"),
                "lr": scalars[tag].get("train/lr")}
a, b = sd["orig"], sd["overlay"]
res["different_tensors"] = sorted(k for k in a if k not in b or not a[k].equal(b[k]))
res["only_overlay_tensors"] = sorted(set(b) - set(a))
res["bit_identical_weights"] = not res["different_tensors"] and not res["only_overlay_tensors"]
common = sorted(set(scalars["orig"]) & set(scalars["overlay"]))
res["scalars_compared"] = len(common)
res["scalars_different"] = [k for k in common if scalars["orig"][k] != scalars["overlay"][k]]
res["scalars_only_orig"] = sorted(set(scalars["orig"]) - set(scalars["overlay"]))
res["scalars_only_overlay"] = sorted(set(scalars["overlay"]) - set(scalars["orig"]))
(out / f"onestep_{name}.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("bit_identical_weights", "different_tensors",
                                      "scalars_compared", "scalars_different",
                                      "scalars_only_orig", "scalars_only_overlay")}, indent=1))
for tag in ("orig", "overlay"):
    print(tag, "loss", res[tag]["loss"], "grad_norm", res[tag]["grad_norm"], "lr", res[tag]["lr"])
PY
