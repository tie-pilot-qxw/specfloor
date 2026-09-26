#!/usr/bin/env bash
# Resume a drafter run after an interruption, refusing the resumes that would not
# continue the same run.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 DSPARK_TRAIN_DATA=<online_train.jsonl> \
#     bash scripts/train/resume_drafter.sh config/dspark/attnconv_qwen3_4b_b7_10ep.py
#
# The trainer resumes from <checkpoint_dir>/step_latest on its own.  This checks
# first that (a) the checkpoint still holds optimizer state (keep_last_n_checkpoints
# drops it from older saves), and (b) lr_cooldown_on_resume is off, since it would
# turn a crash resume into an early anneal.  A different GPU count is allowed only
# under sharding_strategy="no_shard" (the trainer validates this).
set -euo pipefail
CFG=${1:?usage: resume_drafter.sh <config.py>}
shift
CKROOT=$(python - "$CFG" <<'PY'
import sys
from deepspec.utils import load_config, parse_opts_to_config
print(parse_opts_to_config([], load_config(sys.argv[1])).logging.checkpoint_dir)
PY
)
LATEST="$CKROOT/step_latest"
test -e "$LATEST" || { echo "nothing to resume: $LATEST does not exist" >&2; exit 1; }
ls "$LATEST"/training_state.rank*.pt >/dev/null 2>&1 || {
  echo "$(readlink -f "$LATEST") has no optimizer state; it cannot be resumed" >&2; exit 1; }
if grep -qE '^\s*lr_cooldown_on_resume=True' "$CFG"; then
  echo "lr_cooldown_on_resume=True in $CFG; a crash resume must keep it False" >&2; exit 1
fi
echo "resuming from $(readlink -f "$LATEST")"
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_drafter.sh" "$CFG" "$@"
