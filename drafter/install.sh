#!/usr/bin/env bash
# Build a DeepSpec checkout with the prefix-attention drafter overlay applied.
#
#   bash drafter/install.sh <target-dir> [<deepspec-repo-url-or-path>]
#
# Clones deepseek-ai/DeepSpec (or reuses an existing clone at <target-dir>), checks
# out the pinned base commit, and copies drafter/overlay/ on top.  The overlay only
# adds or replaces files; it never deletes upstream ones.  Refuses to touch a clone
# with local modifications.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEST=${1:?usage: install.sh <target-dir> [deepspec-repo-url-or-path]}
URL=${2:-https://github.com/deepseek-ai/DeepSpec.git}
BASE=afdfa7c9382a3341a3e6f17756dd816da79f132c

if [ ! -d "$DEST/.git" ]; then
  git -c advice.detachedHead=false clone --quiet "$URL" "$DEST"
fi
if [ -n "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]; then
  echo "error: $DEST has local modifications; use a fresh directory" >&2
  exit 1
fi
git -C "$DEST" fetch --quiet origin "$BASE" 2>/dev/null || true
git -C "$DEST" checkout --quiet --detach "$BASE"
cp -R "$HERE/overlay/." "$DEST/"
echo "DeepSpec @ ${BASE:0:12} + prefix-attention drafter overlay -> $DEST"
echo "next: cd $DEST && pip install -r requirements.txt && python -m pytest tests -q"
