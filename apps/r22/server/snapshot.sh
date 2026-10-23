#!/bin/bash
# Freeze apps/r22 (engine, entry points, tests, configs) at the current HEAD into a read-only directory and point
# code_current at it. Refuses a dirty apps/r22 tree, so every snapshot equals a commit.
set -euo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/r22/server/paths.env
cd "$R22_WT"
[ -z "$(git status --porcelain -- apps/r22/engine apps/r22/configs apps/r22/*.py apps/r22/tests)" ] || { echo "apps/r22 has uncommitted changes"; exit 1; }
C=$(git rev-parse --short=8 HEAD); D=$R22_OUT/code_$C
if [ ! -d "$D" ]; then
  mkdir -p "$D"
  git archive HEAD apps/r22/engine apps/r22/configs apps/r22/run_one.py apps/r22/guard.py apps/r22/grad_check.py apps/r22/tests \
    | tar -x -C "$D" --strip-components=2
  git rev-parse HEAD > "$D/CODE_COMMIT"
  chmod -R a-w "$D"
fi
ln -sfn "$D" "$R22_CODE_LINK"
echo "code_current -> $D"
