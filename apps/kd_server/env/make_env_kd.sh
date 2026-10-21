#!/bin/bash
# Record of how /home/infres/yinwang/CS_QMI/env_kd was created on 2026-10-20 (login node, no compute).
# Overlay venv: no package is installed or duplicated; torch etc. come from the existing V9/V10 env,
# which is NOT modified.  `env` is itself a uv venv, so `--system-site-packages` alone would expose the
# base interpreter's site-packages (only pip 25.3), not env's; the .pth line layers env's site-packages.
set -euo pipefail
BASE=/home/infres/yinwang/CS_QMI/.uv_python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12   # same base as env
ENV_KD=/home/infres/yinwang/CS_QMI/env_kd
PARENT_SITE=/home/infres/yinwang/CS_QMI/env/lib/python3.12/site-packages
if [ -e "$ENV_KD" ]; then echo "exists: $ENV_KD (refusing to recreate)"; exit 1; fi
"$BASE" -m venv --system-site-packages --without-pip "$ENV_KD"
echo "$PARENT_SITE" > "$ENV_KD/lib/python3.12/site-packages/cs_qmi_env_overlay.pth"
"$ENV_KD/bin/python" -m pip freeze > "$(dirname "$0")/pip_freeze_env_kd.txt"
"$ENV_KD/bin/python" -m pip freeze --all > "$(dirname "$0")/pip_freeze_all_env_kd.txt"
"$ENV_KD/bin/python" -m pip install --dry-run --no-index \
  -r /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019/requirements-core.txt
"$ENV_KD/bin/python" -m pip check
