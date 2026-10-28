#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/reg_close_pairvcs_20261027/logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home_sota
cd /home/infres/yinwang/CS_QMI/handoff/reg_close_pairvcs_20261027
exec /projects/EEG-foundation-model/yinghao/envs/sota_matchers/bin/python -B close_reg.py "$@"
