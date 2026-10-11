#!/usr/bin/env bash
#SBATCH --job-name=r6crema_check_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 CREMA-D checks: unit tests (decoder cross-check with -fps_mode passthrough), the 5-fold actor split record (roles_cv5.json)
# and the video codec of every clip.  r6_av env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export PYTHON=/projects/EEG-foundation-model/yinghao/envs/r6_av/bin/python OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; rc=0
"$PYTHON" -m unittest tests/test_r6_gain_crema.py -v || rc=1
"$PYTHON" scripts/r6_gain_crema.py cvroles || rc=1
"$PYTHON" scripts/r6_gain_crema.py codecs --workers 16 || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
