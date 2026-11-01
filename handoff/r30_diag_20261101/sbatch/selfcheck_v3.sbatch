#!/bin/bash
#SBATCH --job-name=r30diag_cpu3
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101/logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
D=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101
cd $D
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B tools/selfcheck_v3.py /home/infres/yinwang/CS_QMI/handoff/r29_listwise_20261101/code_r29 \
  /home/infres/yinwang/CS_QMI/r30_review_plan/VCS_R29_Review_R30_Plan_20261101/reference/objective_checks.py $D/SELFCHECK_V3_${SLURM_JOB_ID}.json
cp $D/SELFCHECK_V3_${SLURM_JOB_ID}.json $D/SELFCHECK_V3.json
