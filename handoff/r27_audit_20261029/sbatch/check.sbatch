#!/bin/bash
#SBATCH --job-name=r27_audit_check
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029/logs/%x_%j.out
# Tiny-case CPU check: SPG history-reference copy (max == production bitwise; min converges) and DEV rank/recall recompute.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029/check_$SLURM_JOB_ID
mkdir -p $O
cd $D
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u audit_check.py $O/CHECK.json
cp $O/CHECK.json $D/CHECK_$SLURM_JOB_ID.json
echo CHECK_DONE
