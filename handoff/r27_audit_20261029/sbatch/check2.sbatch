#!/bin/bash
#SBATCH --job-name=r27_audit_check2
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029/logs/%x_%j.out
# Tiny-case CPU check 2: exact ball-halfspace projection, constrained solver, task scores, table-1 transitions.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029/check2_$SLURM_JOB_ID
mkdir -p $O
cd $D
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u audit_check2.py $O/CHECK2.json
cp $O/CHECK2.json $D/CHECK2_$SLURM_JOB_ID.json
echo CHECK2_DONE
