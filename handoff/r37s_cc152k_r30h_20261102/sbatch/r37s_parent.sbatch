#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/logs/%x_%j.out
# R37S CC152K seed1001 R30-style H parent: stage parent (5 H warmup epochs -> parent_state.pt, built once) then stage unit
# (H0 continuation, 40 co-train epochs, fork from the parent or resume its own state_last.pt). tools/run_r37s.py --execute
# checks code == INSTALL_MANIFEST, frozen PROTOCOL.json, and the hardware block (GPU type of the first attempt).
# Self-continuation: python under `timeout`; on timeout without DONE.json the job resubmits itself (same name/partition/
# time/nice), at most 4 attempts in total. No --dependency.
# First submission: --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 --time=10:00:00
set -euo pipefail
O=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102
OB=/projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1
OUT=$OB/runs/r37s_cc_s1001_H0
test -f $OUT/DONE.json && { echo "already DONE $OUT"; exit 0; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
mkdir -p $OB
N=$(cat $OB/ATTEMPTS 2>/dev/null || echo 0); N=$((N+1)); echo $N > $OB/ATTEMPTS
echo "attempt $N job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION host $(hostname)"
source $O/sbatch/stage_cc.sh
LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
T0=$(date +%s)
A="--root $O --code-dir $O/code_r37s --data-path $DP --out-base $OB --python $PY --execute"
set +e
rc=0
if [ ! -f $OB/parents/wp_cc_s1001_H_e5/PARENT.json ]; then
  timeout $LIM $PY -u -B $O/tools/run_r37s.py --stage parent $A; rc=$?
fi
if [ $rc -eq 0 ]; then
  LIM2=$(( LIM - ($(date +%s) - T0) ))
  timeout $LIM2 $PY -u -B $O/tools/run_r37s.py --stage unit $A; rc=$?
fi
set -e
if [ -f $OUT/DONE.json ]; then echo "R37S_PARENT_UNIT_DONE"; exit 0; fi
if [ $rc -eq 124 ] && [ $N -lt 4 ]; then
  J=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --qos=normal --partition=$SLURM_JOB_PARTITION --exclude=node51,node53,node60 --time=$(squeue -j $SLURM_JOB_ID -h -o %l) --nice=$(squeue -j $SLURM_JOB_ID -h -o %y) $O/sbatch/r37s_parent.sbatch)
  printf "%s\t%s\t%s\tcontinuation_of_%s (attempt %s timed out; partition %s)\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$J" "$SLURM_JOB_ID" "$N" "$SLURM_JOB_PARTITION" >> $O/SUBMITTED.tsv
  echo "timeout: continuation submitted $J"; exit 0
fi
echo "R37S_FAILED rc=$rc attempt=$N"; exit 1
