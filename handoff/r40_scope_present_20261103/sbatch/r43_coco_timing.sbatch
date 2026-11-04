#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R43 A: COCO memory / time measurement (replaces the in-build timing step of 1084286, which did not run: /usr/bin/time absent on the node).
# 30-update runs of R43_A_C60_IN2R_H1001_N_{frozen,joint}_J in the trainer's gate mode (scratch r43_parents/timing/c60in_s1001; not formal, never reused);
# host RSS sampled every 5 s (peak printed); GPU peak and seconds per update from the gate-mode RESULT/TRAIN_HISTORY.
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for K in frozen joint; do
  t1=$(date +%s)
  $PY -u -B $W/r43o/train_unit.py --units R43_A_C60_IN2R_H1001_N_${K}_J --gate-out /projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/r43_parents/timing/c60in_s1001 --gate-steps 30 > $W/logs/r43_timing_${K}_$SLURM_JOB_ID.log 2>&1 &
  P=$!; peak=0
  while kill -0 $P 2>/dev/null; do r=$(ps -o rss= -p $P 2>/dev/null | tr -d ' '); [ -n "$r" ] && [ "$r" -gt "$peak" ] && peak=$r; sleep 5; done
  wait $P; rc=$?
  grep -E "R41_UNIT_DONE|R41_JOB_END|Traceback|Error" $W/logs/r43_timing_${K}_$SLURM_JOB_ID.log | tail -3
  echo "R43_TIMING $K rc=$rc elapsed_s $(( $(date +%s)-t1 )) peak_rss_GiB $(( peak/1048576 ))"
done
