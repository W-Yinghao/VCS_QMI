#!/bin/bash
#SBATCH --job-name=r27q_fit_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qpool_20261031/logs/%x_%j.out
# Offline FIT correspondence-quality evaluation (tools/fit_eval_offline.py, truth only inside the evaluator) on the 8 finished
# R27-Q final states (state_last.pt). Read-only; TRAIN split only. Submit with --qos/--partition on the command line.
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r27_qpool_20261031/server.env
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
mkdir -p $R27Q_ROOT/fit_eval
for c in f40_s1001 f60_s1001; do
  N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.${c:1:1}.npy
  for a in q1_cos_pool q2_vcs_pool q3_bce_pool q4_snce_pool; do
    S=$R27Q_OUTPUT/$c/$a/state_last.pt
    O=$R27Q_ROOT/fit_eval/FIT_EVAL_${c}_${a}.json
    [ -f $O ] && { echo "skip $O"; continue; }
    [ -f $R27Q_OUTPUT/$c/$a/DONE.json ] || { echo "not done $c $a"; continue; }
    $PYTHON_BIN -u -B $R27Q_ROOT/tools/fit_eval_offline.py --state $S --noise-file $N --data-path $DP --code-dir $R27Q_CODE --out $O
  done
done
echo FIT_EVAL_DONE
