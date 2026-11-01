#!/bin/bash
#SBATCH --job-name=r28_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r28_selective_20261031/logs/%x_%j.out
# R28 read-only FIT diagnostics (design §4; tools/r28_fit_eval.py, truth only inside the evaluator) on existing final states:
# C-native (r27_candsrc cn_mix_t2i) and Q2 (r27_qpool q2_vcs_pool) f40/f60_s1001 state_last.pt, stage A. TRAIN split only.
# Submit with --qos/--partition on the command line.
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r28_selective_20261031/server.env
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
mkdir -p $R28_ROOT/diag
O9=/projects/EEG-foundation-model/yinghao/outputs_v9
for c in f40_s1001 f60_s1001; do
  N=$O9/r25_in2r_20261027/noise/f30k_precomp_0.${c:1:1}.npy
  for src in r27_candsrc_20261031/runs/$c/cn_mix_t2i r27_qpool_20261031/runs/$c/q2_vcs_pool; do
    S=$O9/$src/state_last.pt
    O=$R28_ROOT/diag/R28_FIT_EVAL_${c}_$(basename $src).json
    [ -f $O ] && { echo "skip $O"; continue; }
    $PYTHON_BIN -u -B $R28_ROOT/tools/r28_fit_eval.py --state $S --noise-file $N --data-path $DP --code-dir $R28_CODE --out $O
  done
done
echo R28_DIAG_DONE
