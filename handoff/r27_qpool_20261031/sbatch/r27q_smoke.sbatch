#!/bin/bash
#SBATCH --job-name=r27q_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qpool_20261031/logs/%x_%j.out
# R27-Q real-data GPU smoke + gate (snapshot code_q, frozen config): q1..q4 from the REAL f40_s1001 common prefix,
# 1 co-train epoch x 30 iterations (separate smoke dir, lam 0), then gate/r27q_gate.py from the R27-S smoke state
# -> SMOKE/GPU_GATE.json; then a 3-batch run of the offline FIT evaluator (tools check only, not part of all_pass).
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r27_qpool_20261031/server.env
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
cd $R27Q_CODE
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qpool_20261031/smoke_q_${SLURM_JOB_ID}
C="--data_path $DP --noise_file $R27Q_NOISE40 --noise_ratio 0.4 --seed 1001 --k 5 --log_step 10 --max_iters 30"
for arm in q1_cos_pool q2_vcs_pool q3_bce_pool q4_snce_pool; do
  $PYTHON_BIN -u -B r27_run.py --arm $arm --lam 0.0 $C --output_dir $S/$arm --resume $R27Q_PREFIX40 --num_epochs 2
done
set +e
$PYTHON_BIN -u -B $R27Q_ROOT/gate/r27q_gate.py $S $R27Q_NOISE40 $DP $R27Q_GATE_STATE $R27Q_CODE
rc=$?
cp $S/GPU_GATE.json $R27Q_ROOT/GPU_GATE_${SLURM_JOB_ID}.json
cp $S/GPU_GATE.json $R27Q_ROOT/GPU_GATE.json
$PYTHON_BIN -u -B $R27Q_ROOT/tools/fit_eval_offline.py --state $R27Q_GATE_STATE --noise-file $R27Q_NOISE40 --data-path $DP \
  --code-dir $R27Q_CODE --out $S/FIT_EVAL_SMOKE.json --n-batches 3
rce=$?
set -e
cp $S/FIT_EVAL_SMOKE.json $R27Q_ROOT/FIT_EVAL_SMOKE_${SLURM_JOB_ID}.json 2>/dev/null || true
echo SMOKE_DONE $S gate_rc=$rc evaluator_rc=$rce
