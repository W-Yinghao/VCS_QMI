#!/bin/bash
#SBATCH --job-name=r28_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r28_selective_20261031/logs/%x_%j.out
# R28 real-data GPU smoke + gate (snapshot code_r28, frozen config): g1/g2/r1/r2 from the REAL f40_s1001 common prefix,
# 1 co-train epoch x 30 iterations (separate smoke dir, lam 0), then gate/r28_gate.py from the R27-S smoke state
# -> SMOKE/GPU_GATE.json; then a 3-batch run of the read-only diagnostics tool (tool check only, not part of all_pass).
# Submit with --qos/--partition on the command line (normal RTX6000PRO,L40S,H100 or runfill A100,A40).
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r28_selective_20261031/server.env
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
cd $R28_CODE
S=$R28_OUT_BASE/smoke_r28_${SLURM_JOB_ID}
C="--data_path $DP --noise_file $R28_NOISE40 --noise_ratio 0.4 --seed 1001 --k 5 --log_step 10 --max_iters 30"
for arm in g1_cos_sel g2_vcs_sel r1_res_vcs_sel r2_res_bce_sel; do
  $PYTHON_BIN -u -B r27_run.py --arm $arm --lam 0.0 $C --output_dir $S/$arm --resume $R28_PREFIX40 --num_epochs 2
done
set +e
$PYTHON_BIN -u -B $R28_ROOT/gate/r28_gate.py $S $R28_NOISE40 $DP $R28_GATE_STATE $R28_CODE
rc=$?
cp $S/GPU_GATE.json $R28_ROOT/GPU_GATE_${SLURM_JOB_ID}.json
cp $S/GPU_GATE.json $R28_ROOT/GPU_GATE.json
$PYTHON_BIN -u -B $R28_ROOT/tools/r28_fit_eval.py --state $R28_GATE_STATE --noise-file $R28_NOISE40 --data-path $DP \
  --code-dir $R28_CODE --out $S/FIT_EVAL_SMOKE.json --n-batches 3
rce=$?
set -e
cp $S/FIT_EVAL_SMOKE.json $R28_ROOT/FIT_EVAL_SMOKE_${SLURM_JOB_ID}.json 2>/dev/null || true
echo SMOKE_DONE $S gate_rc=$rc evaluator_rc=$rce
