#!/bin/bash
#SBATCH --job-name=r29_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r29_listwise_20261101/logs/%x_%j.out
# R29 real-data GPU smoke + gate (snapshot code_r29, frozen protocol): H/N/P/L/B (60%) and L (0%) from the common UNTRAINED
# init (2 limited warmup epochs x 30 it + 2 co-train epochs x 30 it, separate smoke dir), an R27 A1 reference run from the same
# init (r27_run.py a1_iface), then gate/r29_gate.py -> SMOKE/GPU_GATE.json. Submit with --qos/--partition on the command line.
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r29_listwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
cd $R29_CODE
S=$R29_OUT_BASE/smoke_r29_${SLURM_JOB_ID}
C="--data_path $DP --data_name f30k_precomp --seed 1001 --k 5 --log_step 10 --max_iters 30 --warmup_epoch 2 --num_epochs 2 --prefix_epoch -1"
for arm in h_native n_infonce p_pointwise_vcs l_listwise_vcs b_listwise_bce; do
  $PYTHON_BIN -u -B r29_run.py --r29_arm $arm $C --noise_file $R29_NOISE60 --noise_ratio 0.6 --output_dir $S/$arm
done
$PYTHON_BIN -u -B r29_run.py --r29_arm l_listwise_vcs $C --noise_file $R29_NOISE0 --noise_ratio 0.0 --output_dir $S/f00_l_listwise_vcs
$PYTHON_BIN -u -B r27_run.py --arm a1_iface --lam 0.0 $C --noise_file $R29_NOISE60 --noise_ratio 0.6 --output_dir $S/a1_ref
set +e
$PYTHON_BIN -u -B $R29_ROOT/gate/r29_gate.py $S $R29_NOISE60 0.6 $DP $R29_CODE
rc=$?
set -e
cp $S/GPU_GATE.json $R29_ROOT/GPU_GATE_${SLURM_JOB_ID}.json
cp $S/GPU_GATE.json $R29_ROOT/GPU_GATE.json
echo SMOKE_DONE $S gate_rc=$rc
