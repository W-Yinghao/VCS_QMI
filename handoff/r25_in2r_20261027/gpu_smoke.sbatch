#!/bin/bash
#SBATCH --job-name=r25_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/logs/%x_%j.out
set -euo pipefail
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r25_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r25_$SLURM_JOB_ID' EXIT
t0=$(date +%s); for f in train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done; echo staged in $(( $(date +%s)-t0 ))s
cd /home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/code
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/smoke_${SLURM_JOB_ID}
N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy
C="--data_path $TMPDIR/r25_$SLURM_JOB_ID --noise_file $N --noise_ratio 0.4 --seed 1001 --log_step 10 --max_iters 30"
nvidia-smi --query-gpu=name,driver_version --format=csv
$PY r25_run.py $C --output_dir $S/cont --warmup_epoch 1 --num_epochs 2
$PY r25_run.py $C --output_dir $S/resume --resume $S/cont/prefix_state.pt --num_epochs 2
$PY r25_calibrate.py $C --output_dir $S/calib --resume $S/cont/prefix_state.pt
read A B CV0 CB0 CV1 CB1 < <($PY -c "
import json;d=json.load(open('$S/calib/CALIBRATION.json'));c=d['c_D']
print(d['a'],d['b'],c.get('vcs_c0'),c.get('bce_c0'),c.get('vcs_c1'),c.get('bce_c1'))")
$PY r25_run.py $C --output_dir $S/vcs_c0 --resume $S/cont/prefix_state.pt --num_epochs 2 --noisy_obj vcs --link c0 --c_D $CV0
$PY r25_run.py $C --output_dir $S/bce_c0 --resume $S/cont/prefix_state.pt --num_epochs 2 --noisy_obj bce --link c0 --c_D $CB0
if [ "$A" != "None" ]; then
$PY r25_run.py $C --output_dir $S/vcs_c1 --resume $S/cont/prefix_state.pt --num_epochs 2 --noisy_obj vcs --link c1 --c1_a $A --c1_b $B --c_D $CV1
$PY r25_run.py $C --output_dir $S/bce_c1 --resume $S/cont/prefix_state.pt --num_epochs 2 --noisy_obj bce --link c1 --c1_a $A --c1_b $B --c_D $CB1
fi
$PY r25_gate.py $S $N
echo SMOKE_DONE
