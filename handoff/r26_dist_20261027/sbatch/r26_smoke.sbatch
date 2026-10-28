#!/bin/bash
#SBATCH --job-name=r26_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/logs/%x_%j.out
set -euo pipefail
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r26_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r26_$SLURM_JOB_ID' EXIT
for f in train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
cd /home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/code
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/smoke_${SLURM_JOB_ID}
N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy
C="--data_path $TMPDIR/r26_$SLURM_JOB_ID --noise_file $N --noise_ratio 0.4 --seed 1001 --log_step 10 --max_iters 30"
nvidia-smi --query-gpu=name,driver_version --format=csv
$PY r25_run.py $C --output_dir $S/prefix --warmup_epoch 1 --num_epochs 1 --stop_after_prefix
$PY r26_calibrate.py $S/CALIBRATION.json $C --resume $S/prefix/prefix_state.pt || echo CALIBRATE_FAILED
for arm in d_vcs d_bce d_snce m_vcs n_vcs; do
  $PY r26_run.py --arm $arm --c_J 1.0 $C --output_dir $S/$arm --resume $S/prefix/prefix_state.pt --num_epochs 2
done
$PY r26_gate.py $S $N $TMPDIR/r26_$SLURM_JOB_ID
echo SMOKE_DONE $S
