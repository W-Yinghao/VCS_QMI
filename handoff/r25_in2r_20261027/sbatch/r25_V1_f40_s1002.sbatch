#!/bin/bash
#SBATCH --job-name=r25_V1_f40_s1002
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/logs/%x_%j.out
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version --format=csv
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r25_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r25_$SLURM_JOB_ID' EXIT
for f in train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
echo staged $(du -sh $LOC)
cd /home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/code
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u r25_run.py --data_path $TMPDIR/r25_$SLURM_JOB_ID --noise_file /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy --noise_ratio 0.4 --seed 1002 --output_dir /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1002/V1 --resume /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1002/prefix/prefix_state.pt --noisy_obj vcs --link c1 --c_D 19.19730554681986 --c1_a 10.76670458491166 --c1_b 0.08897677104687318
echo r25_V1_f40_s1002_DONE
