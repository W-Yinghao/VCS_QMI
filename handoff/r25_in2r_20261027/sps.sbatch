#!/bin/bash
#SBATCH --job-name=r25_S0_sps_f40_s1001
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/logs/%x_%j.out
set -euo pipefail
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/runs/S0_sps_f40_s1001
mkdir -p $O
nvidia-smi --query-gpu=name,driver_version --format=csv | tee $O/GPU.txt
cd /home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/sps_code
$PY run.py --data_path /projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data --vocab_path /projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab \
  --data_name f30k_precomp --noise_ratio 0.4 --noise_file /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy \
  --num_epochs 40 --seed 1001 --gpu 0 --output_dir $O --experiment_name S0_sps_f40_s1001
$PY /home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/tools/sps_fulldev_eval.py $O
echo S0_DONE
