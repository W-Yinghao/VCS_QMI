#!/bin/bash
#SBATCH --job-name=r25_S0_sps_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/logs/%x_%j.out
set -euo pipefail
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/smoke/S0_sps_branchcheck; mkdir -p $O
cd /home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/sps_code
# technical smoke: all co-training branches from epoch 0 (clean_start=noisy_start=0), 1 epoch, from the archived warmup checkpoint; not a result
$PY run.py --data_path /projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data --vocab_path /projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab \
  --data_name f30k_precomp --noise_ratio 0.4 --noise_file /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy \
  --num_epochs 1 --clean_start_epoch 0 --noisy_start_epoch 0 --warmup_model_path /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/attempts/S0_sps_f40_s1001_1069504_indexdevice/warmup_model_4.pth.tar --seed 1001 --gpu 0 --output_dir $O --experiment_name S0_sps_branchcheck
echo SMOKE_DONE
