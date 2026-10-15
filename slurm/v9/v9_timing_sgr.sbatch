#!/usr/bin/env bash
#SBATCH --job-name=v9_timing_sgr
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-01 budget measurement: ONE epoch of the native SGR single model (SGRAF python3.6 branch, unmodified train.py) on Flickr30K precomp with
# the host's own defaults (batch 128, Adam 2e-4, margin 0.2, max-violation).  Output under outputs_v9/timing only.  No V8 path is touched.
E=/projects/EEG-foundation-model/yinghao/envs/v9_sgr; C=/projects/EEG-foundation-model/yinghao/code_v9/SGRAF_py3
DATA=${DATA:?set DATA to the directory that contains f30k_precomp/}; O=/projects/EEG-foundation-model/yinghao/outputs_v9/timing/sgr_f30k_1ep_${SLURM_JOB_ID}; mkdir -p $O
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) sgraf_py3=$(git -C $C rev-parse --short HEAD) data=$DATA"
export PYTHONNOUSERSITE=1; cd $C
/usr/bin/time -v $E/bin/python train.py --data_path "$DATA" --data_name f30k_precomp --vocab_path $C/vocab/ --model_name $O/checkpoint --logger_name $O/log \
  --num_epochs 1 --lr_update 30 --module_name SGR --batch_size 128 --workers 4 2>&1 | grep -v "^$" | tail -60; rc=${PIPESTATUS[0]}
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
