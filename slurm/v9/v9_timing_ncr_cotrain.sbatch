#!/usr/bin/env bash
#SBATCH --job-name=v9_timing_ncr_cotrain
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-01 timing, NCR co-training epoch only: reload the warm-up checkpoint saved by job 1042033 (warm-up already timed there) and run ONE co-training
# epoch (GMM loss pass + two networks + dev validation of the average).  Requires the py3 iterator patch (v9/patches/ncr_py3_iterator.patch).
ROOT=/projects/EEG-foundation-model/yinghao; C=$ROOT/code_v9/2021-NeurIPS-NCR/NCR; DATA=$ROOT/data/ncr_precomp/NCR-data/data; VOCAB=$ROOT/data/ncr_precomp/NCR-data/vocab
WARM=${WARM:-$ROOT/outputs_v9/timing/group_1042033/ncr/warmup_model_0.pth.tar}; NOISE=$ROOT/code_v9/L2RM/noise_index/f30k_precomp_0.4.npy
O=$ROOT/outputs_v9/timing/ncr_cotrain_${SLURM_JOB_ID}; mkdir -p $O; export PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) ncr=$(git -C $C/.. rev-parse --short HEAD) patch=$(git -C $C/.. diff --stat | tail -1) warm=$WARM"
cd $C; s=$(date +%s)
$ROOT/envs/v9_sgr/bin/python run.py --gpu 0 --workers 2 --data_name f30k_precomp --data_path $DATA --vocab_path $VOCAB --noise_ratio 0.4 --noise_file $NOISE --warmup_model_path $WARM --num_epochs 1 --lr_update 30 --seed 917 --output_dir $O > $O/ncr_cotrain.log 2>&1; rc=$?
echo "[$(date -u +%FT%TZ)] NCR co-train 1 epoch (reloaded warm-up) rc=$rc seconds=$(( $(date +%s) - s ))"; grep -i -E "epoch|rsum|r1|Computinng losses\[1100|Validattion" $O/ncr_cotrain.log | tail -10 | cut -c1-160
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
