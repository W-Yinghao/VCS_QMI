#!/usr/bin/env bash
#SBATCH --job-name=p89_mechanism
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p89_mechanism_%j.out
# P89 mechanism records (prereg §Mechanism records): seeds 0-1 of the four S4 cells, epochs 100/400/800, 2048 selection images, own and standard augmentation.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P89 mechanism records"
RUNS=P35_vcs_a5_views4_800ep_seed0,P35_vcs_a5_views4_800ep_seed1,P43_vcs_a5_views4_800ep_augstrong_seed0,P89_vcs_a5_views4_800ep_augstrong_seed1,P41_simclr_views4_800ep_seed0,P41_simclr_views4_800ep_seed1,P89_simclr_views4_800ep_augstrong_seed0,P89_simclr_views4_800ep_augstrong_seed1
"$PYTHON" scripts/s4_mechanism_records.py --runs "$RUNS" --epochs 100,400,800 --n 2048 --aug own --out-dir reports/P89_mechanism; rc1=$?
"$PYTHON" scripts/s4_mechanism_records.py --runs "$RUNS" --epochs 100,400,800 --n 2048 --aug standard --out-dir reports/P89_mechanism; rc2=$?
"$PYTHON" scripts/s4_mechanism_records.py --table reports/P89_mechanism --out reports/P89_mechanism.md; rc3=$?
echo "rc=$rc1/$rc2/$rc3 end=$(date -u +%FT%TZ)"; [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
