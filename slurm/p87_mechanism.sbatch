#!/usr/bin/env bash
#SBATCH --job-name=p87_mechanism
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p87_mechanism_%j.out
# P87 mechanism records (prereg §Mechanism records): seeds 0-1 of the four S4 cells, epochs 100/400/800, 2048 selection images, own and standard augmentation.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P87 mechanism records"
RUNS=P35_vcs_a5_views4_800ep_seed0,P35_vcs_a5_views4_800ep_seed1,P87_skernel_m4096_bw0.5_views4_800ep_seed0,P87_skernel_m4096_bw0.5_views4_800ep_seed1,P87_kcs_bw0.5_views4_800ep_seed0,P87_kcs_bw0.5_views4_800ep_seed1
"$PYTHON" scripts/s4_mechanism_records.py --runs "$RUNS" --epochs 100,400,800 --n 2048 --aug own --out-dir reports/P87_mechanism; rc1=$?
"$PYTHON" scripts/s4_mechanism_records.py --table reports/P87_mechanism --out reports/P87_mechanism.md; rc3=$?
echo "rc=$rc1/$rc2/$rc3 end=$(date -u +%FT%TZ)"; [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
