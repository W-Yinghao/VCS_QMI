#!/usr/bin/env bash
#SBATCH --job-name=p149a1_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P149 add. 1 aggregation: z-site seed 1, h-site seed 2, z-site seed 2 (same frozen P149 aggregator).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
D=$OUTPUT_ROOT/P149a1_twoview; rc=0
S1=P107_AP3_views4_800ep_seed1,P41_simclr_views4_800ep_seed1,P107_AP3_c100_views4_800ep_seed1,P91_c100_simclr_views4_800ep_seed1
S2=P107_AP3_views4_800ep_seed2,P41_simclr_views4_800ep_seed2,P107_AP3_c100_views4_800ep_seed2,P91_c100_simclr_views4_800ep_seed2
echo "### z-site, seed 1"; "$PYTHON" scripts/p149_aggregate.py --dir $D --runs $S1 --suffix _z --out reports/P149A1_results_z_seed1.json || rc=1
echo "### h-site, seed 2"; "$PYTHON" scripts/p149_aggregate.py --dir $D --runs $S2 --out reports/P149A1_results_h_seed2.json || rc=1
echo "### z-site, seed 2"; "$PYTHON" scripts/p149_aggregate.py --dir $D --runs $S2 --suffix _z --out reports/P149A1_results_z_seed2.json || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
