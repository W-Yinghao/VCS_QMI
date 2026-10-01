#!/usr/bin/env bash
#SBATCH --job-name=p109_timing
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P109 cost measurement at full settings on one GPU (one unit of each module); outputs under outputs/P109_timing_<job>/ (not results).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
T=/home/infres/yinwang/CS_QMI/outputs/P109_timing_${SLURM_JOB_ID}; mkdir -p $T; rc=0
banner "P109 timing -> $T"
t0=$(date +%s); "$PYTHON" scripts/p109_x1.py --runs P35_vcs_a5_views4_800ep_seed0 --out $T/x1 || rc=1; echo "TIME x1_one_run $(( $(date +%s) - t0 ))"
t0=$(date +%s); "$PYTHON" scripts/p109_i1_features.py --runs P35_vcs_a5_views4_800ep_seed0 --out $T/i1_features || rc=1; echo "TIME i1_features_one_run $(( $(date +%s) - t0 ))"
t0=$(date +%s); "$PYTHON" scripts/p109_i1_audit.py --features $T/i1_features --run P35_vcs_a5_views4_800ep_seed0 --family colour --cells 'planted:0.1:2000' --repeats 10 --seed 998 --out $T/i1_audit_n2000 --effects || rc=1; echo "TIME i1_audit_10rep_n2000_plus_effects $(( $(date +%s) - t0 ))"
t0=$(date +%s); "$PYTHON" scripts/p109_i1_audit.py --features $T/i1_features --run P35_vcs_a5_views4_800ep_seed0 --family colour --cells 'planted:0.1:500' --repeats 10 --seed 997 --out $T/i1_audit_n500 || rc=1; echo "TIME i1_audit_10rep_n500 $(( $(date +%s) - t0 ))"
t0=$(date +%s); "$PYTHON" scripts/p109_i2_pilot.py --out $T/i2_one_repeat --cells vcs4v800:colour:0.1 --repeats 1 || rc=1; echo "TIME i2_one_cell_one_repeat $(( $(date +%s) - t0 ))"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
