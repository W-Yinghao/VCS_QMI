#!/usr/bin/env bash
#SBATCH --job-name=p124_gpu_smoke
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P124 GPU smoke: the full-size readout on ONE encoder (cost per encoder; anchor = stored training linear accuracy for fine recipe_raw).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P124_GATE_SMOKE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
banner "P124 GPU smoke -> $G"
"$PYTHON" scripts/p124_gran.py --runs ${RUN:-P107_AP3_c100_views4_800ep_seed0} --out $G 2>&1 | tee $G/runner.txt | tail -8; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p124_aggregate.py --in $G --out $G/aggregate 2>&1 | tail -2
echo "{\"smoke_rc\": $rc}" > $G/summary.json
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
