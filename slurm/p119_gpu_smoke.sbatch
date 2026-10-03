#!/usr/bin/env bash
#SBATCH --job-name=p119_gpu_smoke
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P119 GPU smoke: (a) the float16-vs-float32 check on all 15 encoders (small subsets, same GPU device class as the cache);
# (b) transfer + c10extra readouts at smoke size on one encoder (timing / code path).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P119_GATE_SMOKE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
banner "P119 GPU smoke -> $G"; nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" scripts/p119_v.py --modules fpcheck --out $G 2>&1 | tee $G/fpcheck.txt | grep -E "fp-check|FAIL|done" ; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p119_v.py --runs P41_simclr_views4_800ep_seed0 --modules transfer,c10extra --out $G/smoke --smoke 2>&1 | tee $G/smoke_runner.txt | tail -4; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p119_aggregate.py --in $G --out $G/fpcheck_summary || rc=1
echo "{\"gate_rc\": $rc}" > $G/summary.json
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
