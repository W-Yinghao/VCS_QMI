#!/usr/bin/env bash
#SBATCH --job-name=vl1_10_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-10 gate: tests, then a smoke of every Table-A route (96 FIT images, 64 CAL / DEV images, 60 updates) and the raw-CLIP row, timed.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
G=reports/VL1/gate_vl1_10_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0; S=$OUTPUT_ROOT/VL1_10_smoke; rm -rf $S; mkdir -p $S
"$PYTHON" -m pytest -q tests/test_vl1.py 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
for r in raw vcs js softmax siglip rff; do t0=$(date +%s); "$PYTHON" scripts/vl1_10_fit.py --route $r --ns 1000 --seeds 0 --lrs 5e-4 --out $S --smoke 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1; echo "[timing] $r $(( $(date +%s) - t0 )) s" | tee -a $G/timing.txt; done
"$PYTHON" scripts/vl1_10_fit.py --route vcs --ns 1000 --seeds 0 --lrs 5e-4 --scorer concat --out $S --smoke 2>&1 | tail -1
"$PYTHON" scripts/vl1_10_fit.py --route vcs --ns 1000 --seeds 0 --lrs 5e-4 --law phrase_uniform --out $S/phrase_uniform --smoke 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "{\"gate_rc\": $rc}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
