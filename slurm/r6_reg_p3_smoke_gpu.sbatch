#!/usr/bin/env bash
#SBATCH --job-name=r6reg3_smoke_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P3 GPU smoke (one GPU): prepare / perturb on the smoke subset (no DEV window read) — or a copy of an already prepared smoke subset
# (P3_SMOKE_DATA=<root>, a path; saves the CPU-bound prepare on the GPU) — smoke training of the three critics (200 steps) and one recipe epoch
# of CoMIR (32 × 32; × 23 = the recipe), every registration row of the protocol with 2 perturbations per window (CoMIR warp-then-encode is not run
# in REG-P3), the readings.  Dataset: P3_DATASET (default epfl).  Timings for the budget; code check only (smoke numbers are not results).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
D=${P3_DATASET:-epfl}; G=${P3_SMOKE_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3_gpu_smoke_$D}
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) dataset=$D root=$G start=$(date -u +%FT%TZ)"
step() { local t0; t0=$(date +%s); "$PYTHON" scripts/r6_reg_p3.py "$@" --smoke --root "$G" || { echo "exit=1 ($*) end=$(date -u +%FT%TZ)"; exit 1; }; echo "[timing] $* : $(( $(date +%s) - t0 )) s"; }
if [ -n "${P3_SMOKE_DATA:-}" ]; then mkdir -p "$G/smoke" && cp -r "$P3_SMOKE_DATA/smoke/cache" "$P3_SMOKE_DATA/smoke/manifest_smoke.json" "$P3_SMOKE_DATA/smoke/perturbations_smoke.json" "$G/smoke/" || { echo "exit=1 (copy)"; exit 1; }; echo "[prepare] copied from $P3_SMOKE_DATA"
else step prepare --dataset "$D"; step perturb; fi
step train --objective vcs,logistic,infonce --seed 0; step comir --seed 0 --comir-epochs 1
for m in mi nmi kernel vcs logistic infonce comir; do step register --method $m --seed 0 --n-pert 2; done
"$PYTHON" scripts/r6_reg_p3_aggregate.py --smoke --root "$G" --out "$G/smoke/reports" | tail -24; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
