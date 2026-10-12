#!/usr/bin/env bash
#SBATCH --job-name=r6reg3_smoke_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P3 CPU smoke in one allocation (one queue slot; unit tests run separately: slurm/r6_reg_p3_tests_cpu.sbatch): prepare / perturb on the smoke subset (FIT = the first 4 FIT
# windows, CAL = 2 CAL windows, "DEV" = 2 further CAL windows relabelled: no DEV window is read before the freeze) — or a copy of an already
# prepared smoke subset (P3_SMOKE_DATA=<root>, a path) — smoke training of the three critics (200 steps) and CoMIR (2 × 4 × 8), every registration
# row of the protocol in parallel (MI / NMI / CoMIR 2 perturbations per window, kernel / critics 1; CoMIR warp-then-encode is not run in REG-P3),
# then the readings into the smoke root.  Dataset: P3_DATASET (default epfl).  Code check and CPU timings only (smoke numbers are not results).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
D=${P3_DATASET:-epfl}; S=${P3_SMOKE_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3_cpu_smoke_$D}; L=/home/infres/yinwang/CS_QMI/slurm_logs/r6reg3_smoke_cpu_${SLURM_JOB_ID}
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} dataset=$D root=$S start=$(date -u +%FT%TZ)"
if [ -n "${P3_SMOKE_DATA:-}" ]; then mkdir -p "$S/smoke" && cp -r "$P3_SMOKE_DATA/smoke/cache" "$P3_SMOKE_DATA/smoke/manifest_smoke.json" "$P3_SMOKE_DATA/smoke/perturbations_smoke.json" "$S/smoke/" || { echo "exit=1 (copy)"; exit 1; }; echo "[prepare] copied from $P3_SMOKE_DATA"
else T0=$(date +%s); OMP_NUM_THREADS=8 "$PYTHON" scripts/r6_reg_p3.py prepare --dataset "$D" --smoke --root "$S" || { echo "exit=1 (prepare)"; exit 1; }; echo "[prepare] $(( $(date +%s) - T0 )) s"
  "$PYTHON" scripts/r6_reg_p3.py perturb --smoke --root "$S" || { echo "exit=1 (perturb)"; exit 1; }; fi
job() { local tag=$1 n=$2; shift 2; local t0; t0=$(date +%s); OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n "$PYTHON" scripts/r6_reg_p3.py "$@" --smoke --root "$S" > "$L.$tag.log" 2>&1
  local rc=$?; echo "[$tag] exit=$rc $(( $(date +%s) - t0 )) s $(grep -E '^\[register-p3|^\[train|^\[comir|Traceback|Error' "$L.$tag.log" | tail -1 | cut -c1-260)"; return $rc; }
job train_vcs 16 train --objective vcs & job train_logistic 16 train --objective logistic & job train_infonce 16 train --objective infonce & job comir 16 comir --seed 0 & wait
for t in train_vcs train_logistic train_infonce comir; do grep -q Traceback "$L.$t.log" && { echo "exit=1 ($t) end=$(date -u +%FT%TZ)"; exit 1; }; done
job mi 8 register --method mi --n-pert 2 & job nmi 8 register --method nmi --n-pert 2 & job comir_reg 8 register --method comir --n-pert 2 &
job kernel 10 register --method kernel --n-pert 1 & job vcs 10 register --method vcs --n-pert 1 & job logistic 10 register --method logistic --n-pert 1 &
job infonce 10 register --method infonce --n-pert 1 & wait
rc=0; for t in mi nmi comir_reg kernel vcs logistic infonce; do grep -q '^\[register-p3' "$L.$t.log" || { rc=1; echo "MISSING $t"; }; done
"$PYTHON" scripts/r6_reg_p3_aggregate.py --smoke --root "$S" --out "$S/smoke/reports" > "$L.aggregate.log" 2>&1 || rc=1; tail -24 "$L.aggregate.log"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
