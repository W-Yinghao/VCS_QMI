#!/usr/bin/env bash
#SBATCH --job-name=r6reg2_smoke
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P2 CPU smoke in one allocation (one queue slot): unit tests (P2 + REG-P1), then perturb and every row's registration on the REG-P1
# CPU-smoke artefacts (2 CAL / 2 DEV images, FIT 4; critic and CoMIR runs of outputs/R6_REG_cpu_smoke2/smoke), rows in parallel.  Code check and
# CPU timings only; kernel and critic register the first 2 perturbations per image, CoMIR warp-then-encode the first 1 (DEV only).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
O=/home/infres/yinwang/CS_QMI/outputs; P1=$O/R6_REG_cpu_smoke2; P2=$O/R6_REG_P2_cpu_smoke; L=/home/infres/yinwang/CS_QMI/slurm_logs/r6reg2_smoke_${SLURM_JOB_ID}
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -u +%FT%TZ)"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "$PYTHON" -m unittest tests/test_r6_reg_p2.py tests/test_r6_reg_pilot.py -v 2>&1 | tail -32; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "exit=1 (tests) end=$(date -u +%FT%TZ)"; exit 1; }
"$PYTHON" scripts/r6_reg_p2.py perturb --p1-root "$P1" --root "$P2" --smoke || { echo "exit=1 (perturb)"; exit 1; }
run() { local tag=$1 n=$2; shift 2; OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n "$PYTHON" scripts/r6_reg_p2.py register --p1-root "$P1" --root "$P2" --smoke "$@" > "$L.$tag.log" 2>&1
  echo "[$tag] exit=$? $(grep -E '^\[register-p2|Traceback|Error' "$L.$tag.log" | tail -2 | tr '\n' ' ')"; }
run mi 8 --method mi & run nmi 8 --method nmi & run comir 4 --method comir --seed 0 & run comir_wte 24 --method comir_wte --seed 0 --n-pert 1 &
run kernel 8 --method kernel --seed 0 --n-pert 2 & run vcs 8 --method vcs --seed 0 --n-pert 2 &
wait; rc=0; for t in mi nmi comir comir_wte kernel vcs; do grep -q '^\[register-p2' "$L.$t.log" || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
