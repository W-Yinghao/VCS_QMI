#!/usr/bin/env bash
#SBATCH --job-name=p116_gpu_timing
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P116: GPU timing of the runner on one pilot cell (2 repeats) to decide CPU vs GPU for the pilot / full run.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
J=${SLURM_JOB_ID:-local}; G=$REPO_ROOT/reports/P116_GATE_gpu_$J; mkdir -p $G
"$PYTHON" scripts/p116_solve.py --features /home/infres/yinwang/CS_QMI/outputs/P45_precheck_D_simclr --family colour --cells "colour_s0.2:2000" --repeats 2 --perms 200 --seed 62 --out $G/timing_colour_62; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
