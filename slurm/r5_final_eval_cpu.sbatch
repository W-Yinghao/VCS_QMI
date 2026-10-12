#!/usr/bin/env bash
#SBATCH --job-name=r5fe_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r5 final evaluation, CPU stages of scripts/r5_final_eval.py: plan | build | manifest | score | make-dev-refer | integrity | rows-yaml, or the unit tests
# (tests).  First argument = stage, the rest is passed through (e.g. --root <root>).  VL environment, offline, python -I for the stages.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 2; what="${1:-}"; shift
echo "stage=$what host=$(hostname) job=${SLURM_JOB_ID:-NA} commit=$(git rev-parse --short HEAD 2>/dev/null) dirty=$(git status --porcelain -- scripts/r5_final_eval.py slurm/r5_final_eval_cpu.sbatch slurm/r5_final_eval_gpu.sbatch 2>/dev/null | wc -l) start=$(date -u +%FT%TZ)"
case "$what" in
  tests) "$E/bin/python" -m unittest tests/test_r5_final_eval.py -v; rc=$?;;
  plan|build|manifest|score|make-dev-refer|integrity|rows-yaml) "$E/bin/python" -I scripts/r5_final_eval.py "$what" "$@"; rc=$?;;
  *) echo "usage: sbatch slurm/r5_final_eval_cpu.sbatch plan|build|manifest|score|make-dev-refer|integrity|rows-yaml|tests [args]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
