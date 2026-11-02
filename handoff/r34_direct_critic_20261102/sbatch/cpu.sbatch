#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102/logs/%x_%j.out
# R34 CPU step: locks + configs, pytest, package check_math.py (no data except the two observed noise maps)
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
PKG=/home/infres/yinwang/CS_QMI/r30_r32_review/VCS_R30_R32_Independent_Review_20261102
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ)"; t0=$(date +%s)
cd $W
$PY -B r34/make_locks.py
$PY -B -m pytest -q -p no:cacheprovider tests/test_r34.py 2>&1 | tee $W/logs/PYTEST_${SLURM_JOB_ID}.txt
test ${PIPESTATUS[0]} -eq 0
$PY -I -B $PKG/reference/check_math.py --output $W/CHECK_MATH_${SLURM_JOB_ID}.json > /dev/null
$PY -B -c "import json,sys;d=json.load(open('$W/CHECK_MATH_${SLURM_JOB_ID}.json'));print('check_math',d['n_checks'],d['all_pass']);sys.exit(0 if d['all_pass'] else 1)"
echo "R34_CPU_PASS elapsed_s $(( $(date +%s)-t0 ))"
