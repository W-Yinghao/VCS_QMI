#!/usr/bin/env bash
#SBATCH --job-name=p116_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P116 CPU gate: P116 tests (incl. exact reproduction of a historical P105 smoke instance), the P105 / P109 / P102 suites that import the edited
# fitters, and a CPU timing smoke of the runner (3 pilot cells, 2 repeats, full budgets, 200 permutations).  Outputs -> reports/P116_GATE_<job>/.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
J=${SLURM_JOB_ID:-local}; G=$REPO_ROOT/reports/P116_GATE_$J; mkdir -p $G; rc=0
banner "P116 CPU gate -> $G"
"$PYTHON" -m pytest -q tests/test_p116.py > $G/pytest_test_p116.txt 2>&1 || rc=1; tail -2 $G/pytest_test_p116.txt
for t in tests/test_p109.py tests/test_p102.py; do [ -f $t ] && { "$PYTHON" -m pytest -q $t > $G/pytest_$(basename $t .py).txt 2>&1 || rc=1; tail -1 $G/pytest_$(basename $t .py).txt; }; done
F=/home/infres/yinwang/CS_QMI/outputs/P45_precheck_D
for spec in "${F}_vcs4v800|colour|colour_s0.05:1000|61" "${F}_simclr|colour|colour_s0.2:2000|62" "${F}_simclr_blur|blur|blur_s0.5:1000|63"; do
  IFS='|' read -r feat fam cells seed <<< "$spec"
  "$PYTHON" scripts/p116_solve.py --features $feat --family $fam --cells "$cells" --repeats 2 --perms 200 --seed $seed --out $G/timing_${fam}_${seed} || rc=1
done
echo "gate_rc=$rc"; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
