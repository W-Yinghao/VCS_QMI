#!/usr/bin/env bash
#SBATCH --job-name=p118_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P118 CPU gate: tests (P118 + P102 / P109 nested dependencies), truth and visual smokes, full-size timing (truth: weak, 1 repeat; visual: 1 run, 1 cell, 1 repeat).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P118_GATE; mkdir -p $G; rc=0; banner "P118 gate"; export CUDA_VISIBLE_DEVICES=""
"$PYTHON" -m pytest -q tests/test_p118.py tests/test_p102.py tests/test_p109.py > $G/pytest.txt 2>&1 || rc=1; tail -2 $G/pytest.txt
"$PYTHON" scripts/p118_truth.py --smoke --out $G/truth_smoke > $G/truth_smoke.txt 2>&1 || rc=1; tail -2 $G/truth_smoke.txt | cut -c1-300
"$PYTHON" scripts/p118_visual.py --smoke --runs P41_simclr_views4_800ep_seed1 --cells blur_s0.25,null --out $G/visual_smoke > $G/visual_smoke.txt 2>&1 || rc=1; tail -2 $G/visual_smoke.txt | cut -c1-300
/usr/bin/env bash -c "time '$PYTHON' scripts/p118_truth.py --conds weak --repeats 1 --out $G/truth_timing" > $G/truth_timing.txt 2>&1 || rc=1; tail -4 $G/truth_timing.txt | cut -c1-300
/usr/bin/env bash -c "time '$PYTHON' scripts/p118_visual.py --runs P107_AP3_views4_800ep_seed2 --cells colour_s0.1 --repeats 1 --out $G/visual_timing" > $G/visual_timing.txt 2>&1 || rc=1; tail -4 $G/visual_timing.txt | cut -c1-300
echo "gate_rc=$rc exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
