#!/bin/bash
#SBATCH --job-name=cdr1_s0
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/logs/s0_%j.out
set -euo pipefail
PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
cd /home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/s0/pkg
export OMP_NUM_THREADS=4
echo "host $(hostname) start $(date -u +%FT%TZ)"
$PY -I -B -m pytest -o addopts='' -q -p no:cacheprovider 2>&1 | tee ../pytest_s0.txt
$PY -I -B tools/build_registry.py > ../registry_s0.json
$PY -I -B tools/run_demo.py --out ../demo > ../demo_stdout.txt
$PY -I -B tools/run_cli_smoke.py > ../cli_smoke_stdout.txt
$PY -I -B tools/run_registry_smoke.py > ../registry_smoke_stdout.txt
$PY -I -B tools/run_stress_simulations.py --out ../stress > ../stress_stdout.txt
# configs regenerated must equal shipped configs
for f in CORE_FITS.csv DATA_CASES.csv PREDICTION_VIEWS.csv RHO_PROFILES.csv REGISTRY.json; do cmp configs/$f ../../package/configs/$f && echo "CFG_SAME $f"; done
echo "S0_DONE $(date -u +%FT%TZ)"
