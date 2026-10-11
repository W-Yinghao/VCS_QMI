#!/usr/bin/env bash
#SBATCH --job-name=r4_reagg
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Code-review fixes: re-run the R4 tests, the corrected P151 r4 aggregation and the R4-VL0 analysis (all quantities with contrasts).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; rc=0; cd /home/infres/yinwang/CS_QMI/ssl_pilot
echo "== tests"; "$PYTHON" -m unittest tests/test_p151_r4.py tests/test_p151_r4_e2e.py tests/test_r4_vl1_laws.py tests/test_r4_rpc_identity.py -v 2>&1 | tail -8 || rc=1
echo "== P151 r4"; "$PYTHON" scripts/p151_e0_aggregate_r4.py --out reports/R4/P151_r4 || rc=1
echo "== R4-VL0 analysis"; "$PYTHON" scripts/r4_vl0_analyse.py || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
