#!/usr/bin/env bash
#SBATCH --job-name=r4_round1
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V6 server package r4, round 1 (read-only / repair): package reference tests, R4-01 regression test + corrected P151 aggregation,
# R4-E0 P134 audit (new output dir), R4-00 intake hashes of VL3 checkpoints and VL1-21 proposals.  No training, no raw file modified.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; rc=0; R=/home/infres/yinwang/CS_QMI/ssl_pilot
echo "== package r4 reference tests"; (cd $R/r4/package_r4_20261010 && "$PYTHON" -m unittest discover -s tests -v) 2>&1 | tail -45 || rc=1
echo "== R4-01 regression test"; (cd $R && "$PYTHON" -m unittest tests/test_p151_r4.py -v) 2>&1 | tail -12 || rc=1
echo "== R4-01 corrected P151 aggregation"; (cd $R && "$PYTHON" scripts/p151_e0_aggregate_r4.py --out reports/R4/P151_r4) || rc=1
echo "== R4-E0 P134 audit"; [ -e $R/reports/R4/P134_audit ] && echo "exists - skipped" || (cd $R/r4/package_r4_20261010/reference && "$PYTHON" audit_p134.py --input $R/reports/P134 --output $R/reports/R4/P134_audit) || rc=1
echo "== R4-00 hashes: see slurm/r4_hashes_cpu.sbatch (separate, re-runnable)"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
