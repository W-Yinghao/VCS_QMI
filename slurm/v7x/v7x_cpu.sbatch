#!/usr/bin/env bash
#SBATCH --job-name=v7x_cpu
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7X CPU jobs (reports/V7/V7X_*): positional arguments only.
#   x10 <out.json>            legacy controlled-data inventory recompute (raw P85 JSONs, read only)
#   tests                     pytest tests/v7x (+ the v2 package reference tests on our copies)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v7x tests/v7x slurm/v7x | wc -l)"
echo "code: $(cat scripts/v7x/*.py tests/v7x/*.py slurm/v7x/*.sbatch 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the V7X scripts, tests and wrappers)"
what="$1"; shift
case "$what" in
  x10) $PY scripts/v7x/x10_legacy_inventory.py --p85 /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark --out "$1"; rc=$?;;
  tests) $PY -m pytest -q -p no:cacheprovider tests/v7x; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
