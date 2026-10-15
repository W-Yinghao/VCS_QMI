#!/usr/bin/env bash
#SBATCH --job-name=v9_n1
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N1 (reports/V9/V9_N1_PROTOCOL_*.yaml).  CPU only; outputs under outputs_v9/n1 (never the V8 directories).  Positional modes:
#   gate                      tests/v9 + --smoke of both runners + the aggregate on the smoke outputs (smoke outputs are not results)
#   exact <designs>           N1-Exact cells for the comma-separated designs (resumable: finished cell files are skipped)
#   image <seeds>             N1-Image cells for the comma-separated seeds (resumable)
#   aggregate                 tables + figures from the finished cells -> reports/V9/V9_N1_results/
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9 || exit 2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src
PY=/home/infres/yinwang/CS_QMI/env/bin/python; O=/projects/EEG-foundation-model/yinghao/outputs_v9/n1; mkdir -p "$O"
echo "host=$(hostname) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) code=$(cat src/v9/quant.py src/v9/reference_core.py scripts/v9/n1_exact.py scripts/v9/n1_image.py scripts/v9/n1_aggregate.py | sha256sum | cut -c1-16)"
what="$1"; shift; rc=0
case "$what" in
  gate) $PY -m pytest -q tests/v9 -p no:cacheprovider 2>&1 | tail -5; rc=${PIPESTATUS[0]}
        S=$O/smoke_$SLURM_JOB_ID; rm -rf "$S"
        /usr/bin/time -f "exact smoke wall %e s" $PY scripts/v9/n1_exact.py --out $S/exact --smoke || rc=1
        /usr/bin/time -f "exact n2048 timing (1 design, 1 rho, 2 reps) wall %e s" $PY scripts/v9/n1_exact.py --out $S/exact_n2048 --designs moderate --rhos 0.3 --sizes 2048 --repeats 2 || rc=1
        /usr/bin/time -f "image smoke wall %e s" $PY scripts/v9/n1_image.py --out $S/image --smoke || rc=1
        /usr/bin/time -f "image n2048 timing (1 seed, 1 rho, 3 reps, with Adam) wall %e s" $PY scripts/v9/n1_image.py --out $S/image_n2048 --seeds 901 --rhos 0.3 --sizes 2048 --repeats 3 || rc=1
        $PY scripts/v9/n1_aggregate.py --exact $S/exact --image $S/image --out $S/agg/n1 || rc=1
        echo "gate_rc=$rc (smoke outputs in $S; not results)";;
  exact) for d in ${1//,/ }; do $PY scripts/v9/n1_exact.py --out $O/exact --designs $d || rc=1; done;;
  image) $PY scripts/v9/n1_image.py --out $O/image --seeds "$1"; rc=$?;;
  aggregate) mkdir -p reports/V9/V9_N1_results; $PY scripts/v9/n1_aggregate.py --exact $O/exact --image $O/image --out reports/V9/V9_N1_results/n1_aggregate; rc=$?;;
  *) echo "usage: gate | exact <designs> | image <seeds> | aggregate"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
