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
#   x20-smoke-cpu             X20 extractor on 64 images of each role on CPU (float32 code-path check; the GPU run checks fp16)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v7x tests/v7x slurm/v7x | wc -l)"
echo "code: $(cat scripts/v7x/*.py tests/v7x/*.py slurm/v7x/*.sbatch 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the V7X scripts, tests and wrappers)"
what="$1"; shift
case "$what" in
  x10) $PY scripts/v7x/x10_legacy_inventory.py --p85 /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark --out "$1"; rc=$?;;
  tests) export V7X_REF_DIR=/home/infres/yinwang/CS_QMI/outputs/V7/package_ref_v2/reference OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
     $PY -m pytest -q -p no:cacheprovider tests/v7x; rc=$?
     echo "--- package reference tests (owner copy, read only; run from a scratch copy)"; T=$(mktemp -d /tmp/v7xref_XXXX); cp -r /home/infres/yinwang/CS_QMI/outputs/V7/package_ref_v2/reference /home/infres/yinwang/CS_QMI/outputs/V7/package_ref_v2/configs /home/infres/yinwang/CS_QMI/outputs/V7/package_ref_v2/schemas $T/ && chmod -R u+w $T
     (cd $T && PYTHONPATH=reference $E/bin/python -m pytest -q -p no:cacheprovider reference/tests 2>&1 | tail -3); rm -rf $T;;
  x20-smoke-cpu) export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
     $PY scripts/v7x/x20_features.py extract --out /projects/EEG-foundation-model/yinghao/outputs_v7/x20_smoke_cpu --roles FIT CAL_SELECT CAL_PROB DEV --limit 64 --workers 2; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
