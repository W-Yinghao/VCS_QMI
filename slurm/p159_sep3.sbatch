#!/usr/bin/env bash
#SBATCH --job-name=p159_sep3
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P159 (SEP-3; reports/P159_SEP3_NATURAL_PREREG_*_20261015.md).  Positional: run <seeds "300+301+302+303"> | gate.  Resumable (finished cells skipped).
# gate = tests/test_p159.py + tests/test_p157.py, then the --smoke runner (one seed, every cell, 40 updates) and a timed full-size cell per source.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; cd /home/infres/yinwang/CS_QMI/ssl_pilot; O=/home/infres/yinwang/CS_QMI/outputs/P159_sep3
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat src/vcs_estim/sep.py src/vcs_estim/p159.py scripts/p159_sep3.py scripts/p159_aggregate.py | sha256sum | cut -c1-16)"
case "$1" in
  run) "$PYTHON" scripts/p159_sep3.py run --seeds "$2" --out $O; rc=$?;;
  gate) G=reports/P159_GATE; mkdir -p $G
        "$PYTHON" -m pytest -q tests/test_p159.py tests/test_p157.py -p no:cacheprovider 2>&1 | tee $G/tests_${SLURM_JOB_ID}.log | tail -5; rc1=${PIPESTATUS[0]}
        "$PYTHON" scripts/p159_sep3.py run --seeds 300 --out /home/infres/yinwang/CS_QMI/outputs/P159_smoke --smoke 2>&1 | tee $G/smoke_${SLURM_JOB_ID}.log | tail -4; rc2=${PIPESTATUS[0]}
        "$PYTHON" scripts/p159_aggregate.py --dir /home/infres/yinwang/CS_QMI/outputs/P159_smoke --out $G/smoke_aggregate --seeds 1 2>&1 | tail -3; rc3=${PIPESTATUS[0]}
        "$PYTHON" scripts/p159_sep3.py run --seeds 300 --out /home/infres/yinwang/CS_QMI/outputs/P159_timing --sources coco_clip --limit 1 2>&1 | tee $G/timing_${SLURM_JOB_ID}.log | tail -2; rc4=${PIPESTATUS[0]}
        "$PYTHON" scripts/p159_sep3.py run --seeds 300 --out /home/infres/yinwang/CS_QMI/outputs/P159_timing --sources cifar_ap3 --limit 1 2>&1 | tee -a $G/timing_${SLURM_JOB_ID}.log | tail -2; rc5=${PIPESTATUS[0]}
        echo "gate: tests_rc=$rc1 smoke_rc=$rc2 smoke_agg_rc=$rc3 timing_rc=$rc4/$rc5"; rc=$((rc1 | rc2 | rc3 | rc4 | rc5));;
  *) echo "usage: run <seeds> | gate"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
