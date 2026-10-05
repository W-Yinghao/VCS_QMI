#!/usr/bin/env bash
#SBATCH --job-name=p132_cpu_check
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p132_cpu_check_%j.out
# P132 ImageNet preflight, CPU part: (0) unit tests (P132 distributed pairing incl. 2-process gloo; P130 backbone / config policy);
# (1) data check (manifest vs filesystem, class mapping, stat of all 1.33 M files, full decode of 50 train images / class + all 50 k val);
# (2) data-loader throughput (2 / 4 views, blur 0.5 and 0, workers 4 / 8 / 16 / 32, batch 64).  No GPU, no training.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
R=$REPO_ROOT/reports/P132; mkdir -p "$R"; rc=0
banner "P132 CPU check -> $R"
"$PYTHON" -m pytest -q tests/test_p132_imagenet.py tests/test_vcs_ssl_p130.py 2>&1 | tail -5 | tee "$R/tests_${SLURM_JOB_ID}.txt" || rc=1
"$PYTHON" scripts/p132_imagenet_preflight.py datacheck --out "$R/datacheck.json" --workers 32 2>&1 | tail -40 || rc=1
"$PYTHON" scripts/p132_imagenet_preflight.py loader --out "$R/loader.json" --workers-list 4 8 16 32 2>&1 | tail -20 || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
