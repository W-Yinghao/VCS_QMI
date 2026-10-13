#!/usr/bin/env bash
#SBATCH --job-name=rev8_diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV8 diagnostics (reports/REV8/REV8_DIAG_PROTOCOL_FROZEN_20261013.yaml).  Main env (torch 2.9 cu128): RTX6000PRO / H100 / L40S, or CPU for
# tests / smoke / summarize (submit with --partition=CPU --gres=gpu:0).  Positional arguments only.
#   tests | unit <host> <scale> <root> <out> <tag...> | smoke <n> <host> <scale> <root> <out> <tag...> | summarize <in> <out>
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; export PYTHONNOUSERSITE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/rev8_diag.py tests/test_rev8_diag.py slurm/rev8_diag.sbatch | wc -l)"
echo "code: $(cat scripts/rev8_diag.py tests/test_rev8_diag.py slurm/rev8_diag.sbatch | sha256sum | cut -c1-16) (sha256 of the REV8 script, tests and wrapper)"
what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_rev8_diag.py -v; rc=$?;;
  unit) host="$1"; scale="$2"; root="$3"; out="$4"; shift 4
        "$PYTHON" -I scripts/rev8_diag.py unit --host "$host" --scale "$scale" --root "$root" --out "$out" --tags "$@"; rc=$?;;
  smoke) n="$1"; host="$2"; scale="$3"; root="$4"; out="$5"; shift 5
        "$PYTHON" -I scripts/rev8_diag.py unit --host "$host" --scale "$scale" --root "$root" --out "$out" --smoke-n "$n" --tags "$@"; rc=$?;;
  summarize) "$PYTHON" -I scripts/rev8_diag.py summarize --in "$1" --protocol reports/REV8/REV8_DIAG_PROTOCOL_FROZEN_20261013.yaml --out "$2"; rc=$?;;
  *) echo "usage: tests | unit ... | smoke ... | summarize <in> <out>"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
