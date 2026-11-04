#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
cmp $NR2_H/package/reference/proposal_reference.py $NR2_CODE/src/nr2/proposal_reference.py && echo "reference copy byte-identical"
cd $NR2_H/package && $NR1_PY -B -m pytest -q -p no:cacheprovider tests/test_reference.py
cd $NR2_CODE && $NR1_PY -B -m pytest -q -p no:cacheprovider tests/test_nr2_production.py
