#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr3_20261104/sbatch/common_env.sh
cmp $NR3_H/package/reference/ref_init.py $NR3_H/code/nr3/ref_init.py && echo "ref_init copy byte-identical"
cd $NR3_H/package && $NR1_PY -B -m pytest -v -p no:cacheprovider tests/test_reference.py
