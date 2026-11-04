#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr4_20261104/sbatch/common_env.sh
cmp $NR4_H/package/reference/adapters.py $NR4_H/code/nr4/ref_adapters.py && cmp $NR4_H/package/reference/gradient_replay.py $NR4_H/code/nr4/ref_gradient_replay.py && echo "reference copies byte-identical"
cd $NR4_H/package && $NR1_PY -B -m unittest discover -s tests -v
