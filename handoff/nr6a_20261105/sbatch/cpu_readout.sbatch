#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr6a_20261105/sbatch/common_env.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
$NR1_PY -B -m nr6a.readout_p1106
