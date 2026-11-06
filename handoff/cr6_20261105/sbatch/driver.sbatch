#!/usr/bin/env bash
# CR6 CPU driver: waits for acceptance -> freeze -> submits groups (retrying the per-user submit limit, no --dependency) -> graphs -> A2 -> summarize.
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr6_20261105/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
$CR6_PY -I -B cr6_driver.py "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
