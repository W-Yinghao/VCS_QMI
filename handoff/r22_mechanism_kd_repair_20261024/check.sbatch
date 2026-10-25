#!/usr/bin/env bash
#SBATCH --job-name=r22_kd2_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/infres/yinwang/CS_QMI/env_kd/bin/python -B -s /home/infres/yinwang/CS_QMI/handoff/r22_mechanism_kd_repair_20261024/kd_terms.py check
/home/infres/yinwang/CS_QMI/env_kd/bin/python -B -s /home/infres/yinwang/CS_QMI/handoff/r22_mechanism_kd_repair_20261024/kd_terms.py cpu_path
/usr/bin/python3 /home/infres/yinwang/CS_QMI/handoff/r22_mechanism_kd_repair_20261024/admit.py
