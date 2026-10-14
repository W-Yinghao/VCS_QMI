#!/usr/bin/env bash
#SBATCH --job-name=v7x_fetch
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# data_fetch.py run --group <group> on a CPU node (resumable), with the free-space guard watching the runner's own PID inside the job.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
/home/infres/yinwang/CS_QMI/env/bin/python -I scripts/data_fetch.py run --group "$1" & R=$!
bash scripts/data_fetch_guard.sh $R & G=$!
wait $R; rc=$?; kill $G 2>/dev/null; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
