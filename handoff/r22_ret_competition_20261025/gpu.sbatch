#!/usr/bin/env bash
#SBATCH --job-name=r22_retcomp_x
#SBATCH --qos=normal
#SBATCH --partition=L40S,H100,RTX6000PRO
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Competition-feature extraction for all three seeds, under the N2P-C stage c0 read guard (no DEV/TEST/eval_only/noise files).
set -u
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
H=/home/infres/yinwang/CS_QMI/handoff/r22_ret_competition_20261025
T=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c/v9_n2p/c_server/trace_open.py
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
rc=0
for s in 901 902 903; do
  $PY -I -B $T --stage c0 --out $H/traces/extract_s${s}_${SLURM_JOB_ID}.json -- $H/compete.py extract --seed $s || rc=$?
done
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
