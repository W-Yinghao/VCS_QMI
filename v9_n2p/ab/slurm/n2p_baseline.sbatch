#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9 N2P phases B0 / B1: NCR_A or L2RM_A through the patched official code, DEV-only (v9_n2p/ab/b0/n2p_baseline.py).
#   sbatch [partition/qos/time/nice/exclude/job-name] n2p_baseline.sbatch <gate|full> <ncr|l2rm> <seed>
# gate = B0 validation + re-run of the health check (seed 917, warm-up 5 + 3 epochs); full = B1 unit (NCR 5+40, L2RM 5+35).
# N2P_OUT (exported by the submitter) is the confirmed output root; nothing is written elsewhere except this log.
# Resubmitting the same line resumes: finished steps are skipped; an unfinished training attempt is moved to attempts/ and restarted.
set -o pipefail
SCHED=${1:?gate|full}; B=${2:?ncr|l2rm}; SEED=${3:?seed}
: "${N2P_OUT:?export N2P_OUT=<confirmed output root>}"
case $B in ncr) M=NCR_A;; l2rm) M=L2RM_A;; *) echo "baseline must be ncr or l2rm"; exit 2;; esac
if [ "$SCHED" = gate ]; then U=$N2P_OUT/B0GATE_${M}_s${SEED}_r04; else U=$N2P_OUT/V9N2P_f30k_r04_s${SEED}_${M}; fi
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_ab
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C $R rev-parse --short HEAD) unit=$U"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/home/infres/yinwang/CS_QMI/env/bin/python -I $R/v9_n2p/ab/b0/n2p_baseline.py run --baseline $B --schedule $SCHED --seed $SEED --rho 0.4 --unit $U
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
