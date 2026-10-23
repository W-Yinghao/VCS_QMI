#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R22-B: NCR_A host with the online VCS / matched-BCE posterior in place of the GMM clean probability (v9_r22/ncr_vcs/r22_ncr.py).
#   sbatch [partition/qos/time/nice/exclude/job-name] r22_ncr.sbatch <vcs|bce> <chk|full> <seed>
# chk = smoke (B1 post-warm-up checkpoint of the same seed, 2 co-training epochs); full = 5 warm-up + 40 co-training epochs (NCR_A recipe).
# Output root R22_OUT (default /home/infres/yinwang/CS_QMI/outputs_v9_r22/ncr_vcs).  Resubmitting the same line resumes finished steps;
# an unfinished training attempt is moved to attempts/ and restarted (NCR co-training has no resume).
set -o pipefail
ARM=${1:?vcs|bce}; SCHED=${2:?chk|full}; SEED=${3:?seed}
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_r22b
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C $R rev-parse --short HEAD) job=$SLURM_JOB_ID arm=$ARM schedule=$SCHED seed=$SEED"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
/home/infres/yinwang/CS_QMI/env/bin/python -I $R/v9_r22/ncr_vcs/r22_ncr.py run --arm $ARM --schedule $SCHED --seed $SEED --rho 0.4
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ) host=$(hostname)"; exit $rc
