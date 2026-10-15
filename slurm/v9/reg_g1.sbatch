#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g1
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 grouped GPU job: runs the given unit ids one after the other (train / fit + REG-P2 registration; finished units are skipped).
#   sbatch [--nice=N] slurm/v9/reg_g1.sbatch <unit_id> [<unit_id> ...]        (V9_REG_ROOT may override the output root)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) code=$(cat src/v9/reg/g1.py src/v9/reg/r6_reg_pilot.py src/v9/reg/r6_reg_p2.py | sha256sum | cut -c1-16)"
rc_all=0
for u in "$@"; do
  echo "[unit] $(date -u +%FT%TZ) $u"; "$PYTHON" scripts/v9/reg_g1.py run --unit "$u"; rc=$?; echo "[unit-exit] $u rc=$rc $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
