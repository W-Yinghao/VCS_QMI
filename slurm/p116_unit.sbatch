#!/usr/bin/env bash
#SBATCH --job-name=p116_unit
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P116 bundle: runs the units in UNITS (one per line: features|family|cells|repeats|seed|out) sequentially in ONE job; a unit whose <out>.json
# exists is skipped (resumable).  Partition / gres may be overridden on the sbatch command line (e.g. -p A100 --gres=gpu:1).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${UNITS:?}"; rc=0
banner "P116 bundle $UNITS"
while IFS='|' read -r feat fam cells reps seed out || [ -n "$feat" ]; do
  [[ -z "$feat" || "$feat" =~ ^# ]] && continue
  [ -f "$out.json" ] && { echo "[skip] $out"; continue; }
  echo "[unit] $(date -u +%FT%TZ) $fam $cells R=$reps seed=$seed"
  "$PYTHON" scripts/p116_solve.py --features "$feat" --family "$fam" --cells "$cells" --repeats "$reps" --perms 200 --seed "$seed" --out "$out" || rc=$?
done < "$UNITS"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
