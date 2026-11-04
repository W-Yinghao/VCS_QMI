#!/usr/bin/env bash
# usage: transfer_audit.sbatch <COND> <REL...>   (package tools/readonly_transfer_audit.py; no truth)
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr3_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr3_20261104/sbatch/common_env.sh
C=$1; shift; echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
for R in "$@"; do
  case $R in
    OBS|J_MASS_OBS) P=$CR3_OUT/graphs/$C/$R.npz;;
    KEEP_ORACLE) P=$CR2_OUT/oracle/$C/CR2_${C}_KEEP_ORACLE_s1001.npz;;
    FULL_TRUE_ORACLE) P=$CR3_OUT/oracle/$C/FULL_TRUE_ORACLE.npz;;
    *) S=${R%_KEEP_TOP1}; P=$CR2_OUT/relations/$C/CR2_${C}_${S}_KEEP_TOP1_s1001.npz;;
  esac
  $CR3_PY -I -B pkg/tools/readonly_transfer_audit.py --relation $P --observed-npy $CR3_OUT/graphs/$C/observed_image_for_text.npy \
     --out $CR3_OUT/transfer_audit/$C/$R.json > /dev/null
  echo "AUDITED $C $R"
done
echo "EXIT_OK $(date -u +%FT%TZ)"
