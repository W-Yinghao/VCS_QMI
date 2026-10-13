#!/usr/bin/env bash
#SBATCH --job-name=r7e1_hiprune_group
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=09:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Owner decision 2026-10-13 (HiPrune's 12 frozen units grouped one job per benchmark): runs the FROZEN per-unit wrapper
# slurm/r7_e1_gpu.sbatch 'run hiprune <bench> <i/n> anchor' for each '<bench>:<i/n>' given, one after the other on this GPU (L40S:
# FlashAttention-2).  A unit whose marker (its #MARK in the frozen r7_e1_lines.txt) exists is skipped; inner 'exit=' lines are renamed
# 'unit_exit='.  Exit 0 only if every unit ends with its marker.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
F=/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40/eval_e1/r7_e1_lines.txt; fail=0
echo "group host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) units: $*"
for u in "$@"; do
  b=${u%%:*}; s=${u#*:}
  m=$(grep -E "run hiprune $b $s( |$)" "$F" | head -1 | sed 's/.* #MARK //; s/ #.*//')
  [ -n "$m" ] || { echo "[group] $u: no line in $F"; fail=$((fail + 1)); continue; }
  if [ -f "$m" ]; then echo "[group] $u: already complete"; continue; fi
  echo "[group] $u: start $(date -u +%FT%TZ)"
  bash slurm/r7_e1_gpu.sbatch run hiprune "$b" "$s" anchor 2>&1 | sed -u 's/^exit=/unit_exit=/'; rc=${PIPESTATUS[0]}
  if [ -f "$m" ]; then echo "[group] $u: done (rc $rc) $(date -u +%FT%TZ)"; else echo "[group] $u: NO marker (rc $rc)"; fail=$((fail + 1)); fi
done
echo "exit=$fail end=$(date -u +%FT%TZ)"; exit $fail
