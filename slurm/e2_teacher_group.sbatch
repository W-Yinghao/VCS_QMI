#!/usr/bin/env bash
#SBATCH --job-name=r7e2t_group
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=09:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Owner decision 2026-10-13 ("按选项一", about 5 bundles per job instead of one job per bundle): runs the FROZEN per-bundle wrapper
# slurm/r7_e2_gpu.sbatch 'bundle <id>' for each bundle id given, one after the other on this GPU.  A bundle whose marker (its #MARK in the
# frozen teacher_lines.txt) exists is skipped; the frozen teacher also resumes per shard.  The inner wrappers' 'exit=' lines are renamed
# 'bundle_exit=' so the job log carries one final exit line.  Exit 0 only if every bundle ends with its marker.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
T=/home/infres/yinwang/CS_QMI/outputs/R7_E2/lines/teacher_lines.txt; fail=0
echo "group host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) bundles: $*"
for b in "$@"; do
  m=$(grep -E "bundle $b( |$)" "$T" | head -1 | sed 's/.* #MARK //; s/ #.*//')
  [ -n "$m" ] || { echo "[group] $b: no line in $T"; fail=$((fail + 1)); continue; }
  if [ -f "$m" ]; then echo "[group] $b: already complete"; continue; fi
  echo "[group] $b: start $(date -u +%FT%TZ)"
  bash slurm/r7_e2_gpu.sbatch bundle "$b" 2>&1 | sed -u 's/^exit=/bundle_exit=/'; rc=${PIPESTATUS[0]}
  if [ -f "$m" ]; then echo "[group] $b: done (rc $rc) $(date -u +%FT%TZ)"; else echo "[group] $b: NO marker (rc $rc)"; fail=$((fail + 1)); fi
done
echo "exit=$fail end=$(date -u +%FT%TZ)"; exit $fail
