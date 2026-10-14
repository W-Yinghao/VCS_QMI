#!/usr/bin/env bash
#SBATCH --job-name=v7x_group
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Grouped GPU job: runs FROZEN per-unit wrappers one after another on the same GPU (submit slots and model loads saved).
# Each argument is one spec "<wrapper>|<mode> <args...>", e.g. "v7|native-rank /path/ckpt.pth N0_s0" (wrapper v7 = slurm/v7/v7_gpu.sbatch,
# v7x = slurm/v7x/v7x_gpu.sbatch).  A failing spec is logged and the next one runs; the job's exit code is the first failure.
# Each wrapper is snapshotted at job start (bash reads scripts incrementally: an edit to a wrapper while a group runs must not reach it).
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; first=0; SNAP=$(mktemp -d /tmp/v7x_group_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT
cp slurm/v7/v7_gpu.sbatch $SNAP/v7_gpu.sbatch; cp slurm/v7x/v7x_gpu.sbatch $SNAP/v7x_gpu.sbatch
echo "wrappers: $(sha256sum $SNAP/*.sbatch | cut -c1-16 | tr '\n' ' ')"
for spec in "$@"; do
  w=${spec%%|*}; args=${spec#*|}; f=$SNAP/${w}_gpu.sbatch
  echo "=== $(date -u +%FT%TZ) $w $args"; bash $f $args; rc=$?; echo "=== rc=$rc"
  [ $rc -ne 0 ] && [ $first -eq 0 ] && first=$rc
done
echo "exit=$first end=$(date -u +%FT%TZ)"; exit $first
