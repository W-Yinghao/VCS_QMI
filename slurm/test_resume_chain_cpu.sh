#!/usr/bin/env bash
# End-to-end test of the walltime chain on the CPU partition (no GPU taken from training):
#   link 1: fresh smoke training (4 views, 6 steps in 2 pseudo-epochs of 3) interrupted cleanly after step 4 -> STOPPED_BUDGET, last.pt at epoch 1
#   link 2 (afterany): the REAL slurm/run_unit.sbatch body, which must take the resume branch, finish epoch 2, then evaluate initial.pt and epoch_002.pt
# usage: slurm/test_resume_chain_cpu.sh  -> prints RUN_ID and the two job ids
set -euo pipefail
REPO=/home/infres/yinwang/CS_QMI/ssl_pilot; LOGS=/home/infres/yinwang/CS_QMI/slurm_logs
RUN_ID=RESUME_TEST_$(date -u +%Y%m%dT%H%M%S); CFG=configs/test_resume_chain_views4.yaml
SMOKE="--allow-cpu --smoke-steps 6 --smoke-epoch-steps 3"
J1=$(sbatch --parsable --partition=CPU --cpus-per-task=16 --mem=48G --time=02:00:00 --job-name=resume_test_link1 --output=$LOGS/resume_test_link1_%j.out \
  --wrap="source $REPO/slurm/common.sh; export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; banner 'resume test link 1 (interrupted)'; run_fg \$PYTHON -m vcs_ssl.train --config $CFG --run-id $RUN_ID $SMOKE --stop-after-steps 4; rc=\$?; echo exit=\$rc; exit 0" | cut -d';' -f1)
J2=$(sbatch --parsable --dependency=afterany:$J1 --partition=CPU --cpus-per-task=16 --mem=48G --time=02:00:00 --job-name=resume_test_link2 --output=$LOGS/resume_test_link2_%j.out \
  --export=ALL,CFG=$CFG,RUN_ID=$RUN_ID,FINAL_CKPT=epoch_002.pt,EXTRA_ARGS="$SMOKE",OMP_NUM_THREADS=16,MKL_NUM_THREADS=16 \
  --wrap="bash $REPO/slurm/run_unit.sbatch" | cut -d';' -f1)
echo "$RUN_ID $J1 $J2"
