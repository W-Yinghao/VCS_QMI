#!/usr/bin/env bash
# REG-P1: submit the 20 jobs the first driver lost (its afterok on the purged perturb job 1034658 failed while sacct was down).
# Each training job and its registration are submitted back-to-back, so the dependency always names a job still in the queue; stops on any
# empty job id; waits for 2 free slots under CAP.  The aggregation is run separately once every registration has finished (no long chains).
# Run from the login node: nohup setsid bash slurm/r6_reg_p1_resubmit.sh >/dev/null 2>&1 &   — stop by PID only.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-27}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/r6_reg_p1_resubmit.log; IDS=/home/infres/yinwang/CS_QMI/slurm_logs/r6_reg_p1_register_ids.txt
echo "$(date -u +%FT%TZ) start pid $$ cap $CAP" >> "$LOG"
[ -s "$IDS" ] || printf '%s\n' 1034659 1034660 1034661 1034668 1034671 > "$IDS"          # registrations already queued by the first driver
pair() {   # $1 job-name, $2.. first-job args ; then REGNAME and REG args via globals
  while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -gt $((CAP - 2)) ]; do sleep 120; done
  local a b
  a=$(sbatch --parsable --job-name="$1" slurm/r6_reg_gpu.sbatch "${@:2}" 2>>"$LOG")
  [[ "$a" =~ ^[0-9]+$ ]] || { echo "$(date -u +%FT%TZ) STOP: empty id for $1" >> "$LOG"; exit 1; }
  b=$(sbatch --parsable --dependency=afterok:$a --kill-on-invalid-dep=yes --job-name="$REGNAME" slurm/r6_reg_gpu.sbatch "${REGARGS[@]}" 2>>"$LOG")
  [[ "$b" =~ ^[0-9]+$ ]] || { echo "$(date -u +%FT%TZ) STOP: empty id for $REGNAME (after $a)" >> "$LOG"; exit 1; }
  echo "$b" >> "$IDS"; echo "$(date -u +%FT%TZ) $a $1 -> $b $REGNAME" >> "$LOG"
}
for s in 0 1 2; do
  for o in vcs logistic infonce; do
    if [ $s = 0 ] && [ $o != infonce ]; then continue; fi                                  # vcs / logistic s0 already queued
    REGNAME=r6reg_reg_${o}_s$s; REGARGS=(register --method $o --seed $s); pair r6reg_train_${o}_s$s train --objective $o --seed $s
  done
  REGNAME=r6reg_reg_comir_s$s; REGARGS=(register --method comir --seed $s); pair r6reg_comir_s$s comir --seed $s
done
echo "$(date -u +%FT%TZ) done: all pairs submitted; registration ids in $IDS (run the aggregation when they have all finished)" >> "$LOG"
