#!/bin/bash
# V7-03 driver (login node, nohup setsid): the 9 M tuning units (seed 17) -> lr choice per arm (CPU job) -> the 9 formal M units.  A unit is
# resubmitted (resume from its resume.pt) only after its job left the queue without done.json, at most RETRIES times.  Single writer: never
# edit this file while it runs.  The 3 N0 units are submitted separately (they need no tuning).
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 || exit 1
BIG=/projects/EEG-foundation-model/yinghao/outputs_v7; U=$BIG/units; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/v7_driver.log; RETRIES=${RETRIES:-2}; POLL=${POLL:-120}
P="--partition=RTX6000PRO,H100,L40S --exclude=node51,node60"; ARMS="M0_semantic_l1 M1_semantic_logistic M2_semantic_vcs"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
declare -A JOB TRIES
submit() { local unit=$1 arm=$2 lr=$3 seed=$4; local j; j=$(sbatch --parsable --job-name=$unit $P --time=05:00:00 slurm/v7/v7_gpu.sbatch m $arm $lr $seed $unit) || { say "STOP: sbatch failed for $unit"; exit 2; }
  JOB[$unit]=$j; TRIES[$unit]=$(( ${TRIES[$unit]:-0} + 1 )); say "$j $unit ($arm lr $lr seed $seed, try ${TRIES[$unit]})"; }
run_stage() {   # args: list of "unit arm lr seed"
  local spec unit; for spec in "$@"; do set -- $spec; [ -f $U/$1/done.json ] || submit $1 $2 $3 $4; done
  while :; do
    local pending=0
    for spec in "${STAGE[@]}"; do set -- $spec; unit=$1
      [ -f $U/$unit/done.json ] && continue
      if [ -n "${JOB[$unit]}" ] && [ -n "$(squeue -h -j ${JOB[$unit]} -o %i 2>/dev/null)" ]; then pending=1; continue; fi
      sleep 30; [ -f $U/$unit/done.json ] && continue                       # NFS visibility
      [ ${TRIES[$unit]:-0} -gt $RETRIES ] && { say "STOP: $unit failed $RETRIES retries"; exit 3; }
      say "RETRY: job ${JOB[$unit]} left the queue without $U/$unit/done.json"; submit $spec; pending=1
    done
    [ $pending -eq 0 ] && return 0; sleep $POLL
  done; }
say "start pid $$ (RETRIES $RETRIES)"
STAGE=(); for arm in $ARMS; do for lr in 5e-06 1e-05 2e-05; do STAGE+=("V7-TUNE-$arm-lr$lr $arm $lr 17"); done; done
run_stage "${STAGE[@]}"; say "tuning complete (9 units)"
CH=$BIG/lr_choice.json
if [ ! -f $CH ]; then
  j=$(sbatch --parsable --job-name=v7_tune_select --partition=CPU --gres=gpu:0 --cpus-per-task=2 --mem=4G --time=00:20:00 \
      --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out --wrap="cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && /projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I scripts/v7/v7_tune_select.py --units-dir $U --out $CH; echo exit=\$?")
  say "$j lr selection"; while [ -n "$(squeue -h -j $j -o %i 2>/dev/null)" ]; do sleep 30; done; sleep 20
fi
[ -f $CH ] || { say "STOP: no $CH"; exit 4; }
say "lr choice: $(tr -d '\n ' < $CH | grep -oE '"(M[0-9]_[a-z_]+)":\{"chosen_lr":[0-9.e-]+' | tr '\n' ' ')"
STAGE=()
for arm in $ARMS; do lr=$(/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]['chosen_lr'])" $CH $arm)
  for s in 0 1 2; do STAGE+=("V7-FIT-$arm-s$s $arm $lr $s"); done; done
run_stage "${STAGE[@]}"; say "done: 9 formal M units complete"
