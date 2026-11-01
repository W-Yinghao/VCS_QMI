#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nice=10
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/logs/%x_%j.out
# NRCS-v1 formal chain: CHAIN = n<noise>_s<seed> (W2 parent -> F1 cache export [+ one CPU job for the 4 F1 rows] -> 6 E2 children,
# all on this job's GPU) or c2 (the 2 C2 cold-start rows). Units resume per epoch; on timeout the job resubmits itself pinned to
# the same partition (GPU type). On exit it submits the next queued chain while <=2 NRCS GPU jobs run alongside R30 (no --dependency).
set -uo pipefail
CHAIN=$1
W=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$W/package/src:$W/nrcs:$W/upstream
test -f $W/ADAPTER_READY.json || { echo "no ADAPTER_READY"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "chain $CHAIN job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION node $(hostname)"
source $W/sbatch/stage_ims.sh
LIM_END=$(( $(date +%s) + $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
M=$W/package/configs/MATRIX.csv
if [ "$CHAIN" = c2 ]; then UNITS=$(awk -F, '$2=="C2"{print $1}' $M); else
  N=${CHAIN%%_*}; N=${N#n}; S=${CHAIN##*_s}
  UNITS="$(awk -F, -v n=$((10#$N)) -v s=$S '$2=="W2" && $5==n && $4==s {print $1}' $M) EXPORT $(awk -F, -v n=$((10#$N)) -v s=$S '$2=="E2" && $5==n && $4==s {print $1}' $M)"; fi
rec() { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$1" "$2" "$3" "$4" "$5" "$6" >> $W/SUBMITTED.tsv; }
feed() {  # submit next queued chains (flock); $1 = 1 if this job is about to exit (do not count it)
  exec 9>$W/.feed.lock; flock 9
  local r30 mine cap c j
  r30=$(squeue -u $USER -h -o %j | grep -cE '^(r30_|r30v11_|r30a1_)')
  mine=$(( $(squeue -u $USER -h -o %j | grep -c '^nrcs_chain_') - $1 ))
  cap=2; [ "$r30" -eq 0 ] && cap=8
  for c in $(cat $W/CHAINS_QUEUE.txt); do
    [ $mine -ge $cap ] && break
    grep -qP "\tnrcs_chain_$c\t" $W/SUBMITTED.tsv && continue
    j=$(sbatch --parsable --job-name=nrcs_chain_$c $W/sbatch/chain.sbatch $c) && rec nrcs_chain_$c $j "chain $c" normal RTX6000PRO,L40S,H100 "fed by $SLURM_JOB_ID (r30 jobs $r30, cap $cap)" && mine=$((mine+1))
  done
  flock -u 9
}
fail=0
for U in $UNITS; do
  if [ "$U" = EXPORT ]; then
    W2U=$(awk -F, -v n=$((10#$N)) -v s=$S '$2=="W2" && $5==n && $4==s {print $1}' $M)
    if [ ! -f $O/bindings/BINDING_$CHAIN.json ]; then
      $PY -u -B $W/nrcs/export_f1.py --parent $W2U --data $O/data --ims $IMS --runs $O/runs --cache $O/f1_cache --bindings $O/bindings --binding-template $W/RUN_BINDING.json || { echo "EXPORT FAILED"; fail=1; break; }
    fi
    if ! grep -qP "\tnrcs_f1_$CHAIN\t" $W/SUBMITTED.tsv; then
      j=$(sbatch --parsable --job-name=nrcs_f1_$CHAIN --cpus-per-task=2 --mem=16G $W/sbatch/cpu_cmd.sbatch bash $W/sbatch/f1_cond.sh $CHAIN) && rec nrcs_f1_$CHAIN $j "F1 4 rows of $CHAIN" normal CPU,cpu-high "submitted by chain $SLURM_JOB_ID after export"
    fi
    continue
  fi
  test -f $O/runs/$U/DONE.json && { echo "DONE $U"; continue; }
  REM=$(( LIM_END - $(date +%s) )); [ $REM -lt 600 ] && { rc=124; break; }
  timeout $REM $PY -u -B $W/nrcs/train_gpo.py --unit $U --data $O/data --ims $IMS --runs $O/runs > $O/runs/.${U}_$SLURM_JOB_ID.log 2>&1
  rc=$?
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$U" "$SLURM_JOB_ID" "$rc" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $O/runs/ATTEMPTS.tsv
  [ $rc -eq 124 ] && break
  if [ $rc -ne 0 ]; then echo "UNIT FAILED $U rc=$rc (see $O/runs/.${U}_$SLURM_JOB_ID.log)"; fail=1
    [[ $U == *_w2_* ]] && break   # children need the parent
  fi
done
if [ "${rc:-0}" -eq 124 ]; then
  NA=$(grep -cP "\tnrcs_chain_$CHAIN\t" $W/SUBMITTED.tsv)
  if [ $NA -lt 4 ]; then
    j=$(sbatch --parsable --job-name=nrcs_chain_$CHAIN --partition=$SLURM_JOB_PARTITION $W/sbatch/chain.sbatch $CHAIN) && rec nrcs_chain_$CHAIN $j "chain $CHAIN continuation" normal $SLURM_JOB_PARTITION "continuation_of_$SLURM_JOB_ID (same GPU type)"
    echo "timeout: continuation $j"; exit 0
  fi
  echo "timeout and 4 attempts reached"; fail=1
fi
feed 1
[ $fail -eq 0 ] && echo "NRCS_CHAIN_DONE $CHAIN" || { echo "NRCS_CHAIN_FAILED $CHAIN"; exit 1; }
