#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/logs/%x_%j.out
# R39 formal: one python process ($1 = tree r39f|r39c, $2 = train_unit.py|train_oracle.py) for the listed units ($3), in order on
# this GPU. DONE units are skipped; checkpoints resumed. In-unit fail-fast checks (no standalone gate). Self-resume: 10 min before
# the time limit the job resubmits itself once more with the same arguments (max R39_RESUBMIT_MAX=3 generations, logged).
# QOS / partition / exclude / nice / mem / time on the command line (submit_formal.sh).
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
GEN=${R39_GEN:-0}
resub() {
  if [ "$GEN" -lt "${R39_RESUBMIT_MAX:-3}" ]; then
    j=$(sbatch --parsable --export=ALL,R39_GEN=$((GEN+1)) --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 \
        --mem=${R39_MEM} --time=${R39_TIME} -J $SLURM_JOB_NAME $W/sbatch/units.sbatch "$1" "$2" "$3")
    printf "%s\t%s\t%s\tself-resume gen %s of %s (time limit): %s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$j" $((GEN+1)) "$SLURM_JOB_ID" "$3" >> $W/SUBMITTED.tsv
    echo "R39 self-resume submitted $j"
  fi
  kill -TERM $PID 2>/dev/null
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== $(date -u +%FT%TZ) $(hostname) tree $1 runner $2 gen $GEN units $3"
trap 'resub "$1" "$2" "$3"' USR1
$PY -u -B $W/$1/$2 --units "$3" &
PID=$!
wait $PID; rc=$?
if [ $rc -gt 128 ]; then wait $PID; rc=$?; fi
echo "R39_JOB_END rc=$rc $(date -u +%FT%TZ)"
exit $rc
