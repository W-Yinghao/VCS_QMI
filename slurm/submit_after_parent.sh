#!/usr/bin/env bash
# Submit one sbatch job under the normal-QOS CAP, after a parent job: afterok while the parent is still known to slurmctld, otherwise only if the
# parent's log ends with exit=0 (dependencies on purged jobs fail while sacct is down).  Stops on an empty job id or a failed parent.
# Usage: nohup setsid bash slurm/submit_after_parent.sh <parent id> <parent log> <sbatch args...> >/dev/null 2>&1 &     (stop by PID)
P=$1; PLOG=$2; shift 2; CAP=${CAP:-27}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/submit_after_parent.log
echo "$(date -u +%FT%TZ) start pid $$ parent $P: sbatch $*" >> $LOG
while true; do
  if [ "$(squeue -h -u "$USER" -q normal | wc -l)" -lt "$CAP" ]; then
    st=$(squeue -h -j "$P" -o %T 2>/dev/null)
    case "$st" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|COMPLETED) dep="--dependency=afterok:$P --kill-on-invalid-dep=yes";;
      *) if grep -q "^exit=0" "$PLOG" 2>/dev/null; then dep=""; else echo "$(date -u +%FT%TZ) STOP: parent $P not queued and its log has no exit=0" >> $LOG; exit 1; fi;;
    esac
    j=$(sbatch --parsable $dep "$@" 2>>$LOG)
    [[ "$j" =~ ^[0-9]+$ ]] || { echo "$(date -u +%FT%TZ) STOP: empty job id" >> $LOG; exit 1; }
    echo "$(date -u +%FT%TZ) submitted $j ${dep:-(parent finished ok)} sbatch $*" >> $LOG; exit 0
  fi
  sleep 60
done
