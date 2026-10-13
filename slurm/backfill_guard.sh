#!/usr/bin/env bash
# Emulates a lower submission cap (CAPB, default 20) for backfill drivers / feeders that cannot be relaunched: every 30 s, if my normal-QOS
# job count is >= CAPB, SIGSTOP the given PIDs (they stop submitting; their queued jobs keep running); otherwise SIGCONT them.  Main-route
# drivers keep their CAP 27, so they always have headroom.  Exits when none of the PIDs is alive.  Run: CAPB=20 nohup setsid bash
# slurm/backfill_guard.sh <pid> [<pid> ...] >/dev/null 2>&1 &   — stop by PID (then SIGCONT the backfill PIDs).  Log: slurm_logs/backfill_guard.log
CAPB=${CAPB:-20}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/backfill_guard.log; PIDS="$*"; state=""
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$ capb $CAPB pids $PIDS"
while :; do
  alive=""; for p in $PIDS; do kill -0 $p 2>/dev/null && alive="$alive $p"; done
  [ -z "$alive" ] && { say "no backfill pid alive; done"; exit 0; }
  n=$(squeue -h -u "$USER" -q normal | wc -l)
  if [ "$n" -ge "$CAPB" ]; then [ "$state" != stop ] && { kill -STOP $alive; state=stop; say "count $n >= $CAPB: paused$alive"; }
  else [ "$state" != cont ] && { kill -CONT $alive; state=cont; say "count $n < $CAPB: resumed$alive"; }; fi
  sleep 30
done
