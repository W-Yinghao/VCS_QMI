#!/usr/bin/env bash
# One-off (coordinator, 2026-10-13): wait for the E2 smoke (job 1037541) to end, then launch the frozen E2a teacher driver (its own gates check the
# probe report and the replay file and STOP otherwise) and keep the r7e2t_ teacher bundles at Nice 300 (backfill behind main-route units, E2a-6).
L=/home/infres/yinwang/CS_QMI/slurm_logs; S=$L/r7e2_smoke_1037541.out; LOG=$L/after_e2_smoke.log; E2=/home/infres/yinwang/CS_QMI/outputs/R7_E2
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$"
until [ -f $S ] && grep -qE "^exit=|CANCELLED|TIME LIMIT" $S; do sleep 30; done
say "smoke ended: $(grep -E '^exit=' $S | tail -1)"; sleep 120                                   # NFS visibility of the smoke outputs
ls $E2/smoke >/dev/null 2>&1
if grep -q '"pass": true' $E2/smoke/probe/probe_report.json 2>/dev/null && [ -f $E2/smoke/reuse_replay.json ]; then
  cd /home/infres/yinwang/CS_QMI/ssl_pilot && nohup setsid bash slurm/r7_e2_submit.sh teacher >/dev/null 2>&1 &
  sleep 5; PAT='^r7e2t_' NICE=300 nohup setsid bash slurm/nice_filler.sh >/dev/null 2>&1 &
  say "launched the E2a teacher driver and nice_filler (^r7e2t_, 300)"
else say "NOT launched: probe report not passing or replay file missing (see $S)"; fi
