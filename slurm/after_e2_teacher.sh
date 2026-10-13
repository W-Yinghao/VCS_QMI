#!/usr/bin/env bash
# One-off (coordinator, 2026-10-13; R7-E2a note e2a_6): when all 36 teacher bundle markers exist and no r7e2t_ job is queued, relaunch the
# frozen teacher driver once (CAP 20).  It skips the data stage (data_ready.json exists) and every bundle (markers), then runs the frozen
# R7-40a completeness step that the grouped rest driver waits for.  Never relaunches while any teacher job is queued (no duplicate units).
L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/after_e2_teacher.log; T=/home/infres/yinwang/CS_QMI/outputs/R7_E2/teacher
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$"
until [ "$(ls $T/bundle_f*.done.json 2>/dev/null | wc -l)" -ge 36 ] && [ -z "$(squeue -h -u "$USER" -o '%j' | grep '^r7e2t_')" ]; do sleep 300; done
say "all 36 bundle markers present and no teacher job queued: relaunching the teacher driver for its completeness step"
cd /home/infres/yinwang/CS_QMI/ssl_pilot && CAP=20 nohup setsid bash slurm/r7_e2_submit.sh teacher >/dev/null 2>&1 &
say "launched"
