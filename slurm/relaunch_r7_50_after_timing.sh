#!/usr/bin/env bash
# Waits (login node, sleep loop) for D1-30's gqa_dev / chartqa_val profiling timing files, then relaunches the R7-50 phase-L driver, which
# resumes at the manifest step (execution note r7_50_3).  Stop by PID.  Log: slurm_logs/r7_50_relaunch.log
T=/home/infres/yinwang/CS_QMI/outputs/R7_D1/d1_30/timing; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/r7_50_relaunch.log
echo "$(date -u +%FT%TZ) start pid $$ waiting for D1-30 timing files" >> $LOG
until [ -f $T/gqa_dev/timing_gqa_dev.json ] && [ -f $T/gqa_dev/sparsevlm_timing_gqa_dev.json ] && [ -f $T/chartqa_val/timing_chartqa_val.json ] && [ -f $T/chartqa_val/sparsevlm_timing_chartqa_val.json ]; do sleep 300; done
sleep 120   # NFS visibility margin
cd /home/infres/yinwang/CS_QMI/ssl_pilot && CAP=27 POLL=60 nohup setsid bash slurm/r7_50_submit.sh >/dev/null 2>&1 &
echo "$(date -u +%FT%TZ) timing files present; R7-50 phase-L driver relaunched" >> $LOG
