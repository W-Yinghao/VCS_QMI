#!/usr/bin/env bash
# Submits the frozen CREMA-D GAIN-P1 run (reports/R6/R6_GAIN_P1_CREMAD_PROTOCOL_FROZEN_20261012.yaml): waits until 3 normal-QOS slots are free
# under CAP, then submits both per-family bundles and the CPU aggregate (afterok on both, while they are still queued).  Stops on an empty id.
# Run: nohup setsid bash slurm/r6_gain_crema_submit.sh >/dev/null 2>&1 &   (stop by PID)
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; CAP=${CAP:-27}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/r6_gain_crema_submit.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }; say "start pid $$ cap $CAP"
while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -gt $((CAP - 3)) ]; do sleep 60; done
b=$(sbatch --parsable --job-name=r6crema_p1_brier slurm/r6_gain_crema_bundle.sbatch cv5 10 brier 2>>"$LOG"); [[ "$b" =~ ^[0-9]+$ ]] || { say "STOP: brier bundle id empty"; exit 1; }
c=$(sbatch --parsable --job-name=r6crema_p1_ce slurm/r6_gain_crema_bundle.sbatch cv5 10 ce 2>>"$LOG"); [[ "$c" =~ ^[0-9]+$ ]] || { say "STOP: ce bundle id empty (brier $b submitted)"; exit 1; }
g=$(sbatch --parsable --dependency=afterok:$b:$c --kill-on-invalid-dep=yes --job-name=r6crema_p1_agg slurm/r6_gain_crema_cpu.sbatch aggregate --mode cv5 2>>"$LOG")
[[ "$g" =~ ^[0-9]+$ ]] || { say "STOP: aggregate id empty (bundles $b $c submitted)"; exit 1; }
say "submitted brier $b, ce $c, aggregate $g"
