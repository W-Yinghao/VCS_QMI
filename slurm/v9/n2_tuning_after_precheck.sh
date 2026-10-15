#!/usr/bin/env bash
# V9-N2: after the seed-917 precheck job ends with exit=0, plan the tuning runs from its merge groups, commit the plan (own paths), start the tuning feeder.
# usage: nohup setsid bash slurm/v9/n2_tuning_after_precheck.sh <precheck_job_id> >/dev/null 2>&1 &
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9 || exit 1
J=$1; L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/v9_n2_feed.log; PY=/home/infres/yinwang/CS_QMI/env/bin/python; export PYTHONNOUSERSITE=1
echo "$(date -u +%FT%TZ) tuning-after-precheck pid $$ waiting for $J" >> "$LOG"
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 60; done
lf=$L/v9_n2_precheck_${J}.out
if ! grep -q "exit=0" "$lf" 2>/dev/null; then echo "$(date -u +%FT%TZ) precheck $J did not exit 0 — tuning NOT planned" >> "$LOG"; exit 1; fi
$PY scripts/v9/n2_plan_units.py --precheck-dir /projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k --seeds 917 --kind tuning --dest reports/V9/V9_N2_units >> "$LOG" 2>&1 || { echo "$(date -u +%FT%TZ) tuning plan failed" >> "$LOG"; exit 1; }
mkdir -p reports/V9/V9_N2_precheck && cp /projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4/s917/precheck/precheck_f30k_rho0.4_s917.* /projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0/s917/precheck/precheck_f30k_rho0_s917.* reports/V9/V9_N2_precheck/ 2>>"$LOG"
git add reports/V9/V9_N2_units reports/V9/V9_N2_precheck && git commit -q -m "V9-N2: tuning seed 917 shared stage (precheck job $J, not a gate) and the tuning run plan after its merge groups (results only)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) tuning plan committed $(git log -1 --format=%h); starting tuning feeder" >> "$LOG"
nohup setsid bash slurm/v9/n2_feed.sh reports/V9/V9_N2_units/tuning_feed.txt >/dev/null 2>&1 &
