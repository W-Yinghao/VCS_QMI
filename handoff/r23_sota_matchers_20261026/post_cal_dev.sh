#!/usr/bin/env bash
# When every pre-registered CAL config of all 11 methods has 80 rows: run the frozen `harness.py lock` (CAL only, no DEV read),
# then feed one DEV job per LOCKED method through the existing line feeder (TAG r23_sota_dev). Touch STOP_POST_CAL_DEV to stop.
D=/home/infres/yinwang/CS_QMI/handoff/r23_sota_matchers_20261026
F=/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/deps_watch
PY=/projects/EEG-foundation-model/yinghao/envs/sota_matchers/bin/python
LOG=$D/post_cal_dev.log
cd "$D" || exit 2
echo "$(date -u +%FT%TZ) start (pid $$)" >> "$LOG"
while :; do
  [ -e STOP_POST_CAL_DEV ] && { echo "$(date -u +%FT%TZ) STOP" >> "$LOG"; exit 0; }
  st=$(TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home_sota "$PY" -B -c "
import sys; sys.argv=['x']; sys.path.insert(0,'.')
import harness as H, methods as M
miss=[(m,c) for m in M.FAMILIES for c in H.configs(m) if len(H.rows(H.HERE/'cal'/m/(c+'.jsonl')))<80]
print('COMPLETE' if not miss else 'WAITING %d %s' % (len(miss), sorted({m for m,_ in miss})))" 2>>"$LOG")
  echo "$(date -u +%FT%TZ) $st" >> "$LOG"
  [ "${st%% *}" = COMPLETE ] && break
  sleep 300
done
TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home_sota "$PY" -B harness.py lock >> "$LOG" 2>&1 || { echo "LOCK FAILED" >> "$LOG"; exit 3; }
"$PY" -B -c "
import json
lk=json.load(open('LOCK.json'))['methods']
T={'roma':'05:00:00','minima_roma':'05:00:00','dinov2_host':'08:00:00'}
with open('LINES_DEV.txt','w') as f:
    for m,v in lk.items():
        if v['status']=='LOCKED':
            f.write(f\"sbatch --job-name=r23_sota_dev_{m} --time={T.get(m,'03:00:00')} /home/infres/yinwang/CS_QMI/handoff/r23_sota_matchers_20261026/gpu.sbatch dev --method {m}\n\")
" >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) LOCK written; feeding $(wc -l < LINES_DEV.txt) DEV lines" >> "$LOG"
bash "$F/app_lines_feed.sh" "$D/LINES_DEV.txt" r23_sota_dev 30 0
echo "$(date -u +%FT%TZ) done" >> "$LOG"
