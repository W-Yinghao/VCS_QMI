#!/usr/bin/env bash
# Submit one sbatch script when the user's queue is below the 30-job cap (waits, polling every 60 s, up to MAX_WAIT s).
# usage: slurm/submit_when_free.sh <label> <sbatch args...>   -> prints "<label> -> <jobid>" or "<label> TIMEOUT"; records in reports/job_ids.json
set -u
LABEL=$1; shift; CAP=${CAP:-30}; MAX_WAIT=${MAX_WAIT:-540}; t0=$(date +%s)
cd /home/infres/yinwang/CS_QMI/ssl_pilot
while :; do
  n=$(squeue -h -u "$USER" | wc -l)
  if [ "$n" -lt "$CAP" ]; then
    id=$(sbatch --parsable "$@" 2>/dev/null || true)
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      echo "$LABEL -> $id (queue was $n)"
      /home/infres/yinwang/CS_QMI/env/bin/python - "$LABEL" "$id" <<'PY'
import json, sys, time, os, fcntl
label, jid = sys.argv[1], sys.argv[2]; p = "reports/job_ids.json"
with open(p + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX); hist = json.load(open(p))
    hist.append({"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": label + " (submit_when_free)", "commit": os.popen("git rev-parse HEAD").read().strip(), "jobs": [jid]})
    tmp = p + ".tmp"; json.dump(hist, open(tmp, "w"), indent=1); os.replace(tmp, p)
PY
      exit 0
    fi
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT" ]; then echo "$LABEL TIMEOUT (queue $n)"; exit 4; fi
  sleep 60
done
