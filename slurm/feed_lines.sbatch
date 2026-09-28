#!/usr/bin/env bash
#SBATCH --job-name=feed_lines
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/feed_%x_%j.out
# Generic queue feeder under the 30-job cap: each non-comment line of LINES_FILE is a complete shell command that ends in an `sbatch --parsable ...`
# call (env assignments may prefix it).  Lines are submitted in file order whenever the user's queue is below CAP; a rejected submission
# (no numeric id) is retried after 120 s.  Every id is appended to reports/job_ids.json under STAGE.  Skips a line whose SKIP_IF file exists.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${LINES_FILE:?}"; : "${STAGE:?}"; CAP=${CAP:-29}
banner "feeder ${STAGE}: ${LINES_FILE} cap=${CAP}"
n_ok=0
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  label=$(echo "$line" | grep -oE -- '--job-name=[^ ]+' | head -1 | cut -d= -f2); label=${label:-line}
  while :; do
    while [ "$(squeue -h -u "$USER" | wc -l)" -ge "$CAP" ]; do sleep 120; done
    id=$(eval "$line" 2>/dev/null | cut -d';' -f1 || true)
    if [[ "$id" =~ ^[0-9]+$ ]]; then break; fi
    echo "[retry] $(date -u +%FT%TZ) $label rejected (queue $(squeue -h -u "$USER" | wc -l))"; sleep 120
  done
  echo "[submit] $(date -u +%FT%TZ) $label -> $id queue=$(squeue -h -u "$USER" | wc -l)"; n_ok=$((n_ok+1))
  "$PYTHON" - "$STAGE" "$label" "$id" <<'PY'
import json, sys, time, os, fcntl
stage, label, jid = sys.argv[1:4]; p = "reports/job_ids.json"
with open(p + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX); hist = json.load(open(p))
    hist.append({"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": f"{stage} (feeder): {label}", "commit": os.popen("git rev-parse HEAD").read().strip(), "jobs": [jid]})
    tmp = p + f".tmp.{os.getpid()}"; json.dump(hist, open(tmp, "w"), indent=1); os.replace(tmp, p)
PY
  sleep 5
done < "$LINES_FILE"
echo "feeder done: $n_ok submitted end=$(date -u +%FT%TZ)"
