#!/usr/bin/env bash
# P5: 3 methods x seeds {0,1,2}, 200 epochs, ALL submitted in parallel (no chaining), one GPU each, descriptive job names.
# Usage: ./submit_confirm200.sh                 -> all 9 runs + P6
#        ./submit_confirm200.sh vcs:1 vcs:2 ... -> only the listed method:seed units (+ P6 depending on them and on EXTRA_DEPS)
# EXTRA_DEPS="1007792:1007795:1007798" adds already-running job ids to the P6 dependency.
set -euo pipefail
cd "$(dirname "$0")"
units=("$@"); [ ${#units[@]} -eq 0 ] && units=(vcs:0 vcs:1 vcs:2 simclr:0 simclr:1 simclr:2 vicreg:0 vicreg:1 vicreg:2)
declare -A JOB
submit() { local name=$1; shift; local id; id=$(sbatch --parsable "$@" | cut -d';' -f1); JOB[$name]=$id; echo "$name -> job $id ($*)"; }
ids=()
for u in "${units[@]}"; do
  m=${u%%:*}; s=${u##*:}
  submit "p5_${m}_s${s}" --job-name="p5_${m}_s${s}" --export=ALL,METHOD=$m,SEED=$s p5_confirm.sbatch
  ids+=("${JOB[p5_${m}_s${s}]}")
done
dep=$(IFS=:; echo "${ids[*]}"); [ -n "${EXTRA_DEPS:-}" ] && dep="${EXTRA_DEPS}:${dep}"
submit p6 --job-name=p6_summarize --dependency=afterany:${dep} p6_summarize_confirm.sbatch
python3 - <<PY
import json, time, os
jobs = dict(l.split("=") for l in """$(for k in "${!JOB[@]}"; do echo "$k=${JOB[$k]}"; done)""".split())
rec = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": "P5_confirm200 parallel (+P6 summarize)", "jobs": jobs,
       "p6_dependency": "afterany:${dep}", "commit": os.popen("git rev-parse HEAD").read().strip()}
p = "../reports/job_ids.json"; hist = json.load(open(p)) if os.path.exists(p) else []
hist.append(rec); json.dump(hist, open(p, "w"), indent=2); print(json.dumps(rec, indent=2))
PY
