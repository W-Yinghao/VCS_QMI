#!/usr/bin/env bash
# P5: 3 methods x seeds {0,1,2}, 200 epochs. One chain per method (seed0 -> seed1 -> seed2, afterany); the three chains run
# concurrently (3 single-GPU jobs at a time). P6 summarizes after the last job of every chain.
set -euo pipefail
cd "$(dirname "$0")"
declare -A JOB
submit() { local name=$1; shift; local id; id=$(sbatch --parsable "$@" | cut -d';' -f1); JOB[$name]=$id; echo "$name -> job $id ($*)"; }
last=()
for m in vcs simclr vicreg; do
  dep=""
  for s in 0 1 2; do
    submit "p5_${m}_s${s}" $dep --export=ALL,METHOD=$m,SEED=$s p5_confirm.sbatch
    dep="--dependency=afterany:${JOB[p5_${m}_s${s}]}"
  done
  last+=("${JOB[p5_${m}_s2]}")
done
submit p6 --dependency=afterany:$(IFS=:; echo "${last[*]}") p6_summarize_confirm.sbatch
python3 - <<PY
import json, time, os
jobs = dict(l.split("=") for l in """$(for k in "${!JOB[@]}"; do echo "$k=${JOB[$k]}"; done)""".split())
rec = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": "P5_confirm200 (+P6 summarize)", "jobs": jobs,
       "commit": os.popen("git rev-parse HEAD").read().strip()}
p = "../reports/job_ids.json"; hist = json.load(open(p)) if os.path.exists(p) else []
hist.append(rec); json.dump(hist, open(p, "w"), indent=2); print(json.dumps(rec, indent=2))
PY
