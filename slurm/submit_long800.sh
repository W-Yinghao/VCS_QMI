#!/usr/bin/env bash
# P8: VCS only, seeds {0,1,2}, 800 epochs, submitted in parallel; P9 summarize after all three.
set -euo pipefail
cd "$(dirname "$0")"
declare -A JOB
submit() { local name=$1; shift; local id; id=$(sbatch --parsable "$@" | cut -d';' -f1); JOB[$name]=$id; echo "$name -> job $id ($*)"; }
ids=()
for s in 0 1 2; do
  submit "p8_vcs_s${s}" --job-name="p8_vcs800_s${s}" --export=ALL,CFG=configs/cifar10_long800_vcs_seed${s}.yaml,RUN_ID=P8_vcs800_seed${s},FINAL_CKPT=epoch_800.pt run_unit.sbatch
  ids+=("${JOB[p8_vcs_s${s}]}")
done
submit p9 --job-name=p9_summarize --dependency=afterany:$(IFS=:; echo "${ids[*]}") --export=ALL,STAGE=P8_long800,OUT=reports/P9_long800_results_table.md summarize_stage.sbatch
python3 - <<PY
import json, time, os
jobs = dict(l.split("=") for l in """$(for k in "${!JOB[@]}"; do echo "$k=${JOB[$k]}"; done)""".split())
rec = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": "P8_long800 VCS-only seeds 0-2 parallel (+P9 summarize)", "jobs": jobs,
       "commit": os.popen("git rev-parse HEAD").read().strip()}
p = "../reports/job_ids.json"; hist = json.load(open(p)) if os.path.exists(p) else []
hist.append(rec); json.dump(hist, open(p, "w"), indent=2); print(json.dumps(rec, indent=2))
PY
