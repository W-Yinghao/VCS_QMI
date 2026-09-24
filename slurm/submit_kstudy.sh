#!/usr/bin/env bash
# P10: VCS only, K in {8,64} x seeds {0,1,2}, 200 epochs, parallel; P11 summarize after all six.
# GATE_DEP=<job id> makes every unit wait for that job to succeed (afterok), e.g. the CPU test job.
set -euo pipefail
cd "$(dirname "$0")"
declare -A JOB
submit() { local name=$1; shift; local id; id=$(sbatch --parsable "$@" | cut -d';' -f1); JOB[$name]=$id; echo "$name -> job $id ($*)"; }
ids=()
for k in 8 64; do for s in 0 1 2; do
  submit "p10_k${k}_s${s}" --job-name="p10_vcs_k${k}_s${s}" ${GATE_DEP:+--dependency=afterok:$GATE_DEP} --export=ALL,CFG=configs/cifar10_k${k}_vcs_seed${s}.yaml,RUN_ID=P10_vcs_k${k}_seed${s},FINAL_CKPT=epoch_200.pt run_unit.sbatch
  ids+=("${JOB[p10_k${k}_s${s}]}")
done; done
submit p11 --job-name=p11_summarize --dependency=afterany:$(IFS=:; echo "${ids[*]}") --export=ALL,STAGE=P10_kstudy200,OUT=reports/P11_kstudy200_results_table.md summarize_stage.sbatch
python3 - <<PY
import json, time, os
jobs = dict(l.split("=") for l in """$(for k in "${!JOB[@]}"; do echo "$k=${JOB[$k]}"; done)""".split())
rec = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": "P10_kstudy200 VCS K in {8,64} seeds 0-2 parallel (+P11 summarize)", "jobs": jobs,
       "commit": os.popen("git rev-parse HEAD").read().strip()}
p = "../reports/job_ids.json"; hist = json.load(open(p)) if os.path.exists(p) else []
hist.append(rec); json.dump(hist, open(p, "w"), indent=2); print(json.dumps(rec, indent=2))
PY
