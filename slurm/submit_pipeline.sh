#!/usr/bin/env bash
# Submit P0 -> P1 -> P2 -> P3(vcs -> simclr -> vicreg, sequential: one GPU job at a time) -> P4 with afterok dependencies.
# A failing gate leaves the downstream jobs pending with DependencyNeverSatisfied; cancel them with scancel.
set -euo pipefail
cd "$(dirname "$0")"
LOGS=/home/infres/yinwang/CS_QMI/slurm_logs; mkdir -p "$LOGS" ../reports
START_FROM=${1:-p0}   # p0|p1|p2|p3|p4
dep=""
declare -A JOB
submit() { local name=$1; shift; local id; id=$(sbatch --parsable "$@" | cut -d';' -f1); JOB[$name]=$id; echo "$name -> job $id ($*)"; }
if [[ $START_FROM == p0 ]]; then submit p0 p0_preflight.sbatch; dep="--dependency=afterok:${JOB[p0]}"; fi
if [[ $START_FROM =~ ^p[01]$ ]]; then submit p1 $dep p1_tests.sbatch; dep="--dependency=afterok:${JOB[p1]}"; fi
if [[ $START_FROM =~ ^p[012]$ ]]; then submit p2 $dep p2_smoke.sbatch; dep="--dependency=afterok:${JOB[p2]}"; fi
if [[ $START_FROM =~ ^p[0123]$ ]]; then
  submit p3_vcs $dep --export=ALL,METHOD=vcs p3_pilot.sbatch
  submit p3_simclr --dependency=afterany:${JOB[p3_vcs]} --export=ALL,METHOD=simclr p3_pilot.sbatch
  submit p3_vicreg --dependency=afterany:${JOB[p3_simclr]} --export=ALL,METHOD=vicreg p3_pilot.sbatch
  dep="--dependency=afterany:${JOB[p3_vicreg]}"
fi
submit p4 $dep p4_summarize.sbatch
python3 - "$@" <<PY
import json, time, os
jobs = dict(l.split("=") for l in """$(for k in "${!JOB[@]}"; do echo "$k=${JOB[$k]}"; done)""".split())
rec = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "start_from": "$START_FROM", "jobs": jobs,
       "commit": os.popen("git rev-parse HEAD").read().strip()}
p = "../reports/job_ids.json"
hist = json.load(open(p)) if os.path.exists(p) else []
hist.append(rec); json.dump(hist, open(p, "w"), indent=2); print(json.dumps(rec, indent=2))
PY
