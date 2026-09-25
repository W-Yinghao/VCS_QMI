#!/usr/bin/env bash
# Direct parallel submission of units (run_id config per line) + a stage summarize job depending on all of them.
# Usage: ./submit_units.sh <units_file> <STAGE> <OUT_table.md> [FINAL_CKPT=epoch_200.pt]   (use babysit_submit.sbatch when the 30-job cap is near)
set -euo pipefail
cd "$(dirname "$0")"
UNITS=$1; STAGE=$2; OUT=$3; FINAL=${4:-epoch_200.pt}
ids=(); pairs=()
while read -r run_id cfg; do
  [ -z "$run_id" ] && continue
  id=$(sbatch --parsable --job-name="${run_id#P*_vcs_}" --export=ALL,CFG="$cfg",RUN_ID="$run_id",FINAL_CKPT="$FINAL" run_unit.sbatch | cut -d';' -f1)
  echo "$run_id -> $id ($cfg)"; ids+=("$id"); pairs+=("$run_id=$id")
done < "$UNITS"
dep=$(IFS=:; echo "${ids[*]}")
sid=$(sbatch --parsable --job-name="${STAGE}_summarize" --dependency=afterany:${dep} --export=ALL,STAGE="$STAGE",OUT="$OUT" summarize_stage.sbatch | cut -d';' -f1)
echo "summarize -> $sid"; pairs+=("summarize=$sid")
python3 - "$STAGE" "${pairs[@]}" <<'PY'
import json, sys, time, os
stage, pairs = sys.argv[1], sys.argv[2:]
p = "../reports/job_ids.json"; hist = json.load(open(p)) if os.path.exists(p) else []
hist.append({"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage, "jobs": dict(x.split("=") for x in pairs),
             "commit": os.popen("git rev-parse HEAD").read().strip()})
json.dump(hist, open(p, "w"), indent=2); print("recorded", len(pairs), "jobs")
PY
