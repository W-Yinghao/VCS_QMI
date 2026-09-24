#!/usr/bin/env bash
#SBATCH --job-name=p12_babysit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/babysit_%x_%j.out
# Queue feeder: submits units (run_id config) from UNITS_FILE one at a time whenever the user's submitted-job count is below CAP,
# then submits the stage summarize job depending on all of them. Never resubmits a run_id that already has a run dir with a
# COMPLETED/RUNNING status (skip-if-done); records every job id in reports/job_ids.json.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${UNITS_FILE:?}"; : "${STAGE:?}"; : "${OUT:?}"; CAP=${CAP:-28}; FINAL_CKPT=${FINAL_CKPT:-epoch_200.pt}
banner "babysitter for ${STAGE}: units=${UNITS_FILE} cap=${CAP}"
ids=(); submitted=()
while read -r run_id cfg; do
  [ -z "$run_id" ] && continue
  st=$("$PYTHON" -c "import json;print(json.load(open('$OUTPUT_ROOT/$run_id/status.json'))['status'])" 2>/dev/null || echo NONE)
  if [ "$st" = "COMPLETED" ]; then echo "[skip] $run_id already COMPLETED"; continue; fi
  while [ "$(squeue -h -u "$USER" | wc -l)" -ge "$CAP" ]; do sleep 120; done
  id=$(sbatch --parsable --job-name="${run_id#P12_vcs_}" --export=ALL,CFG="$cfg",RUN_ID="$run_id",FINAL_CKPT="$FINAL_CKPT" "$REPO_ROOT/slurm/run_unit.sbatch" | cut -d';' -f1)
  echo "[submit] $(date -u +%FT%TZ) $run_id -> $id ($cfg) queue=$(squeue -h -u "$USER" | wc -l)"
  ids+=("$id"); submitted+=("$run_id=$id")
  sleep 5
done < "$UNITS_FILE"
if [ ${#ids[@]} -gt 0 ]; then
  dep=$(IFS=:; echo "${ids[*]}")
  sid=$(sbatch --parsable --job-name="${STAGE}_summarize" --dependency=afterany:${dep} --export=ALL,STAGE="$STAGE",OUT="$OUT" "$REPO_ROOT/slurm/summarize_stage.sbatch" | cut -d';' -f1)
  echo "[submit] summarize -> $sid (afterany:${dep})"; submitted+=("summarize=$sid")
fi
"$PYTHON" - "$STAGE" "${submitted[@]}" <<'PY'
import json, sys, time, os
stage, pairs = sys.argv[1], sys.argv[2:]
p = "reports/job_ids.json"; hist = json.load(open(p)) if os.path.exists(p) else []
hist.append({"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage + " (babysitter)",
             "jobs": dict(x.split("=") for x in pairs), "commit": os.popen("git rev-parse HEAD").read().strip()})
json.dump(hist, open(p, "w"), indent=2); print("recorded", len(pairs), "jobs")
PY
echo "end=$(date -u +%FT%TZ)"
