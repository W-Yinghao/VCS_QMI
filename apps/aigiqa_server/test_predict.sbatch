#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# AGIQA-3K alignment pilot: TEST predictions under the sealed release (package EXPERIMENT_LOCK section 4, aigiqa README section 5).
# Owner 2026-10-21 「开始 (Recommended)」. Label-free: reads TEST inputs/images only; prepared/.../sealed stays mode 000.
# Sequence: check-release; predict-aigiqa for each of the 18 run keys of the native freeze; native_imagereward + import_scores.
# Each predict-aigiqa re-runs the package's full check-release (re-hashes all 78 best.pt): that, not inference, sets the runtime.
# Rerunnable after an interruption: keys with an existing .meta.json are skipped; nothing here reads labels.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_aigiqa/apps/aigiqa_server/common.sh
REL=$AIG_PROTOCOL/aigiqa_alignment_release.json
FRZ=$AIG_PROTOCOL/aigiqa_alignment_release.native_freeze.json
INPUTS=$AIG_PREPARED/test_inputs.csv
T=$AIG_OUT/results/test
mkdir -p "$T/predictions" "$T/external" "$T/records"
aig_slurm_record "$T/records/predict_slurm_${SLURM_JOB_ID}.json" start
"$AIG_PY" "$AIG_SERVER/freeze_env.py" check --lock "$AIG_SERVER/ENV_LOCK.json" > "$T/records/env_check_${SLURM_JOB_ID}.json" 2>/dev/null || exit $?
"$AIG_PY" "$AIG_SERVER/gpu_health.py" "$T/records/gpu_health_${SLURM_JOB_ID}.json" > /dev/null || exit $?
ROOT=$("$AIG_PY" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['image_root'])" "$AIG_PREPARED/manifest.json")
KEYS=$("$AIG_PY" -I -c "import json,sys; print(' '.join(r['run_key'] for r in json.load(open(sys.argv[1]))['selected_runs']))" "$FRZ")
cd "$AIG_PKG"
"$AIG_PY" -m vcs_common.experiment check-release --release "$REL" > "$T/records/check_release_${SLURM_JOB_ID}.json" || exit $?
for KEY in $KEYS; do
  OUTP=$T/predictions/$KEY.csv
  if [ -f "$OUTP.meta.json" ]; then echo "skip $KEY (already predicted)"; continue; fi
  [ -e "$OUTP" ] && { echo "stop: $OUTP exists without .meta.json (interrupted write); remove it, then resubmit" >&2; exit 6; }
  echo "$(date -u +%FT%TZ) predict $KEY"
  "$AIG_PY" -m vcs_common.experiment predict-aigiqa --release "$REL" --run-key "$KEY" --inputs "$INPUTS" --image-root "$ROOT" --output "$OUTP" > /dev/null || exit $?
done
# Native frozen ImageReward-v1.0 baseline (README section 5): official preprocessing, then imported against the same freeze/inputs.
PROV=$T/external/imagereward_provenance.json
"$AIG_PY" -I - "$AIG_PKG/aigiqa/configs/external_imagereward_provenance.json" "$AIG_PROTOCOL/base_alignment.server.json" "$PROV" <<'PY' || exit $?
import json, os, sys
pkg, base, out = sys.argv[1:4]
p = json.load(open(pkg)); p["checkpoint_sha256"] = json.load(open(base))["model"]["checkpoint_sha256"]  # only the staged hash is filled
if not os.path.exists(out):
    json.dump(p, open(out, "w"), indent=1, ensure_ascii=False)
elif json.load(open(out)) != p:
    sys.exit("existing provenance differs")
PY
if [ ! -f "$T/external/imagereward_raw.csv" ]; then
  echo "$(date -u +%FT%TZ) native_imagereward"
  "$AIG_PY" -m aigiqa.cli native_imagereward --config "$AIG_PROTOCOL/base_alignment.server.json" --inputs "$INPUTS" --image-root "$ROOT" \
    --output-scores "$T/external/imagereward_raw.csv" > "$T/records/native_imagereward_${SLURM_JOB_ID}.json" || exit $?
fi
if [ ! -f "$T/predictions/native_imagereward.csv.meta.json" ]; then
  "$AIG_PY" -m aigiqa.cli import_scores --scores "$T/external/imagereward_raw.csv" --inputs "$INPUTS" --freeze "$FRZ" --target alignment \
    --provenance "$PROV" --output "$T/predictions/native_imagereward.csv" > /dev/null || exit $?
fi
# Completion marker read by test_evaluate.py: hashes of every prediction + meta, release/freeze/input hashes, the comparison pairs.
"$AIG_PY" -I - "$T" "$REL" "$FRZ" "$INPUTS" <<'PY'
import hashlib, json, os, sys, time
T, rel, frz, inputs = sys.argv[1:5]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
fz = json.load(open(frz))
keys = [r["run_key"] for r in fz["selected_runs"]] + ["native_imagereward"]
preds = {}
for k in keys:
    p = os.path.join(T, "predictions", k + ".csv")
    preds[k] = {"csv": p, "csv_sha256": sha(p), "meta_sha256": sha(p + ".meta.json")}
seeds = sorted(r["config"]["seed"] for r in fz["selected_runs"] if r["family"] == "vcs_exact_text")
done = {"status": "PASS", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "release": rel, "release_sha256_field": json.load(open(rel))["release_sha256"], "release_file_sha256": sha(rel),
        "native_freeze_sha256": sha(frz), "test_inputs_sha256": sha(inputs), "n_predictions": len(preds), "predictions": preds,
        "compare_pairs": [[f"vcs_exact_text__seed{s}", f"bce_text__seed{s}", s] for s in seeds],
        "labels_read": False, "claim_status": "PILOT_DIAGNOSTIC_ONLY"}
assert len(preds) == 19 and len(done["compare_pairs"]) == 3
json.dump(done, open(os.path.join(T, "PREDICTIONS_DONE.json"), "w"), indent=1)
print(json.dumps({k: done[k] for k in ("status", "n_predictions", "release_sha256_field")}))
PY
rc=$?
aig_slurm_record "$T/records/predict_slurm_${SLURM_JOB_ID}.json" end "$rc"
exit "$rc"
