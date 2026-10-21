#!/bin/bash
# S4 (login node; standard-library hashing only, no training): register the KD CAL lock BEFORE any candidate
# trains, exactly as shared/docs/EXPERIMENT_LOCK_CN.md §3 (single grid holding all 15 families so that the
# documented matched group dkd_bce,dkd_vcs_exact,dkd_vcs_sampled,dkd_infonce lives in one lock).
# Grid: protocol/kd_cal_grid50 = 50 candidates (owner 2026-10-20: dkd_vid_pooled w100 removed before
# registration on G0 evidence; see DELTA_NOTE.md). Partitions RTX6000PRO,L40S (H100 added later by scontrol
# after its G0 passes).
# Prerequisites (checked below): strict FIT teacher post-check PASS; G0 overall PASS on every GPU type the
# CAL jobs may use (pass those G0 summaries as arguments); no lock yet; every run dir fresh.
# Usage: register_cal_lock.sh <G0_SUMMARY.json> [<G0_SUMMARY.json> ...]
set -euo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/common.sh
cd "$KD_ROOT"
[ "$#" -ge 1 ] || { echo "give the G0_SUMMARY.json of every GPU type to be used"; exit 1; }
LOCK=protocol/kd_cal_lock.json
test ! -e "$LOCK" || { echo "lock already exists: $LOCK"; exit 1; }
test -f runs/teacher_wrn40_2/teacher.pt -a -f runs/teacher_wrn40_2/teacher_provenance.json
ls ledger/teacher/teacher_wrn40_2__slurm*.post.json > /dev/null
"$KD_PY" - "$@" <<'EOF'
import glob, json, sys
posts = [json.load(open(p)) for p in sorted(glob.glob("ledger/teacher/teacher_wrn40_2__slurm*.post.json"))]
assert any(p.get("status") == "PASS" for p in posts), "no PASS teacher post-check"
for path in sys.argv[1:]:
    s = json.load(open(path))
    assert s.get("overall") == "PASS" and not s.get("selftest"), f"G0 not PASS: {path}"
    print("G0 PASS:", path, s.get("slurm_partition"))
EOF
"$KD_PY" -m vcs_common.experiment register --app kd --grid protocol/kd_cal_grid50/grid_manifest.json \
  --run-root runs/kd_cal --output "$LOCK" --scope pilot --match dkd_bce,dkd_vcs_exact,dkd_vcs_sampled,dkd_infonce
chmod 444 "$LOCK"
"$KD_PY" -m vcs_common.experiment check --lock "$LOCK"
"$KD_PY" -m vcs_common.experiment list-jobs --lock "$LOCK" > protocol/kd_cal_lock.list_jobs.json
"$KD_PY" "$KD_SERVER/tools/emit_cal_sbatch.py" --lock "$LOCK" \
  --precomputed "$KD_SERVER/records/PRECOMPUTED_CAL_JOB_IDS_50.json" --g0 "$@" \
  --partitions RTX6000PRO,L40S --nice 50 --prefix kd_cal --out "$KD_SERVER/lines/S5_cal_lines.txt"
sha256sum "$LOCK" protocol/kd_cal_grid50/grid_manifest.json protocol/cifar100_roles.json runs/teacher_wrn40_2/teacher.pt
echo "next: copy $LOCK and the list-jobs output into apps/kd_server/records/ and commit; then submit lines/S5_cal_lines.txt"
