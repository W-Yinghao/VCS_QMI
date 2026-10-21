#!/bin/bash
# S8 TEST unseal (owner approved 2026-10-21 「开始」): the EXPERIMENT_LOCK_CN §3 steps on the sealed final release.
#   check-release -> evaluate-kd once per selected_job_id -> kd compare -> kd profile (one run per family, seed 31415).
# A job that already has any TEST output (test_metrics.json or test_predictions.csv) is never evaluated again; a
# resubmission therefore only evaluates the jobs still without TEST output. GPU needed: the frozen run configs say
# device=cuda and kd.runner refuses a CPU fallback. Pilot/diagnostic only; no selection after TEST.
# Outputs: $KD_ROOT/reports/test_eval/ (TEST_SUMMARY.json, kd_descriptive_comparison.json, profile/, logs/);
# per-run test_metrics.json / test_predictions.csv are written by the package into each final run directory.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/common.sh
export KD_BATCH_SCRIPT="$0"
cd "$KD_ROOT" || exit 2
REL="$KD_ROOT/protocol/kd_final_release.json"
OUT="$KD_ROOT/reports/test_eval"; JOB=${SLURM_JOB_ID:-manual}
mkdir -p "$OUT/logs" "$OUT/profile"
kd_ledger start --kind test_eval --tag final_release
trap 'echo "SIGTERM at $(date -u +%FT%TZ)"; kd_ledger end --kind test_eval --tag final_release --rc 143 --stage sigterm; exit 143' TERM
if ! "$KD_PY" "$KD_SERVER/tools/env_check.py" --cuda --out "$OUT/logs/env_check_slurm$JOB.json" > /dev/null; then
  kd_ledger end --kind test_eval --tag final_release --rc 3 --stage env_check; exit 3
fi
if ! "$KD_PY" -m vcs_common.experiment check-release --release "$REL" > "$OUT/logs/check_release_slurm$JOB.json"; then
  kd_ledger end --kind test_eval --tag final_release --rc 4 --stage check_release; exit 4
fi
# selected job id <TAB> run dir <TAB> family <TAB> seed, from the release and its lock (metadata only)
mapfile -t ROWS < <("$KD_PY" - "$REL" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); jobs = {j["job_id"]: j for j in json.load(open(r["lock_path"]))["jobs"]}
for i in r["selected_job_ids"]:
    print(f"{i}\t{jobs[i]['run_dir']}\t{jobs[i]['family']}\t{jobs[i]['seed']}")
EOF
)
[ "${#ROWS[@]}" -eq 45 ] || { echo "expected 45 selected jobs, got ${#ROWS[@]}"; kd_ledger end --kind test_eval --tag final_release --rc 5 --stage rows; exit 5; }
STATUS="$OUT/logs/status_slurm$JOB.tsv"; : > "$STATUS"
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r ID DIR FAM SEED <<< "$row"
  if [ -e "$DIR/test_metrics.json" ] || [ -e "$DIR/test_predictions.csv" ]; then
    st=$([ -e "$DIR/test_metrics.json" ] && echo already_evaluated || echo PARTIAL_TEST_OUTPUT_needs_owner_decision)
    echo -e "$ID\t$st" >> "$STATUS"; echo "refuse $ID: $st"; continue
  fi
  "$KD_PY" -m vcs_common.experiment evaluate-kd --release "$REL" --job-id "$ID" > "$OUT/logs/$ID.json" 2>&1
  rc=$?; echo -e "$ID\t$([ $rc -eq 0 ] && echo evaluated || echo "failed_rc$rc")" >> "$STATUS"; echo "evaluate-kd $ID rc=$rc"
done
"$KD_PY" - "$REL" "$OUT" "$STATUS" <<'EOF'
import hashlib, json, subprocess, sys
from pathlib import Path
rel, out, status = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
r = json.load(open(rel)); jobs = {j["job_id"]: j for j in json.load(open(r["lock_path"]))["jobs"]}
dirs = [jobs[i]["run_dir"] for i in r["selected_job_ids"]]
done = [d for d in dirs if (Path(d) / "test_metrics.json").is_file()]
summary = {"schema": "kd_server_test_summary_v1", "release": rel, "release_sha256": r["release_sha256"],
           "claim_status": r["claim_status"], "label": "pilot/diagnostic; no selection after TEST",
           "this_job_status": dict(l.rstrip("\n").split("\t") for l in open(status) if l.strip()),
           "n_selected": len(dirs), "n_with_test_metrics": len(done)}
if len(done) == len(dirs):
    cmp_out = out / "kd_descriptive_comparison.json"
    if not cmp_out.exists():
        subprocess.run([sys.executable, "-m", "kd", "compare", "--runs", *dirs, "--out", str(cmp_out)], check=True, stdout=subprocess.DEVNULL)
    c = json.load(open(cmp_out))
    summary["families"] = {f: {"n_seeds": v["n_seeds"], "mean_top1": v["mean_top1"], "sd_top1": v["sd_top1"],
                               "mean_nll": sum(s["nll"] for s in v["seeds"].values()) / len(v["seeds"])} for f, v in sorted(c["methods"].items())}
    order = ["dkd_vcs_exact", "dkd_bce", "dkd_vcs_sampled", "dkd_infonce", "dkd"]  # matched group + DKD anchor
    pairs = {}
    for p in c["paired_comparisons"]:
        if p["a"] in order and p["b"] in order:
            a, b, sign = (p["a"], p["b"], 1) if order.index(p["a"]) < order.index(p["b"]) else (p["b"], p["a"], -1)
            pairs[f"{a} - {b}"] = {"paired_seeds": p["paired_seeds"], "delta_top1_per_seed": [sign * d for d in p["delta_top1_per_seed"]],
                                   "mean_delta_top1": sign * p["mean_delta_top1"], "sd_delta_top1": p["sd_delta_top1"]}
    summary["matched_group_paired_top1"] = pairs
    summary["primary_contrasts"] = ["dkd_vcs_exact - dkd_bce", "dkd_vcs_exact - dkd"]
    summary["units"] = c["units"]
    for f in sorted({jobs[i]["family"] for i in r["selected_job_ids"]}):
        run = next(jobs[i]["run_dir"] for i in r["selected_job_ids"] if jobs[i]["family"] == f and jobs[i]["seed"] == 31415)
        prof = out / "profile" / f"{f}_seed31415.json"
        if not prof.exists():
            subprocess.run([sys.executable, "-m", "kd", "profile", "--run", run, "--out", str(prof)], check=True, stdout=subprocess.DEVNULL)
    summary["profiles"] = sorted(str(p) for p in (out / "profile").glob("*.json"))
summary["test_metrics_sha256"] = {Path(d).name: hashlib.sha256((Path(d) / "test_metrics.json").read_bytes()).hexdigest() for d in done}
(out / "TEST_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: summary[k] for k in ("n_selected", "n_with_test_metrics")}))
sys.exit(0 if len(done) == len(dirs) else 1)
EOF
rc=$?
kd_ledger end --kind test_eval --tag final_release --rc "$rc" --stage done
exit "$rc"
