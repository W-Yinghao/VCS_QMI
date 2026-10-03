#!/usr/bin/env bash
#SBATCH --job-name=p114_cpu_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p114_gate_%j.out
# P114 / P115 (v5 NEXT-A-JS-AP3, NEXT-A-AUG; owner 2026-10-03) CPU gate. No GPU and no data download.
#   1. test suites: reference core, integration, S-line v2, P95, P100, P104, P107, P114; package v3 support/validate_core.py
#   2. frozen recipe (hpK 4v/B256 seed 0): first-step gradients and init hashes on CPU vs the P35 GPU run
#   3. config-hash invariance: every existing config resolves to the same config_hash under HEAD code and under the P114 code
#   4. the 3 P114 + 4 P115 configs + the A-P3 parent: 6-step smokes; stop/resume for all seven
#   5. summary JSON (reports/P114_GATE_<job>/summary.json); gate run dirs deleted
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -o pipefail
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/P114_GATE_$J; REP=$REPO_ROOT/reports/P114_GATE_$J
REAL_OUT=$OUTPUT_ROOT
mkdir -p "$GATE" "$REP"
unzip -q -o /home/infres/yinwang/CS_QMI/VCS_SSL_Performance_Research_and_Server_v3.zip "VCS_SSL_Performance_v3/support/*" -d "$GATE/pkg"; PKG=$GATE/pkg/VCS_SSL_Performance_v3
export OUTPUT_ROOT=$GATE
banner "P114 CPU gate -> $GATE ; reports -> $REP"
rc_all=0
step() { echo; echo "### [$(date -u +%T)] $1"; }
fail() { echo "FAIL: $1"; rc_all=1; }
keep() { mkdir -p "$REP/$1"; for f in first_step_gradients.json run_manifest.json summary.json status.json; do [ -f "$GATE/$1/$f" ] && cp "$GATE/$1/$f" "$REP/$1/"; done
         cp "$GATE/$1"/logs/steps.jsonl "$REP/$1/" 2>/dev/null; return 0; }
smoke() { local name=$1 cfg=$2 steps=$3 es=$4; shift 4
  echo "--- smoke $name  ($cfg)"; "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps "$steps" --smoke-epoch-steps "$es" "$@" 2>&1 | tail -4
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "smoke $name rc=$r"; return $r; }
stopped() { local name=$1 cfg=$2
  echo "--- stop/resume $name ($cfg)"; "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --stop-after-steps 2 2>&1 | tail -2
  local st; st=$("$PYTHON" -c "import json;print(json.load(open('$GATE/$name/status.json'))['status'])"); [ "$st" = "STOPPED_BUDGET" ] || fail "$name expected STOPPED_BUDGET, got $st"
  "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --resume "$GATE/$name/checkpoints/last.pt" 2>&1 | tail -2
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "resume $name rc=$r"; return 0; }

step "1. test suites"
for t in test_ssl_core test_integration test_vcs_ssl_v2 test_vcs_ssl_p95 test_vcs_ssl_p100 test_vcs_ssl_p104 test_vcs_ssl_p107 test_vcs_ssl_p114; do
  "$PYTHON" -m pytest -q tests/$t.py -p no:cacheprovider > "$REP/pytest_$t.txt" 2>&1; r=$?; echo "$t: $(tail -1 "$REP/pytest_$t.txt")"; [ $r -eq 0 ] || fail "$t rc=$r"
done
if [ -f "$PKG/support/validate_core.py" ]; then
  (cd "$PKG/support" && "$PYTHON" validate_core.py) > "$REP/validate_core.txt" 2>&1; r=$?; echo "validate_core: $(tail -1 "$REP/validate_core.txt")"; [ $r -eq 0 ] || fail "validate_core rc=$r"
else
  echo "validate_core.py not found at $PKG (skipped)"
fi

step "2. frozen recipe first-step gradients + init hashes vs P35"
smoke gate_recipe_hpK configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml 6 3 --no-epoch-eval
"$PYTHON" - "$GATE/gate_recipe_hpK" "$REAL_OUT/P35_vcs_a5_views4_800ep_seed0" "$REP/first_step_gradient_compare.json" <<'PY' || fail "first-step gradient / init hash compare"
import json, sys
g, p = sys.argv[1], sys.argv[2]
a, b = json.load(open(f"{g}/first_step_gradients.json")), json.load(open(f"{p}/first_step_gradients.json")); out = {}
for k in ("grad_norm_encoder", "grad_norm_projector", "grad_norm_critic"):
    out[k] = {"gate_cpu": a[k], "P35_gpu": b[k], "rel_diff": abs(a[k] - b[k]) / abs(b[k])}
out["max_rel_diff"] = max(v["rel_diff"] for v in out.values())
ia, ib = json.load(open(f"{g}/run_manifest.json"))["init_hashes"], json.load(open(f"{p}/run_manifest.json"))["init_hashes"]
out["init_hashes_equal"] = ia == ib  # informational on CPU nodes: CPU normal_ kernels differ by ISA; the binding check is on the GPU smoke
out["init_hashes_equal_note"] = "CPU-node init hashes can differ from GPU-node runs (vectorised RNG path); compared again in p104_gpu_smoke"
json.dump(out, open(sys.argv[3], "w"), indent=2); print(json.dumps(out, indent=1))
sys.exit(0 if out["max_rel_diff"] < 1e-2 else 1)
PY

step "3. config-hash invariance: every existing config resolves to the same config_hash with HEAD code and with the P114 code"
git -C "$REPO_ROOT" archive HEAD src reference | tar -x -C "$GATE" && mkdir -p "$GATE/headcode" && mv "$GATE/src" "$GATE/reference" "$GATE/headcode/"
cat > "$GATE/hashes.py" <<'PY'
import glob, json, os, sys
from vcs_ssl.config import load_config
env = {"DATA_ROOT": os.environ["DATA_ROOT"], "OUTPUT_ROOT": "/home/infres/yinwang/CS_QMI/outputs", "MANIFEST_ROOT": os.environ["MANIFEST_ROOT"]}
out = {}
for f in sorted(glob.glob("configs/*.yaml")):
    if "hpJS_" in f:
        continue  # the new P114 configs need the P114 schema field
    try:
        out[f] = load_config(f, env=env)["_meta"]["config_hash"]
    except Exception as e:  # noqa: BLE001
        out[f] = f"ERROR {type(e).__name__}"
json.dump(out, open(sys.argv[1], "w"), indent=1)
PY
PYTHONPATH="$GATE/headcode/src:$GATE/headcode" "$PYTHON" "$GATE/hashes.py" "$GATE/hash_head.json" && "$PYTHON" "$GATE/hashes.py" "$GATE/hash_now.json"
"$PYTHON" - "$GATE" "$REP/config_hash_invariance.json" <<'PY' || fail "config-hash invariance"
import json, sys
a, b = json.load(open(f"{sys.argv[1]}/hash_head.json")), json.load(open(f"{sys.argv[1]}/hash_now.json"))
diff = [f for f in a if a[f] != b.get(f)]
ok_head = [f for f in a if not a[f].startswith("ERROR")]
json.dump({"configs": len(a), "loadable_at_HEAD": len(ok_head), "changed_by_P114_code": diff}, open(sys.argv[2], "w"), indent=2)
print(f"{len(a)} configs ({len(ok_head)} loadable at HEAD); config_hash changed by the P114 code: {len(diff)} {diff[:10]}")
sys.exit(1 if diff else 0)
PY

step "4. P114 + P115 smokes (4 views, B 256) + the A-P3 parent as step-time reference; stop/resume for all seven"
smoke gate_AP3 configs/cifar10_hpY_AP3_views4_800ep_seed0.yaml 6 3 --no-epoch-eval
for c in configs/cifar10_hpJS_AP3_views4_800ep_seed{0,1,2}.yaml configs/cifar10_hpAF_{AP3,simclr}_{croponly,jitteronly}_views4_800ep_seed0.yaml; do
  v=$(basename "$c" .yaml | sed -E 's/^cifar10_hp(JS|AF)_//; s/_views4_800ep//')
  smoke gate_$v "$c" 6 3 --no-epoch-eval
  stopped gate_resume_$v "$c"
done

step "5. summary"
"$PYTHON" - "$REP" <<'PY' || fail "summary"
import json, sys, glob, os, statistics as st
rep = sys.argv[1]; out = {}
for d in sorted(glob.glob(f"{rep}/gate_*")):
    n = os.path.basename(d); f = f"{d}/steps.jsonl"
    if not os.path.isfile(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    s = json.load(open(f"{d}/status.json")) if os.path.isfile(f"{d}/status.json") else {}
    sm = json.load(open(f"{d}/summary.json")) if os.path.isfile(f"{d}/summary.json") else {}
    rec = {"status": s.get("status"), "steps": len(rows), "step_seconds_median": st.median(r["step_seconds"] for r in rows[1:] or rows),
           "J_first": rows[0].get("J_raw"), "J_last": rows[-1].get("J_raw"), "loss_last": rows[-1].get("loss"), "cost": sm.get("cost")}
    for k in ("cos_scale", "cos_bias", "js_loss"):
        if k in rows[-1]:
            rec[k] = rows[-1][k]
    out[n] = rec
base = out.get("gate_AP3", {}).get("step_seconds_median")
for n, r in out.items():
    if base:
        r["step_time_ratio_vs_AP3"] = r["step_seconds_median"] / base
json.dump(out, open(f"{rep}/summary.json", "w"), indent=2)
for n, r in out.items():
    print(f"{n:24s} {r['status']:14s} steps {r['steps']} step_s {r['step_seconds_median']:.2f} ratio_vs_AP3 {r.get('step_time_ratio_vs_AP3', float('nan')):.2f} J {r['J_first']} -> {r['J_last']}")
PY
rm -rf "$GATE"
echo "gate_rc=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
