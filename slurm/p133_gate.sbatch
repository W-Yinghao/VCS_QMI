#!/usr/bin/env bash
#SBATCH --job-name=p133_cpu_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p133_gate_%j.out
# P133 (MoCo-consistent momentum queue) CPU gate.  No GPU, no data download.
#   1. test suites: reference core, integration, v2, P100, P104, P114, P126, P133
#   2. config-hash invariance: every existing config resolves to the same config_hash with HEAD code and with the working-tree code
#   3. constant-online-map gate on real CIFAR-10 after 20 real P133 steps at peak lr (scripts/p133_gate_constant_map.py)
#   4. smokes: both P133 configs + their parents (A-P3, P41 SimCLR) 6 steps; stop/resume for both P133 configs
#   5. summary JSON (reports/P133_GATE_<job>/summary.json); gate run dirs deleted
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
set -o pipefail
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/P133_GATE_$J; REP=$REPO_ROOT/reports/P133_GATE_$J
REAL_OUT=$OUTPUT_ROOT
mkdir -p "$GATE" "$REP"
export OUTPUT_ROOT=$GATE
banner "P133 CPU gate -> $GATE ; reports -> $REP"
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
for t in test_ssl_core test_integration test_vcs_ssl_v2 test_vcs_ssl_p100 test_vcs_ssl_p104 test_vcs_ssl_p114 test_vcs_ssl_p126 test_vcs_ssl_p133; do
  OUTPUT_ROOT=$REAL_OUT "$PYTHON" -m pytest -q tests/$t.py -p no:cacheprovider > "$REP/pytest_$t.txt" 2>&1; r=$?; echo "$t: $(tail -1 "$REP/pytest_$t.txt")"; [ $r -eq 0 ] || fail "$t rc=$r"
done

step "2. config-hash invariance (HEAD code vs working-tree code)"
git -C "$REPO_ROOT" archive HEAD src reference | tar -x -C "$GATE" && mkdir -p "$GATE/headcode" && mv "$GATE/src" "$GATE/reference" "$GATE/headcode/"
cat > "$GATE/hashes.py" <<'PY'
import glob, json, os, sys
from vcs_ssl.config import load_config
env = {"DATA_ROOT": os.environ["DATA_ROOT"], "OUTPUT_ROOT": "/home/infres/yinwang/CS_QMI/outputs", "MANIFEST_ROOT": os.environ["MANIFEST_ROOT"]}
out = {}
for f in sorted(glob.glob("configs/*.yaml")):
    if "hpMC_" in f or "hpR50_" in f:
        continue  # new configs of this round need the new schema fields (P133 here; P130 = fork A)
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
json.dump({"configs": len(a), "loadable_at_HEAD": len(ok_head), "changed_by_working_tree_code": diff}, open(sys.argv[2], "w"), indent=2)
print(f"{len(a)} configs ({len(ok_head)} loadable at HEAD); config_hash changed by the working-tree code: {len(diff)} {diff[:10]}")
sys.exit(1 if diff else 0)
PY

step "3. constant-online-map gate (real CIFAR-10, 20 P133 steps at peak lr)"
"$PYTHON" scripts/p133_gate_constant_map.py --config configs/cifar10_hpMC_vcs_views4_800ep_seed0.yaml --steps 20 --run-dir "$GATE/constmap" \
    --out "$REP/constant_map.json" 2>&1 | tail -30 || fail "constant-map gate"

step "4. smokes (4 views, B 256) + stop/resume"
smoke gate_AP3 configs/cifar10_hpY_AP3_views4_800ep_seed0.yaml 6 3 --no-epoch-eval
smoke gate_simclr configs/cifar10_hpN_simclr_views4_800ep_seed0.yaml 6 3 --no-epoch-eval
for m in vcs simclr; do
  smoke gate_p133_$m configs/cifar10_hpMC_${m}_views4_800ep_seed0.yaml 6 3 --no-epoch-eval
  stopped gate_resume_p133_$m configs/cifar10_hpMC_${m}_views4_800ep_seed0.yaml
  "$PYTHON" -c "import json;print('$m', json.load(open('$GATE/gate_p133_$m/run_manifest.json')).get('p133_moco_consistent'))" || fail "p133 manifest $m"
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
    out[n] = {"status": s.get("status"), "steps": len(rows), "step_seconds_median": st.median(r["step_seconds"] for r in rows[1:] or rows),
              "J_first": rows[0].get("J_raw"), "J_last": rows[-1].get("J_raw"), "loss_last": rows[-1].get("loss"),
              **{k: rows[-1][k] for k in rows[-1] if k.startswith("p133_")}}
cm = json.load(open(f"{rep}/constant_map.json")) if os.path.isfile(f"{rep}/constant_map.json") else None
res = {"smokes": out, "constant_map": None if cm is None else {k: v for k, v in cm.items() if k not in ("candidates", "train_J")},
       "config_hash_invariance": json.load(open(f"{rep}/config_hash_invariance.json")) if os.path.isfile(f"{rep}/config_hash_invariance.json") else None}
json.dump(res, open(f"{rep}/summary.json", "w"), indent=1); print(json.dumps(res, indent=1)[:4000])
PY
rm -rf "$GATE"
echo "gate_rc=$rc_all"; echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
