#!/usr/bin/env bash
#SBATCH --job-name=p95_cpu_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p95_gate_%j.out
# P95 (package v1 improvements inside full SSL) CPU gate — no GPU, no data download.
#   1. test suites: reference core, integration, S-line v2, P95
#   2. frozen recipe (hpK 4v/B256 seed 0): first-step gradients on CPU vs the P35 GPU run (the P95 code leaves the recipe path untouched)
#   3. 4-view smokes of the five variants from the stage-A configs (6 steps, pseudo-epochs of 3), per-step time and J recorded;
#      stop/resume for dictionary and for refresh (gate copy of the R50 config with refresh_every_epochs = 1, refresh_batches = 2);
#      pilot evaluation of the dictionary smoke (critic hold-out with the dictionary critic in eval mode)
#   4. summary JSON (reports/P95_GATE_<job>/summary.json); gate run dirs deleted
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -o pipefail
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/P95_GATE_$J; REP=$REPO_ROOT/reports/P95_GATE_$J
mkdir -p "$GATE" "$REP"
export OUTPUT_ROOT=$GATE
banner "P95 CPU gate -> $GATE ; reports -> $REP"
rc_all=0
step() { echo; echo "### [$(date -u +%T)] $1"; }
fail() { echo "FAIL: $1"; rc_all=1; }
keep() { mkdir -p "$REP/$1"; for f in first_step_gradients.json run_manifest.json summary.json status.json; do [ -f "$GATE/$1/$f" ] && cp "$GATE/$1/$f" "$REP/$1/"; done
         cp "$GATE/$1"/logs/steps.jsonl "$GATE/$1"/logs/critic_refresh.jsonl "$REP/$1/" 2>/dev/null; cp "$GATE/$1"/evaluations/evaluation_*.json "$REP/$1/" 2>/dev/null; return 0; }
smoke() { local name=$1 cfg=$2 steps=$3 es=$4; shift 4
  echo "--- smoke $name  ($cfg)"; "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps "$steps" --smoke-epoch-steps "$es" "$@" 2>&1 | tail -4
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "smoke $name rc=$r"; return $r; }
stopped() { local name=$1 cfg=$2
  echo "--- stop/resume $name ($cfg)"; "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --stop-after-steps 2 2>&1 | tail -2
  local st; st=$("$PYTHON" -c "import json;print(json.load(open('$GATE/$name/status.json'))['status'])"); [ "$st" = "STOPPED_BUDGET" ] || fail "$name expected STOPPED_BUDGET, got $st"
  "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --resume "$GATE/$name/checkpoints/last.pt" 2>&1 | tail -2
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "resume $name rc=$r"; return 0; }

step "1. test suites"
for t in test_ssl_core test_integration test_vcs_ssl_v2 test_vcs_ssl_p95; do
  "$PYTHON" -m pytest -q tests/$t.py -p no:cacheprovider > "$REP/pytest_$t.txt" 2>&1; r=$?; echo "$t: $(tail -1 "$REP/pytest_$t.txt")"; [ $r -eq 0 ] || fail "$t rc=$r"
done

step "2. frozen recipe first-step gradients vs P35"
smoke gate_recipe_hpK configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml 6 3 --no-epoch-eval
"$PYTHON" - "$GATE/gate_recipe_hpK/first_step_gradients.json" /home/infres/yinwang/CS_QMI/outputs/P35_vcs_a5_views4_800ep_seed0/first_step_gradients.json "$REP/first_step_gradient_compare.json" <<'PY' || fail "first-step gradient compare"
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); out = {}
for k in ("grad_norm_encoder", "grad_norm_projector", "grad_norm_critic"):
    out[k] = {"gate_cpu": a[k], "P35_gpu": b[k], "rel_diff": abs(a[k] - b[k]) / abs(b[k])}
out["max_rel_diff"] = max(v["rel_diff"] for v in out.values())
json.dump(out, open(sys.argv[3], "w"), indent=2); print(json.dumps(out, indent=1))
sys.exit(0 if out["max_rel_diff"] < 1e-2 else 1)
PY

step "3. variant smokes (stage-A configs, 4 views)"
for c in residual_lam0.25 residual_lam0.5 dictionary noise_tau0.1 noise_tau0.3 refresh_R50 js_matched; do
  smoke gate_$c configs/cifar10_hpV_${c}_views4_800ep_seed0.yaml 6 3 --no-epoch-eval
done
"$PYTHON" - "$GATE" <<'PY'
import sys, yaml
g = sys.argv[1]
c = yaml.safe_load(open("configs/cifar10_hpV_refresh_R50_views4_800ep_seed0.yaml"))
c["model"]["critic"]["refresh_every_epochs"] = 1; c["model"]["critic"]["refresh_batches"] = 2
open(f"{g}/gate_refresh_R1.yaml", "w").write(yaml.safe_dump(c, sort_keys=False))
PY
smoke gate_refresh_R1 "$GATE/gate_refresh_R1.yaml" 6 3 --no-epoch-eval
stopped gate_resume_dictionary configs/cifar10_hpV_dictionary_views4_800ep_seed0.yaml
stopped gate_resume_refresh_R1 "$GATE/gate_refresh_R1.yaml"
echo "--- evaluate pilot gate_dictionary"; "$PYTHON" -m vcs_ssl.evaluate --run-dir "$GATE/gate_dictionary" --checkpoint last.pt --protocol pilot --num-workers 4 2>&1 | tail -2 || fail "evaluate gate_dictionary"
keep gate_dictionary

step "4. summary"
"$PYTHON" - "$REP" <<'PY' || fail "summary"
import json, sys, glob, os, statistics as st
rep = sys.argv[1]; out = {}
for d in sorted(glob.glob(f"{rep}/gate_*")):
    n = os.path.basename(d); f = f"{d}/steps.jsonl"
    if not os.path.isfile(f):
        continue
    rows = [json.loads(l) for l in open(f) if l.strip()]
    s = json.load(open(f"{d}/status.json")) if os.path.isfile(f"{d}/status.json") else {}
    rec = {"status": s.get("status"), "steps": len(rows), "step_seconds_median": st.median(r["step_seconds"] for r in rows[1:] or rows),
           "J_first": rows[0].get("J_raw"), "J_last": rows[-1].get("J_raw"), "loss_last": rows[-1].get("loss")}
    for k in ("js_loss", "dict_w_cosine", "dict_w_mlp", "dict_w_bilinear", "dict_disagreement_D", "res_cos_part_mean", "res_mlp_part_mean",
              "noise_coord_sd_emp", "noise_total_rms_emp", "noise_total_rms", "cos_scale", "cos_bias"):
        if k in rows[-1]:
            rec[k] = rows[-1][k]
    if os.path.isfile(f"{d}/critic_refresh.jsonl"):
        rec["refresh"] = [json.loads(l) for l in open(f"{d}/critic_refresh.jsonl") if l.strip()]
    ev = glob.glob(f"{d}/evaluation_*.json")
    if ev:
        e = json.load(open(ev[0])); rec["pilot_eval"] = {"linear": e.get("linear_val_top1_pct"), "knn": e.get("knn_val_top1_pct")}
    out[n] = rec
base = out.get("gate_recipe_hpK", {}).get("step_seconds_median")
for n, r in out.items():
    if base:
        r["step_time_ratio_vs_recipe"] = r["step_seconds_median"] / base
json.dump(out, open(f"{rep}/summary.json", "w"), indent=2)
for n, r in out.items():
    print(f"{n:28s} {r['status']:10s} steps {r['steps']} step_s {r['step_seconds_median']:.2f} ratio {r.get('step_time_ratio_vs_recipe', float('nan')):.2f} J {r['J_first']} -> {r['J_last']}")
PY
rm -rf "$GATE"
echo "gate_rc=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
