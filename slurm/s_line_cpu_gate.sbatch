#!/usr/bin/env bash
#SBATCH --job-name=s_line_cpu_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/s_line_gate_%j.out
# S line (package v2 Spec §6-7, §12) CPU gate — no GPU, no data download.
#   1. CIFAR-100 link (flat local copy -> torchvision layout) + identity manifest (created once, verified afterwards)
#   2. test suites: reference core, integration, S-line v2
#   3. real-data smokes: frozen recipe first-step gradients vs the P35 run; rff_tanh (S-Kernel) 4v/2v (+resume); cs_kernel_native 4v/2v (+resume);
#      CIFAR-100 VCS with epoch eval + pilot evaluation (100-class kNN / probe); final_official_test must refuse cifar100
#   4. summary JSON; gate run dirs deleted (reports kept in reports/S_LINE_GATE_<job>/)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -o pipefail   # "cmd | tail || fail" must see cmd's status, not tail's
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/S_LINE_GATE_$J; REP=$REPO_ROOT/reports/S_LINE_GATE_$J
mkdir -p "$GATE" "$REP"
export OUTPUT_ROOT=$GATE     # ${OUTPUT_ROOT} in every config resolves to the gate directory
banner "S-line CPU gate -> $GATE ; reports -> $REP"
rc_all=0
step() { echo; echo "### [$(date -u +%T)] $1"; }
fail() { echo "FAIL: $1"; rc_all=1; }
keep() { mkdir -p "$REP/$1"; for f in first_step_gradients.json bandwidth_calibration.json run_manifest.json summary.json status.json; do [ -f "$GATE/$1/$f" ] && cp "$GATE/$1/$f" "$REP/$1/"; done; cp "$GATE/$1"/evaluations/evaluation_*.json "$REP/$1/" 2>/dev/null; return 0; }
smoke() { # name cfg steps epoch_steps [extra trainer args]
  local name=$1 cfg=$2 steps=$3 es=$4; shift 4
  echo "--- smoke $name  ($cfg)"; "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps "$steps" --smoke-epoch-steps "$es" "$@" 2>&1 | tail -4
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "smoke $name rc=$r"; return $r; }
stopped() { # name cfg: run 4 steps, stop cleanly after 2 (exit code of STOPPED_BUDGET is not 0), then resume to completion
  local name=$1 cfg=$2
  echo "--- stop/resume $name ($cfg)"; "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --stop-after-steps 2 2>&1 | tail -2
  local st; st=$("$PYTHON" -c "import json;print(json.load(open('$GATE/$name/status.json'))['status'])"); [ "$st" = "STOPPED_BUDGET" ] || fail "$name expected STOPPED_BUDGET, got $st"
  "$PYTHON" -m vcs_ssl.train --config "$cfg" --run-id "$name" --allow-cpu --smoke-steps 4 --smoke-epoch-steps 2 --no-epoch-eval --resume "$GATE/$name/checkpoints/last.pt" 2>&1 | tail -2
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "resume $name rc=$r"; return 0; }
evalp() { local name=$1 ck=$2; echo "--- evaluate pilot $name $ck"; "$PYTHON" -m vcs_ssl.evaluate --run-dir "$GATE/$name" --checkpoint "$ck" --protocol pilot --num-workers 4 2>&1 | tail -2
  local r=${PIPESTATUS[0]}; keep "$name"; [ $r -eq 0 ] || fail "evaluate $name rc=$r"; return $r; }

step "1. CIFAR-100: torchvision layout link + identity manifest"
C100=/home/infres/yinwang/CS_QMI/data/cifar100; mkdir -p "$C100"; ln -sfn /projects/EEG-foundation-model/yinghao/FMCA-AV/cifar100 "$C100/cifar-100-python"; ls -la "$C100/cifar-100-python/"
"$PYTHON" scripts/make_cifar100_manifest.py --root "$C100" --out "$MANIFEST_ROOT/cifar100_dev45k_val5k.json" --report "$REP/cifar100_manifest.json" 2>&1 | tail -30 || fail "cifar100 manifest"
[ -s "$MANIFEST_ROOT/cifar100_dev45k_val5k.json" ] || fail "cifar100 manifest missing"

step "2. test suites"
"$PYTHON" -m pytest -q tests/test_ssl_core.py -p no:cacheprovider > "$REP/pytest_ssl_core.txt" 2>&1; r=$?; tail -2 "$REP/pytest_ssl_core.txt"; [ $r -eq 0 ] || fail "test_ssl_core rc=$r"
"$PYTHON" -m pytest -q tests/test_integration.py -p no:cacheprovider > "$REP/pytest_integration.txt" 2>&1; r=$?; tail -2 "$REP/pytest_integration.txt"; [ $r -eq 0 ] || fail "test_integration rc=$r"
"$PYTHON" -m pytest -q tests/test_vcs_ssl_v2.py -p no:cacheprovider > "$REP/pytest_vcs_ssl_v2.txt" 2>&1; r=$?; tail -2 "$REP/pytest_vcs_ssl_v2.txt"; [ $r -eq 0 ] || fail "test_vcs_ssl_v2 rc=$r"

step "3a. frozen recipe (hpK 4v/B256 seed 0): first-step gradients on CPU vs the P35 GPU run"
smoke gate_recipe_hpK configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml 2 2 --no-epoch-eval
"$PYTHON" - "$GATE/gate_recipe_hpK/first_step_gradients.json" /home/infres/yinwang/CS_QMI/outputs/P35_vcs_a5_views4_800ep_seed0/first_step_gradients.json "$REP/first_step_gradient_compare.json" <<'PY' || fail "first-step gradient compare"
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); out = {}
for k in ("grad_norm_encoder", "grad_norm_projector", "grad_norm_critic"):
    out[k] = {"gate_cpu": a[k], "P35_gpu": b[k], "rel_diff": abs(a[k] - b[k]) / abs(b[k])}
out["max_rel_diff"] = max(v["rel_diff"] for v in out.values())
out["reading"] = "same config, seed, first batch and shifts; CPU vs cuDNN float32 arithmetic is the only difference (expected < 1e-2 relative)"
json.dump(out, open(sys.argv[3], "w"), indent=2); print(json.dumps(out, indent=1))
PY

step "3b. S-Kernel rff_tanh, 4 views, 1x dev config (m 1024, bw 1) + pilot evaluation (critic hold-out with the RFF critic)"
smoke gate_skernel_4v configs/cifar10_hpS_skernel_m1024_bw1_1x_seed0.yaml 6 3 --no-epoch-eval && evalp gate_skernel_4v last.pt
step "3c. S-Kernel rff_tanh, 2 views: stop + resume (sigma buffer carried by the checkpoint)"
stopped gate_skernel_2v configs/cifar10_hpS_gate_skernel_2v.yaml
step "3d. CS-K-native, 4 views, 1x dev config (bw 1) + pilot evaluation"
smoke gate_kcs_4v configs/cifar10_hpS_kcs_bw1_1x_seed0.yaml 6 3 --no-epoch-eval && evalp gate_kcs_4v last.pt
step "3e. CS-K-native, 2 views: stop + resume (kernel_sigma checked)"
stopped gate_kcs_2v configs/cifar10_hpS_gate_kcs_2v.yaml
step "3f. CIFAR-100 VCS (2x config, seed 0): smoke with epoch eval (100-class kNN) + pilot evaluation (100-class probe) + final-protocol refusal"
smoke gate_c100_vcs configs/cifar100_hpS_vcs_a5_views4_200ep_seed0.yaml 6 3 && evalp gate_c100_vcs last.pt
if "$PYTHON" -m vcs_ssl.evaluate --run-dir "$GATE/gate_c100_vcs" --checkpoint last.pt --protocol final_official_test --standin-selection --num-workers 2 > "$REP/final_refusal.txt" 2>&1; then
  fail "final_official_test did NOT refuse cifar100"
else
  echo "final_official_test refused cifar100 (expected): $(grep -m1 -o 'PermissionError.*' "$REP/final_refusal.txt" | cut -c1-200)"
fi

step "3g. S4 mechanism-record script: CPU smoke on existing checkpoints (VCS recipe, SimCLR, VCS strong) + table"
export OUTPUT_ROOT=/home/infres/yinwang/CS_QMI/outputs  # the script reads the real run directories
"$PYTHON" scripts/s4_mechanism_records.py --runs P35_vcs_a5_views4_800ep_seed0,P41_simclr_views4_800ep_seed0,P43_vcs_a5_views4_800ep_augstrong_seed0 --epochs 800 --smoke --cpu --workers 0 --aug own --out-dir "$REP/mech_smoke" 2>&1 | tail -4 || fail "mechanism smoke (own)"
"$PYTHON" scripts/s4_mechanism_records.py --runs P43_vcs_a5_views4_800ep_augstrong_seed0 --epochs 800 --smoke --cpu --workers 0 --aug standard --out-dir "$REP/mech_smoke" 2>&1 | tail -2 || fail "mechanism smoke (standard)"
"$PYTHON" scripts/s4_mechanism_records.py --table "$REP/mech_smoke" --out "$REP/mech_smoke.md" || fail "mechanism table"
export OUTPUT_ROOT=$GATE

step "4. summary"
"$PYTHON" - "$GATE" "$REP" "$MANIFEST_ROOT/cifar100_dev45k_val5k.json" <<'PY' || fail "summary"
import json, sys, torch
from pathlib import Path
gate, rep, man = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = {"gate_dir": str(gate), "runs": {}}
for rd in sorted(p for p in gate.iterdir() if p.is_dir()):
    r = {}
    for f in ("status.json", "bandwidth_calibration.json", "first_step_gradients.json"):
        if (rd / f).is_file():
            d = json.load(open(rd / f)); r[f[:-5]] = {k: d[k] for k in d if k in ("status", "completed_epoch", "optimizer_step", "kind", "sigma", "median_pair_distance", "bandwidth_multiple", "n_features", "trainable_params", "grad_norm_encoder", "grad_norm_projector", "grad_norm_critic", "check")}
    rm = json.load(open(rd / "run_manifest.json")) if (rd / "run_manifest.json").is_file() else {}
    r["method"] = rm.get("method"); r["dataset"] = rm.get("dataset"); r["n_classes"] = rm.get("n_classes"); r["n_views"] = rm.get("hparams", {}).get("n_views"); r["critic_impl"] = rm.get("critic_impl"); r["critic_params"] = rm.get("critic_params")
    cks = {}
    for ck in ("initial.pt", "last.pt"):
        p = rd / "checkpoints" / ck
        if p.is_file():
            c = torch.load(p, map_location="cpu", weights_only=False)
            cks[ck] = {"step": c.get("optimizer_step"), "kernel_sigma": c.get("kernel_sigma"), "rff_sigma": (float(c["critic_state"]["sigma"]) if c.get("critic_state") and "sigma" in c["critic_state"] else None)}
    r["checkpoints"] = cks
    ev = sorted((rd / "evaluations").glob("evaluation_*.json")) if (rd / "evaluations").is_dir() else []
    if ev:
        e = json.load(open(ev[-1])); r["evaluation"] = {k: e.get(k) for k in ("checkpoint", "linear_val_top1_pct", "knn_val_top1_pct", "heldout_J", "h_effective_rank", "n_classes", "dataset", "eval_seconds")}
    S["runs"][rd.name] = r
if man.is_file():
    m = json.load(open(man)); S["cifar100_manifest"] = {"sha256": m["manifest_sha256"], "n_fit": m["n_fit"], "n_selection": m["n_selection"], "n_classes": m.get("n_classes")}
if (rep / "first_step_gradient_compare.json").is_file():
    S["first_step_gradient_compare"] = json.load(open(rep / "first_step_gradient_compare.json"))
json.dump(S, open(rep / "GATE_SUMMARY.json", "w"), indent=2); print(json.dumps(S, indent=1))
PY

step "5. cleanup (gate run dirs removed; reports kept)"
rm -rf "$GATE"
echo "GATE_RC=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
