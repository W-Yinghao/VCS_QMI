#!/usr/bin/env bash
#SBATCH --job-name=p107_gpu_smoke
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p107_gpu_smoke_%j.out
# P107 GPU smoke (package v4 module A): 100 real-CIFAR optimizer steps per A-batch cell at the full recipe (4 views, B 256) plus P104 G2
# (the parent cell) on the same GPU as step-time reference.  Per variant: an uninterrupted 100-step run (pseudo-epochs of 50) and a stop-at-50 /
# resume-to-100 run; recorded: steady-state step time, peak memory, finite gradients, J at step 100, resumed-vs-uninterrupted state
# difference (NONDET=1 adds a second uninterrupted run as the GPU run-to-run baseline).  No epoch evaluation.  usage: sbatch --export=ALL,VARIANTS="G1 G3 U1 N1" slurm/p107_gpu_smoke.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
: "${VARIANTS:?}"
J=${SLURM_JOB_ID:-local}; SM=/home/infres/yinwang/CS_QMI/outputs/P107_SMOKE_$J; REP=$REPO_ROOT/reports/P107_GATE_SMOKE_$J
mkdir -p "$SM" "$REP"; export OUTPUT_ROOT=$SM
banner "P107 GPU smoke [$VARIANTS] -> $SM ; reports -> $REP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$REP/gpu.txt"
rc_all=0
fail() { echo "FAIL: $1"; rc_all=1; }
trn() { "$PYTHON" -m vcs_ssl.train --config "$1" --run-id "$2" --smoke-steps 100 --smoke-epoch-steps 50 --no-epoch-eval "${@:3}" 2>&1 | tail -3; return ${PIPESTATUS[0]}; }
# (the label "recipe" below is the P104 G2 reference cell, configs/cifar10_hpX_G2_views4_800ep_seed0.yaml)
cfg_of() { [ "$1" = recipe ] && echo configs/cifar10_hpX_G2_views4_800ep_seed0.yaml || echo configs/cifar10_hpY_$1_views4_800ep_seed0.yaml; }
for v in recipe $VARIANTS; do
  c=$(cfg_of "$v"); echo; echo "### [$(date -u +%T)] $v ($c)"
  trn "$c" "full_$v" || fail "full $v"
  [ "$v" = recipe ] && continue
  [ "${NONDET:-0}" = 1 ] && { trn "$c" "full2_$v" || fail "full2 $v"; }  # second uninterrupted run: GPU run-to-run baseline for the resume check
  trn "$c" "res_$v" --stop-after-steps 50 || true
  trn "$c" "res_$v" --resume "$SM/res_$v/checkpoints/last.pt" || fail "resume $v"
done
"$PYTHON" - "$SM" "$REP" $VARIANTS <<'PY' || fail "summary"
import json, math, os, sys, statistics as st
import torch
sm, rep, names = sys.argv[1], sys.argv[2], sys.argv[3:]
out = {}
for v in ["recipe"] + names:
    d = f"{sm}/full_{v}"
    if not os.path.isfile(f"{d}/summary.json"):
        out[v] = {"status": "MISSING"}; continue
    s = json.load(open(f"{d}/summary.json"))
    rows = [json.loads(l) for l in open(f"{d}/logs/steps.jsonl") if l.strip()]
    last = rows[-1]
    rec = {"status": s["status"], "steps": s["optimizer_step"], "step_s_steady": s["steady_state_step_seconds_mean"],
           "views_per_s_actual": s.get("steady_state_views_per_s_actual"), "peak_mem_mb": {k: s[k] for k in s if "peak" in k},
           "J_last_logged": last.get("J_raw"), "loss_last_logged": last.get("loss"), "last_logged_step": last.get("step"),
           "grads_finite": all(all(math.isfinite(x) for x in (r.get(k) for k in r if k.startswith("grad_norm")) if x is not None) for r in rows),
           "cos_scale": last.get("cos_scale"), "cos_bias": last.get("cos_bias"), "cost": s.get("cost"),
           "critic_impl": s.get("critic_impl"), "trainable_affine": s["hparams"].get("trainable_affine")}
    r = f"{sm}/res_{v}"
    if v != "recipe" and os.path.isfile(f"{r}/summary.json"):
        a = torch.load(f"{d}/checkpoints/last.pt", map_location="cpu", weights_only=False)
        b = torch.load(f"{r}/checkpoints/last.pt", map_location="cpu", weights_only=False)
        diff = 0.0
        for part in ("encoder_state", "projector_state", "critic_state"):
            for k in a[part]:
                if a[part][k].is_floating_point():
                    diff = max(diff, float((a[part][k] - b[part][k]).abs().max()))
        rows_r = [json.loads(l) for l in open(f"{r}/logs/steps.jsonl") if l.strip()]
        rec["resume"] = {"status": json.load(open(f"{r}/summary.json"))["status"], "max_abs_state_diff_vs_uninterrupted": diff,
                         "J_last_resumed": rows_r[-1].get("J_raw"), "noise_calls_equal": (bool(torch.equal(a["critic_state"]["noise_calls"], b["critic_state"]["noise_calls"]))
                                                                                       if "noise_calls" in a["critic_state"] else None),
                         "cost_equal": {k: a.get("p104_cost", {}).get(k) == b.get("p104_cost", {}).get(k) for k in ("positive_pairs", "negative_pairs", "critic_pair_evaluations", "encoder_updates", "critic_updates")}}
    if os.path.isfile(f"{sm}/full2_{v}/logs/steps.jsonl"):
        J1 = {x["step"]: x.get("J_raw") for x in rows}
        J2 = {x["step"]: x.get("J_raw") for x in (json.loads(l) for l in open(f"{sm}/full2_{v}/logs/steps.jsonl") if l.strip())}
        rec["rerun_J_absdiff"] = {s_: abs(J1[s_] - J2[s_]) for s_ in (0, 1, 49, 50, 99) if s_ in J1 and s_ in J2}
    if "resume" in rec:
        J1 = {x["step"]: x.get("J_raw") for x in rows}
        Jr = {x["step"]: x.get("J_raw") for x in rows_r}
        rec["resume"]["J_absdiff"] = {s_: abs(J1[s_] - Jr[s_]) for s_ in (0, 49, 50, 99) if s_ in J1 and s_ in Jr}
    out[v] = rec
p35 = json.load(open("/home/infres/yinwang/CS_QMI/outputs/P35_vcs_a5_views4_800ep_seed0/run_manifest.json"))["init_hashes"]
for v in out:
    if os.path.isfile(f"{sm}/full_{v}/run_manifest.json"):
        ih = json.load(open(f"{sm}/full_{v}/run_manifest.json"))["init_hashes"]
        out[v]["init_hash_equal_P35"] = {k: ih.get(k) == p35.get(k) for k in p35}
base = out.get("recipe", {}).get("step_s_steady")
for v, r in out.items():
    if base and r.get("step_s_steady"):
        r["step_time_ratio_vs_G2"] = r["step_s_steady"] / base
        r["projected_800ep_hours"] = r["step_s_steady"] * 175 * 800 / 3600  # 45000 fit images / B 256 -> 175 steps per epoch (drop_last)
json.dump(out, open(f"{rep}/summary.json", "w"), indent=2)
for v, r in out.items():
    print(v, json.dumps({k: r.get(k) for k in ("status", "step_s_steady", "step_time_ratio_vs_G2", "projected_800ep_hours", "peak_mem_mb", "J_last_logged", "grads_finite", "resume")}))
PY
for d in "$SM"/*/; do n=$(basename "$d"); mkdir -p "$REP/$n"; cp "$d"/summary.json "$d"/run_manifest.json "$d"/first_step_gradients.json "$d"/logs/steps.jsonl "$REP/$n/" 2>/dev/null; done
rm -rf "$SM"
echo "smoke_rc=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
