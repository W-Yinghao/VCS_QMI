#!/usr/bin/env bash
#SBATCH --job-name=p130_gpu_smoke
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p130_gpu_smoke_%j.out
# P130 GPU smoke (CIFAR-100 ResNet-50, v6 §8.3): 150 real-CIFAR-100 optimizer steps (pseudo-epochs of 75) per R50 config at the full recipe
# (4 views, B 256) plus the ResNet-18 A-P3 C100 parent on the same GPU as step-time reference; a stop-at-75 / resume-to-150 run for A-P3 R50;
# the full evaluation protocol (frozen-h linear + kNN on 2048-d h) on the A-P3 R50 smoke checkpoint.  Recorded: steady-state step time, peak
# memory, finite loss, projected s / epoch and hours / 800 epochs on this GPU.  No epoch evaluation during training.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
J=${SLURM_JOB_ID:-local}; SM=/home/infres/yinwang/CS_QMI/outputs/P130_SMOKE_$J; REP=$REPO_ROOT/reports/P130_GATE_SMOKE_$J
mkdir -p "$SM" "$REP"; export OUTPUT_ROOT=$SM
banner "P130 GPU smoke -> $SM ; reports -> $REP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$REP/gpu.txt"
rc_all=0; fail() { echo "FAIL: $1"; rc_all=1; }
trn() { "$PYTHON" -m vcs_ssl.train --config "$1" --run-id "$2" --smoke-steps 150 --smoke-epoch-steps 75 --no-epoch-eval "${@:3}" 2>&1 | tail -3; return ${PIPESTATUS[0]}; }
declare -A CFG=([r18_AP3]=configs/cifar100_hpY_AP3_c100_views4_800ep_seed0.yaml [r50_AP3]=configs/cifar100_hpR50_AP3_views4_800ep_seed0.yaml
                [r50_JS]=configs/cifar100_hpR50_JS_AP3_views4_800ep_seed0.yaml [r50_simclr]=configs/cifar100_hpR50_simclr_views4_800ep_seed0.yaml)
for v in r18_AP3 r50_AP3 r50_JS r50_simclr; do
  echo; echo "### [$(date -u +%T)] $v (${CFG[$v]})"
  trn "${CFG[$v]}" "full_$v" || fail "full $v"
done
echo; echo "### [$(date -u +%T)] resume check r50_AP3"
trn "${CFG[r50_AP3]}" res_r50_AP3 --stop-after-steps 75 || true
trn "${CFG[r50_AP3]}" res_r50_AP3 --resume "$SM/res_r50_AP3/checkpoints/last.pt" || fail "resume r50_AP3"
echo; echo "### [$(date -u +%T)] evaluation protocol on full_r50_AP3/epoch_002.pt"
"$PYTHON" -m vcs_ssl.evaluate --run-dir "$SM/full_r50_AP3" --checkpoint epoch_002.pt --protocol pilot 2>&1 | tail -4 || fail "evaluate r50"
"$PYTHON" - "$SM" "$REP" <<'PY' || fail "summary"
import json, math, sys, statistics as st, glob
import torch
sm, rep = sys.argv[1], sys.argv[2]
out = {"gpu": open(f"{rep}/gpu.txt").read().strip()}
for v in ["r18_AP3", "r50_AP3", "r50_JS", "r50_simclr"]:
    d = f"{sm}/full_{v}"
    steps = [json.loads(l) for l in open(f"{d}/logs/steps.jsonl")]
    t = [s["interval_mean_step_seconds"] for s in steps if isinstance(s.get("interval_mean_step_seconds"), (int, float)) and s.get("step", 0) >= 50]
    summ = json.load(open(f"{d}/summary.json"))
    loss = [s.get("loss") for s in steps if s.get("loss") is not None]
    med = st.median(t) if t else float("nan"); spe = 45000 // 256
    out[v] = {"median_step_s": med, "projected_s_per_epoch": med * spe, "projected_h_800ep": med * spe * 800 / 3600,
              "peak_allocated_mb": summ.get("peak_allocated_mb"), "peak_reserved_mb": summ.get("peak_reserved_mb"),
              "finite_loss": all(math.isfinite(x) for x in loss), "status": json.load(open(f"{d}/status.json"))["status"]}
a = torch.load(f"{sm}/full_r50_AP3/checkpoints/last.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{sm}/res_r50_AP3/checkpoints/last.pt", map_location="cpu", weights_only=False)
ea, eb = a.get("encoder") or a.get("encoder_state"), b.get("encoder") or b.get("encoder_state")
out["resume_max_abs_diff_encoder"] = max(float((ea[k].float() - eb[k].float()).abs().max()) for k in ea) if ea and eb else None
ev = glob.glob(f"{sm}/full_r50_AP3/evaluations/evaluation_epoch_002.json")
out["r50_eval"] = {k: v for k, v in json.load(open(ev[0])).items() if "top1" in k} if ev else None
json.dump(out, open(f"{rep}/smoke_summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
PY
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
