#!/usr/bin/env bash
#SBATCH --job-name=p138_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p138_gpu_%j.out
# P138 GPU job: (1) 60-step training smoke of each stage-2 config next to its parent on the same GPU (s/step, peak memory, pair counts);
# (2) stage-1 read-only check (smoke, then the full 4 models x epochs 100 / 800).  Read-only for every existing run directory.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/P138_GPU_SMOKE_$J; REP=$REPO_ROOT/reports/P138_GPU_SMOKE_$J
mkdir -p "$GATE" "$REP"; banner "P138 GPU job -> $REP"; nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader | tee "$REP/gpu.txt"
rc=0
OUTPUT_ROOT_SAVED=$OUTPUT_ROOT; export OUTPUT_ROOT=$GATE
for pair in "ap3_c10:configs/cifar10_hpY_AP3_views4_800ep_seed0.yaml" "p138_vcs_c10:configs/cifar10_hpPK16_vcs_c10_seed0.yaml" \
            "js_c10:configs/cifar10_hpJS_AP3_views4_800ep_seed0.yaml" "p138_js_c10:configs/cifar10_hpPK16_js_c10_seed0.yaml" \
            "p138_vcs_c100:configs/cifar100_hpPK16_vcs_c100_seed0.yaml"; do
  n=${pair%%:*}; c=${pair#*:}
  "$PYTHON" -m vcs_ssl.train --config "$c" --run-id "smoke_$n" --smoke-steps 60 --smoke-epoch-steps 60 --no-epoch-eval 2>&1 | tail -2 || rc=1
  mkdir -p "$REP/$n"; cp "$GATE/smoke_$n"/{summary.json,status.json,run_manifest.json} "$REP/$n/" 2>/dev/null; cp "$GATE/smoke_$n/logs/steps.jsonl" "$REP/$n/" 2>/dev/null
done
export OUTPUT_ROOT=$OUTPUT_ROOT_SAVED
"$PYTHON" - "$REP" <<'PY'
import json, glob, os, sys, statistics as st
rep = sys.argv[1]; out = {}
for d in sorted(glob.glob(f"{rep}/*/")):
    n = os.path.basename(d.rstrip("/"))
    rows = [json.loads(l) for l in open(f"{d}/steps.jsonl")] if os.path.isfile(f"{d}/steps.jsonl") else []
    sm = json.load(open(f"{d}/summary.json")) if os.path.isfile(f"{d}/summary.json") else {}
    if not rows: out[n] = {"status": "no steps"}; continue
    last = rows[-1]
    out[n] = {"steps": len(rows), "step_seconds_median": st.median(r["step_seconds"] for r in (rows[5:] or rows)),
              "loss_first": rows[0].get("loss"), "loss_last": last.get("loss"), "J_last": last.get("J_raw"), "peak_allocated_mb": sm.get("peak_allocated_mb"),
              "n_pos": last.get("n_pos"), "n_neg": last.get("n_neg"), "critic_pair_evals": last.get("critic_pair_evals"), "shift_example": last.get("shift")}
json.dump(out, open(f"{rep}/summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
PY
rm -rf "$GATE"
RUNS=P107_AP3_views4_800ep_seed0,P107_AP3_c100_views4_800ep_seed0,P114_JSAP3_views4_800ep_seed0,P120_JS_AP3_c100_views4_800ep_seed0
"$PYTHON" scripts/p138_pair_readonly.py --runs P107_AP3_views4_800ep_seed0 --checkpoints 800 --out-dir "$REP/readonly_smoke" --smoke || rc=1
"$PYTHON" scripts/p138_pair_readonly.py --runs "$RUNS" --checkpoints 100,800 --out-dir /home/infres/yinwang/CS_QMI/outputs/P138_readonly || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
