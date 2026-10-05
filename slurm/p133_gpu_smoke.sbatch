#!/usr/bin/env bash
#SBATCH --job-name=p133_gpu_smoke
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p133_gpu_smoke_%j.out
# P133 GPU smoke: 60 steps of each P133 config and of the A-P3 / SimCLR parents on the same GPU (s/step, peak memory, J / loss, queue fill).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
J=${SLURM_JOB_ID:-local}; GATE=/home/infres/yinwang/CS_QMI/outputs/P133_GPU_SMOKE_$J; REP=$REPO_ROOT/reports/P133_GPU_SMOKE_$J
mkdir -p "$GATE" "$REP"; export OUTPUT_ROOT=$GATE
banner "P133 GPU smoke -> $REP"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for pair in "ap3:configs/cifar10_hpY_AP3_views4_800ep_seed0.yaml" "p133_vcs:configs/cifar10_hpMC_vcs_views4_800ep_seed0.yaml" \
            "simclr:configs/cifar10_hpN_simclr_views4_800ep_seed0.yaml" "p133_simclr:configs/cifar10_hpMC_simclr_views4_800ep_seed0.yaml"; do
  n=${pair%%:*}; c=${pair#*:}
  "$PYTHON" -m vcs_ssl.train --config "$c" --run-id "smoke_$n" --smoke-steps 60 --smoke-epoch-steps 60 --no-epoch-eval 2>&1 | tail -2 || rc=1
  mkdir -p "$REP/$n"; cp "$GATE/smoke_$n"/{summary.json,status.json,run_manifest.json} "$REP/$n/" 2>/dev/null; cp "$GATE/smoke_$n/logs/steps.jsonl" "$REP/$n/" 2>/dev/null
done
"$PYTHON" - "$REP" <<'PY'
import json, glob, os, sys, statistics as st
rep = sys.argv[1]; out = {}
for d in sorted(glob.glob(f"{rep}/*/")):
    n = os.path.basename(d.rstrip("/"))
    rows = [json.loads(l) for l in open(f"{d}/steps.jsonl")] if os.path.isfile(f"{d}/steps.jsonl") else []
    sm = json.load(open(f"{d}/summary.json")) if os.path.isfile(f"{d}/summary.json") else {}
    if not rows: out[n] = {"status": "no steps"}; continue
    out[n] = {"steps": len(rows), "step_seconds_median": st.median(r["step_seconds"] for r in rows[5:] or rows), "J_or_loss_first": rows[0].get("J_raw", rows[0].get("loss")),
              "J_or_loss_last": rows[-1].get("J_raw", rows[-1].get("loss")), "peak_allocated_mb": sm.get("peak_allocated_mb"),
              **{k: rows[-1][k] for k in rows[-1] if k.startswith("p133_") or k == "queue_fill"}}
json.dump(out, open(f"{rep}/summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
PY
rm -rf "$GATE"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
