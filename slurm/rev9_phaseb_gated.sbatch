#!/usr/bin/env bash
#SBATCH --job-name=rev9_phaseb_g
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV9 phase B after execution note 1 (RNG-state device fix in scripts/rev9_phaseb.py).  Gate first: the I0 guard on one unit
# (vcs_lr0.0003_s0) into a scratch directory must meet the frozen guard (estimator max abs diff 0, gate within 1e-3); only then the
# frozen 'phaseb' mode of slurm/rev9.sbatch runs (all 21 phase-B runs into outputs/REV9/selectors).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; SM=/home/infres/yinwang/CS_QMI/outputs/REV9_SMOKE_resume
banner "REV9 gate: I0 guard smoke (vcs_lr0.0003_s0)"
"$PYTHON" scripts/rev9_phaseb.py run --tag vcs_lr0.0003_s0 --interventions I0 --d1 "$D1" --out "$SM" || { echo "GATE FAIL: smoke run"; exit 4; }
"$PYTHON" - "$SM/vcs_lr0.0003_s0_I0.json" <<'PY' || { echo "GATE FAIL: guard"; exit 5; }
import json, sys
r = json.load(open(sys.argv[1]))["rev9"]; est, gate = r["max_abs_diff_estimator_vs_frozen"], r["max_abs_diff_gate_vs_frozen"]
print(f"REV9 GATE: estimator diff {est:.3e} (must be 0), gate diff {gate:.3e} (must be <= 1e-3)", flush=True)
raise SystemExit(0 if est == 0 and gate <= 1e-3 else 1)
PY
echo "REV9 GATE PASS"
bash slurm/rev9.sbatch phaseb; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
