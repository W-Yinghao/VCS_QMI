#!/usr/bin/env bash
#SBATCH --job-name=rev9f_direct_h100
#SBATCH --partition=H100
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV9 execution note 3 (labelled secondary): the frozen D1-FULL direct_lr0.0001_s0 unit was trained on an H100 NVL; its REV9-FULL I0 rerun on an
# RTX PRO 6000 failed the guard (gate diff 2.11).  Rerun I0 / I1 / I2 on H100 into REV9_FULL_H100; the I0 guard decides whether this secondary
# reading of the direct objective is usable.  The primary REV9-FULL readout of the direct objective stays stopped (frozen rule).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; export PYTHONNOUSERSITE=1
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$PYTHON" scripts/rev9_phaseb.py run --tag direct_lr0.0001_s0 --interventions I0 I1 I2 --d1 /home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL --out /home/infres/yinwang/CS_QMI/outputs/REV9_FULL_H100/selectors || exit 4
"$PYTHON" - <<'PY' || { echo "H100 GUARD FAIL"; exit 5; }
import json
r = json.load(open("/home/infres/yinwang/CS_QMI/outputs/REV9_FULL_H100/selectors/direct_lr0.0001_s0_I0.json"))["rev9"]; e, g = r["max_abs_diff_estimator_vs_frozen"], r["max_abs_diff_gate_vs_frozen"]
print(f"H100 GUARD: estimator diff {e:.3e}, gate diff {g:.3e}", flush=True); raise SystemExit(0 if e == 0 and g <= 1e-3 else 1)
PY
echo "H100 GUARD PASS end=$(date -u +%FT%TZ)"
