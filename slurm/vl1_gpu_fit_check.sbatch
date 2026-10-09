#!/usr/bin/env bash
#SBATCH --job-name=vl_gpu_fit_check
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Equivalence gate for GPU execution of the frozen fitter: re-fit CLIP VCS N 1000 seed 0 (VL1-10) on the GPU and compare with the CPU result.
# Pass: |dJ_CAL| < 0.001 and |dTop1_DEV| < 0.003 for the estimator-selected checkpoint (and the selected lr equal).  Also times the fit.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; export VL_DEVICE=cuda; T=$OUTPUT_ROOT/VL_gpu_fit_check; rm -rf $T
"$PYTHON" -m pytest -q tests/test_vl1.py 2>&1 | tail -2
"$PYTHON" scripts/vl1_10_fit.py --route vcs --ns 1000 --seeds 0 --out $T || exit 1
"$PYTHON" - <<'PY'
import json
a = json.load(open("/home/infres/yinwang/CS_QMI/outputs/VL1_10/vcs_N1000_s0.json")); b = json.load(open("/home/infres/yinwang/CS_QMI/outputs/VL_gpu_fit_check/vcs_N1000_s0.json"))
ea, eb = a["estimator_selected"], b["estimator_selected"]
dJ = eb["cal"]["J_common"] - ea["cal"]["J_common"]; dT = eb["dev"]["top1_image_macro"] - ea["dev"]["top1_image_macro"]
fs = (sum(c["fit_seconds"] for c in a["candidates"]), sum(c["fit_seconds"] for c in b["candidates"]))
ok = abs(dJ) < 0.001 and abs(dT) < 0.003 and ea["lr"] == eb["lr"]
print(f"CPU lr {ea['lr']} upd {ea['update']} J {ea['cal']['J_common']:.5f} top1 {ea['dev']['top1_image_macro']:.4f} | GPU lr {eb['lr']} upd {eb['update']} J {eb['cal']['J_common']:.5f} top1 {eb['dev']['top1_image_macro']:.4f}")
print(f"dJ {dJ:+.5f} dTop1 {dT:+.4f} fit seconds CPU {fs[0]:.0f} GPU {fs[1]:.0f} -> {'PASS' if ok else 'FAIL'}")
json.dump({"dJ": dJ, "dTop1": dT, "cpu_seconds": fs[0], "gpu_seconds": fs[1], "pass": ok}, open("/home/infres/yinwang/CS_QMI/ssl_pilot/reports/VL1/vl_gpu_fit_check.json", "w"), indent=1)
PY
echo "exit=$? end=$(date -u +%FT%TZ)"
