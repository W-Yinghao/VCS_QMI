#!/usr/bin/env bash
#SBATCH --job-name=rev10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV10 (reports/REV9/REV10_LR_FACTORIAL_PROTOCOL_FROZEN_20261015.yaml).  Positional arguments only.
#   phaseb              I0 (seed 0) / I1 / I2 on the 6 units, frozen rev9_phaseb.py; the I0 guard of each cell is checked and written to guard.json
#                       (a failed guard is recorded, the readout of that cell is then stopped by the protocol); RTX PRO 6000 (the units' hardware)
#   eval <bench>        frozen D1-30 runner: the 6 frozen units -> REV10_D1/d1_30/pred, then their I1 / I2 units -> REV10/d1_30/pred (SparseVLM env, H100 / L40S)
#   score               frozen rev9_score.py, SparseVLM env, tags vcs_lr0.0001 logistic_lr0.0003 (CPU)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; export PYTHONNOUSERSITE=1
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; DN=/home/infres/yinwang/CS_QMI/outputs/REV10_D1; RV=/home/infres/yinwang/CS_QMI/outputs/REV10
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat scripts/rev9_phaseb.py scripts/rev9_score.py scripts/r7_d1_30_run.py | sha256sum | cut -c1-16)"
what="$1"; shift; rc=0
case "$what" in
  phaseb) for t in vcs_lr0.0001 logistic_lr0.0003; do for s in 0 1 2; do iv="I1 I2"; [ $s -eq 0 ] && iv="I0 I1 I2"
            "$PYTHON" scripts/rev9_phaseb.py run --tag ${t}_s$s --interventions $iv --d1 $DN --out $RV/selectors || rc=$?; done; done
          "$PYTHON" - $RV/selectors <<'PY' || rc=$?
import json, sys
from pathlib import Path
out = {}
for t in ("vcs_lr0.0001", "logistic_lr0.0003"):
    r = json.loads((Path(sys.argv[1]) / f"{t}_s0_I0.json").read_text())["rev9"]; est, gate = r["max_abs_diff_estimator_vs_frozen"], r["max_abs_diff_gate_vs_frozen"]
    out[t] = {"estimator": est, "gate": gate, "passed": est == 0 and gate <= 1e-3}; print(f"REV10 GUARD {t}: estimator {est:.3e} gate {gate:.3e} passed {out[t]['passed']}")
(Path(sys.argv[1]) / "guard.json").write_text(json.dumps(out, indent=1))
PY
          ;;
  eval) export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf; b="$1"
        $E/bin/python scripts/r7_d1_30_run.py --sets-dir $D1/d1_30/sets --bench "$b" --configs-file $RV/configs/configs_frozen_$b.txt --shard 0/1 \
          --out $DN/d1_30/pred --selectors-dir $DN/selectors --pca $D1/cache/pca.pt || rc=$?
        $E/bin/python scripts/r7_d1_30_run.py --sets-dir $D1/d1_30/sets --bench "$b" --configs-file $RV/configs/configs_iv_$b.txt --shard 0/1 \
          --out $RV/d1_30/pred --selectors-dir $RV/selectors --pca $D1/cache/pca.pt || rc=$?;;
  score) mkdir -p $RV/results; $E/bin/python scripts/rev9_score.py --d1 $DN --rv $RV --tags vcs_lr0.0001 logistic_lr0.0003 --out $RV/results/rev10_results.json; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
