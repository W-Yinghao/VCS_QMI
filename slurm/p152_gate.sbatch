#!/usr/bin/env bash
#SBATCH --job-name=p152_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P152 gate: estimator-core regression tests, a smoke of the recorder, and ONE full-size timing unit (I 6, seed 0, vcs + js) whose select-best
# value is compared with the P85 record (reproduction, descriptive).  Full-size output is kept (outputs/P152_e1) and reused by the frozen run.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
G=reports/P152_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
"$PYTHON" -m pytest -q tests/test_vcs_estim_v2.py tests/test_vcs_estim_stage0.py 2>&1 | tee $G/pytest_regression.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p152_e1_stability.py --I 6 --seed 0 --out $OUTPUT_ROOT/P152_smoke --smoke 2>&1 | tail -8; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
if [ -n "$P152_SMOKE_ONLY" ]; then echo "{\"gate_rc\": $rc, \"smoke_only\": true}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc; fi
t0=$(date +%s); "$PYTHON" scripts/p152_e1_stability.py --I 6 --seed 0 --kinds vcs,js --out $OUTPUT_ROOT/P152_e1 2>&1 | tail -4; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "[timing] full unit (2 kinds) seconds $(( $(date +%s) - t0 ))" | tee $G/timing.txt
"$PYTHON" - "$OUTPUT_ROOT/P152_e1/I6_s0.json" <<'PY' | tee $G/reproduction.txt
import json, sys; d = json.load(open(sys.argv[1]))
for k, v in d["kinds"].items(): print(k, v.get("reproduction_vs_P85"), "gnorm cv", v["gradient_norm"]["cv"], "nonfinite", v["fit"]["nonfinite_steps"])
PY
echo "{\"gate_rc\": $rc}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
