#!/usr/bin/env bash
#SBATCH --job-name=p149a1_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P149 add. 1 gate: unit tests unchanged, smoke of the new --site z path on the P149 smoke fixture, and the default (h) path on the same fixture
# reproducing the P149 gate smoke J values exactly (the site option must not change the h computation).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
G=reports/P149A1_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0; S=$OUTPUT_ROOT/P149_smoke
"$PYTHON" -m pytest -q tests/test_p149.py 2>&1 | tee $G/pytest_p149.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p149_twoview.py measure --fixture $S/P107_AP3_views4_800ep_seed1 --out $S/measure_P107_AP3_views4_800ep_seed1_z --smoke --site z 2>&1 | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p149_twoview.py measure --fixture $S/P107_AP3_views4_800ep_seed1 --out $S/measure_P107_AP3_views4_800ep_seed1_hcheck --smoke 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" - $S <<'PY' | tee $G/h_reproduction.txt || rc=1
import json, sys
a = json.load(open(sys.argv[1] + "/measure_P107_AP3_views4_800ep_seed1.json"))["settings"]; b = json.load(open(sys.argv[1] + "/measure_P107_AP3_views4_800ep_seed1_hcheck.json"))["settings"]
d = max(abs(a[s][k]["J_common"] - b[s][k]["J_common"]) for s in a for k in a[s]); print("h path vs P149 gate smoke: max |dJ| =", d); sys.exit(0 if d < 1e-6 else 1)
PY
"$PYTHON" -c "import json;d=json.load(open('$S/measure_P107_AP3_views4_800ep_seed1_z.json'));print('z site', d['protocol']['site'], 'dims ok' if d['protocol'] else '')" | tee -a $G/h_reproduction.txt
echo "{\"gate_rc\": $rc}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
