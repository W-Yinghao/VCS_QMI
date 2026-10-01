#!/usr/bin/env bash
#SBATCH --job-name=p109_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P109 gate: unit tests + end-to-end smokes of X1, I1 (features + audit + effects) and I2 on CPU.  Smoke outputs under reports/P109_GATE_<job>/.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P109_GATE_${SLURM_JOB_ID}; mkdir -p $G; rc=0
banner "P109 gate -> $G"
"$PYTHON" -m pytest -q tests/test_p109.py 2>&1 | tee $G/pytest.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_p102.py 2>&1 | tail -3 | tee $G/pytest_neighbours.txt   # P104 suite belongs to the main session
T0=$(date +%s); "$PYTHON" scripts/p109_x1.py --runs P104_G2_views4_800ep_seed0,P41_simclr_views4_800ep_seed0 --out $G/x1 --smoke --workers 4 2>&1 | tee $G/x1.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1; echo "x1 smoke seconds $(( $(date +%s) - T0 ))"
T0=$(date +%s); "$PYTHON" scripts/p109_i1_features.py --runs P104_G2_views4_800ep_seed0 --out $G/i1_features --smoke --workers 4 2>&1 | tee $G/i1_features.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1; echo "i1 features smoke seconds $(( $(date +%s) - T0 ))"
"$PYTHON" scripts/p109_i1_audit.py --features $G/i1_features --run P104_G2_views4_800ep_seed0 --family colour --cells 'planted:0.2:200;null_label_only:0:200' --seed 999 --out $G/i1_audit --effects --smoke 2>&1 | tee $G/i1_audit.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p109_i2_pilot.py --out $G/i2_pilot --cells vcs4v800:colour:0,vcs4v800:colour:0.2 --smoke 2>&1 | tee $G/i2.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "gate_rc=$rc"; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
