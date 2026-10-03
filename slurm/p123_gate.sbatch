#!/usr/bin/env bash
#SBATCH --job-name=p123_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P123 CPU gate: tests (P123 + P115), smoke (C10 A-P3 + C100 SimCLR, epoch 800, B 16, 1 batch, all 11 counterfactual scorers),
# P115 reproduction (A-P3 C10 seed 0, epoch 800, batch 0 at full size, actual only) and a full-size timing of one checkpoint (A-P3 C10, 1 batch).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
J=${SLURM_JOB_ID:-local}; REP=$REPO_ROOT/reports/P123_GATE_$J; mkdir -p "$REP"; O=/home/infres/yinwang/CS_QMI/outputs/P123_GATE_$J
banner "P123 gate -> $REP"
rc=0
"$PYTHON" -m pytest -q tests/test_p123.py tests/test_p115_diag.py > "$REP/pytest.txt" 2>&1 || rc=1; tail -2 "$REP/pytest.txt"
"$PYTHON" scripts/p123_grad.py --runs P107_AP3_views4_800ep_seed0,P91_c100_simclr_views4_800ep_seed0,P35_vcs_a5_views4_800ep_seed0 --checkpoints 800 --smoke --cpu --out-dir $O/smoke > "$REP/smoke.txt" 2>&1 || rc=1; tail -4 "$REP/smoke.txt"
"$PYTHON" scripts/p123_grad.py --runs P107_AP3_views4_800ep_seed0,P41_simclr_views4_800ep_seed0 --repro-p115 --cpu --out-dir $O/repro > "$REP/repro.txt" 2>&1 || rc=1; grep -h "repro P115" "$REP/repro.txt"
"$PYTHON" scripts/p123_grad.py --runs P107_AP3_views4_800ep_seed0 --checkpoints 800 --grad-batches 1 --cpu --out-dir $O/timing > "$REP/timing.txt" 2>&1 || rc=1; grep -h "s  actual" "$REP/timing.txt"
cp -r $O/smoke $O/repro "$REP/" 2>/dev/null
echo "gate_rc=$rc"; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
