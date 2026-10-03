#!/usr/bin/env bash
#SBATCH --job-name=p122_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P122 CPU gate: unit tests + reduced-size smokes (CIFAR-10 two encoders x two laws; CIFAR-100 one encoder) on CPU.  Outputs to reports/P122_GATE_<job>/.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=$REPO_ROOT/reports/P122_GATE_${SLURM_JOB_ID:-local}; mkdir -p "$G"; rc=0
banner "P122 CPU gate -> $G"
"$PYTHON" -m pytest -q tests/test_p122.py > "$G/pytest_test_p122.txt" 2>&1 || rc=1; tail -2 "$G/pytest_test_p122.txt"
"$PYTHON" -m pytest -q tests/test_p109.py > "$G/pytest_test_p109.txt" 2>&1 || rc=1; tail -1 "$G/pytest_test_p109.txt"
"$PYTHON" scripts/p122_evidence.py --dataset cifar10 --runs P107_AP3_views4_800ep_seed0,P41_simclr_views4_800ep_seed0 --laws standard,strong \
   --out "$G/smoke_c10" --cache /home/infres/yinwang/CS_QMI/outputs/P122_GATE_cache --smoke --cpu || rc=1
"$PYTHON" scripts/p122_evidence.py --dataset cifar100 --runs P91_c100_vcs_a5_views4_800ep_seed0 --laws standard \
   --out "$G/smoke_c100" --cache /home/infres/yinwang/CS_QMI/outputs/P122_GATE_cache --smoke --cpu || rc=1
echo "gate_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
