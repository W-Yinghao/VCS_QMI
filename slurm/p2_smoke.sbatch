#!/usr/bin/env bash
#SBATCH --job-name=vcs_p2_smoke
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p2_smoke_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P2 smoke: 3 methods x 100 real-data steps (own 100-step schedule) + GPU resume check (VCS)"
rc=0
for m in vcs simclr vicreg; do
  run_fg "$PYTHON" -m vcs_ssl.train --config configs/cifar10_pilot_$m.yaml --smoke-steps 100 --smoke-epoch-steps 50 --run-id "smoke_${m}_${SLURM_JOB_ID}" || rc=$?
done
# GPU-level resume check: stop VCS at 50, resume to 100, compare with the continuous smoke run (report observed diff; exactness not promised on CUDA)
run_fg "$PYTHON" -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml --smoke-steps 100 --smoke-epoch-steps 50 --stop-after-steps 50 --run-id "smoke_vcs_interrupted_${SLURM_JOB_ID}" --no-epoch-eval
run_fg "$PYTHON" -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml --smoke-steps 100 --smoke-epoch-steps 50 --resume "$OUTPUT_ROOT/smoke_vcs_interrupted_${SLURM_JOB_ID}/checkpoints/last.pt" --no-epoch-eval || rc=$?
"$PYTHON" scripts/p2_gate.py --output-root "$OUTPUT_ROOT" --job "$SLURM_JOB_ID" --report reports/P2_smoke_gate.md || rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
