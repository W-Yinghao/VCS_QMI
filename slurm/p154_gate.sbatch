#!/usr/bin/env bash
#SBATCH --job-name=p154_gate
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P154 gate: unit tests; real ImageNet-100 smoke (VCS 2 epochs x 60 steps, throughput / data wait), resume to epoch 3, JS / SimCLR loss paths on
# the GPU (1 epoch x 20 steps), and the full evaluation path (121.7 k + 5 k centre crops, linear probe, kNN) on the smoke checkpoint, timed.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P154_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0; export OUTPUT_ROOT=$OUTPUT_ROOT/P154_smoke; mkdir -p $OUTPUT_ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee $G/gpu.txt; echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK}" | tee -a $G/gpu.txt
"$PYTHON" -m pytest -q tests/test_p154.py 2>&1 | tee $G/pytest_p154.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_p139_imagenet_pipeline.py 2>&1 | tee $G/pytest_p139.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
t0=$(date +%s); "$PYTHON" scripts/p154_in100.py train --method vcs --run-id smoke_vcs --epochs 2 --max-steps-per-epoch 60 --workers 30 2>&1 | grep -vi warn | tee $G/smoke_vcs.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "[timing] vcs 2 x 60 steps: $(( $(date +%s) - t0 )) s" | tee -a $G/timing.txt
"$PYTHON" scripts/p154_in100.py train --method vcs --run-id smoke_vcs --epochs 3 --max-steps-per-epoch 60 --workers 30 2>&1 | grep -E "resume|epoch 3" | tee -a $G/smoke_vcs.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
for m in js simclr; do "$PYTHON" scripts/p154_in100.py train --method $m --run-id smoke_$m --epochs 1 --max-steps-per-epoch 20 --workers 30 2>&1 | grep -E "epoch 1/" | tee -a $G/smoke_$m.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1; done
t0=$(date +%s); "$PYTHON" scripts/p154_in100.py eval --run-id smoke_vcs --checkpoint epoch_3.pt --workers 30 2>&1 | grep -E "\[eval" | tee $G/smoke_eval.txt; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "[timing] eval: $(( $(date +%s) - t0 )) s" | tee -a $G/timing.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a $G/gpu.txt
echo "{\"gate_rc\": $rc}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
