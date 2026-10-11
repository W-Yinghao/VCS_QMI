#!/usr/bin/env bash
#SBATCH --job-name=r4_vl1_prep
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-VL1 preparation (no training): $1 = "propose" (detector proposals for FIT images of the 3 datasets) or "measure" (memory / throughput of the
# proposal-aware objective, scripts/r4_vl1_measure.py).  Both are short; results feed the R4-VL1 protocol draft for the owner's budget decision.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home
cd /home/infres/yinwang/CS_QMI/ssl_pilot; echo "task=$1 host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
case "$1" in
  propose) $E/bin/python scripts/vl1_21_detector.py propose --roles FIT; rc=$?;;
  measure) $E/bin/python scripts/r4_vl1_measure.py --images 1 2 4 8 16; rc=$?; $E/bin/python scripts/r4_vl1_measure.py --images 1 2 4 8 16 --grad-ckpt || rc=1
           mv reports/R4/R4_VL1_measurement.json reports/R4/R4_VL1_measurement_gradckpt.json 2>/dev/null; $E/bin/python scripts/r4_vl1_measure.py --images 1 2 4 8 16 || rc=1;;
  *) echo "unknown task $1"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
