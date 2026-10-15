#!/usr/bin/env bash
#SBATCH --job-name=v7c_auto_export
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7C-01 / 02 inputs (no new training, no GenAI label): the frozen prompt-only parser on the 606 EvalMuse CAL_SELECT / CAL_PROB / DEV prompts,
# then O and E_auto (parser elements) for the 9 formal M units on every CAL / DEV image (frozen x30_m_bench readout, EvalMuse archive reader).
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
BIG=/projects/EEG-foundation-model/yinghao/outputs_v7; A=/home/infres/yinwang/CS_QMI/outputs/V7_CLOSEOUT/auto; PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD)"
t0=$(date +%s); /projects/EEG-foundation-model/yinghao/envs/vl_baselines/bin/python -I scripts/v7/v7_parser.py run --prompts $A/evalmuse_cal_dev_prompts.jsonl --out $A/evalmuse_cal_dev_parser_cache.jsonl --batch 16; r0=$?; t1=$(date +%s)
echo "parser_rc=$r0 seconds=$((t1 - t0))"; [ $r0 -eq 0 ] || exit $r0
rc=0; for role in CAL_SELECT CAL_PROB DEV; do
  $PY scripts/v7x/v7c_auto_export.py export --units $(ls -d $BIG/units/V7-FIT-M*) --cands $A/cands_$role.jsonl --cache $A/evalmuse_cal_dev_parser_cache.jsonl --tag evalmuse_auto_$role || rc=1
done; echo "export_rc=$rc seconds=$(( $(date +%s) - t1 )) end=$(date -u +%FT%TZ)"; exit $rc
