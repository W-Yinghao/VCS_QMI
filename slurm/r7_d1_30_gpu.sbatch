#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_30_gpu
#SBATCH --partition=H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-30 GPU units (LLaVA-1.5 host in the SparseVLM env: torch 2.1.2 + flash-attn 2.3.3, no Blackwell kernels -> H100 / L40S only).
# Resumable per (configuration, item); a rerun continues.  Timing must run on the reference GPU: sbatch --partition=H100 --nodelist=nodesumo01.
# First arg: run <bench> <i/n> | sparsevlm <bench> <i/n> | timing <bench> | devrisk | smoke.  Positional arguments only (no --export).
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R=$D1/d1_30; SM=$D1/d1_30_smoke
P=reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
case "$what" in
  run) $E/bin/python scripts/r7_d1_30_run.py --sets-dir "$R/sets" --bench "$1" --configs-file "$R/configs/configs_$1.txt" --shard "$2" --out "$R/pred"; rc=$?;;
  sparsevlm) $E/bin/python scripts/r7_d1_30_sparsevlm.py --sets-dir "$R/sets" --bench "$1" --shard "$2" --out "$R/pred"; rc=$?;;
  timing) $E/bin/python scripts/r7_d1_30_timing.py --sets-dir "$R/sets" --bench "$1" --configs-file "$R/configs/timing_$1.txt" --out "$R/timing/$1"; rc=$?
          [ $rc -eq 0 ] && { $E/bin/python scripts/r7_d1_30_sparsevlm.py --sets-dir "$R/sets" --bench "$1" --out "$R/timing" --timing 64; rc=$?; };;
  devrisk) $E/bin/python scripts/r7_d1_30_devrisk.py --out "$R/devrisk" --protocol "$P" --lr-choice "$D1/selectors/lr_choice.json"; rc=$?;;
  smoke) $E/bin/python scripts/r7_d1_30_smoke.py --out "$SM"; rc=$?
         [ $rc -eq 0 ] && { $E/bin/python scripts/r7_d1_30_sparsevlm.py --sets-dir "$SM/sets64" --bench gqa_fixture textvqa_fixture chartqa_proxy --out "$SM/pred"; rc=$?; }
         [ $rc -eq 0 ] && { $E/bin/python scripts/r7_d1_30_sparsevlm.py --sets-dir "$SM/sets" --bench gqa_fixture textvqa_fixture --rows 192 64 --out "$SM/pred_repro"; rc=$?; }
         [ $rc -eq 0 ] && { $E/bin/python scripts/r7_d1_30_sparsevlm.py --sets-dir "$SM/sets64" --bench gqa_fixture textvqa_fixture chartqa_proxy --out "$SM/timing" \
                              --timing 16 --allow-any-gpu; rc=$?; }
         [ $rc -eq 0 ] && { $E/bin/python scripts/r7_d1_30_score.py --sets-dir "$SM/sets64" --pred-dir "$SM/pred" --timing-dir "$SM/timing" \
                              --lr-choice "$SM/selectors/lr_choice.json" --out "$SM/results" --benches gqa_fixture textvqa_fixture chartqa_proxy --reps 200; rc=$?; }
         [ $rc -eq 0 ] && { $E/bin/python scripts/r7_d1_30_smoke_report.py --smoke "$SM"; rc=$?; };;
  *) echo "usage: run <bench> <i/n> | sparsevlm <bench> <i/n> | timing <bench> | devrisk | smoke"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
