#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_gpu
#SBATCH --partition=H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-10 GPU units.  The LLaVA-1.5 host runs in the SparseVLM env (torch 2.1.2 + flash-attn 2.3.3): no Blackwell kernels, so H100 / L40S
# only — never add RTX6000PRO here.  One host process per job (weights staged to node-local disk once), per-chunk appends, skip-if-done.
# First arg: bundle <id> | probe | timing | smoke (probe + timing, one load).  Timing must run on the pinned reference GPU (sbatch --partition=H100 --nodelist=nodesumo01).
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; FX=/projects/EEG-foundation-model/yinghao/data/r7_fixtures
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
case "$what" in
  bundle) $E/bin/python scripts/r7_d1_teacher.py bundle "$1" --manifest-dir "$D1/data" --out "$D1/teacher"; rc=$?;;
  probe) $E/bin/python scripts/r7_d1_teacher.py probe --out "$D1/probe" --fixture "$FX/gqa.jsonl" \
           --r720-preds /home/infres/yinwang/CS_QMI/outputs/R7_fixtures/d1/pred_full.jsonl --check-generation; rc=$?;;
  timing) $E/bin/python scripts/r7_d1_teacher.py timing --out "$D1/timing" --fixture "$FX/gqa.jsonl" "$@"; rc=$?;;
  smoke) $E/bin/python scripts/r7_d1_teacher.py smoke --out "$D1/smoke" --fixture "$FX/gqa.jsonl" \
           --r720-preds /home/infres/yinwang/CS_QMI/outputs/R7_fixtures/d1/pred_full.jsonl "$@"; rc=$?;;
  cache_smoke) $E/bin/python scripts/r7_d1_cache.py smoke --fixture "$FX/gqa.jsonl" --out "$D1/cache_smoke"; rc=$?;;
  d1_20_smoke)   # one slot: feature cache on the fixture (LLaVA env) then the three selectors' pipeline smoke (main env, torch 2.9)
    $E/bin/python scripts/r7_d1_cache.py smoke --fixture "$FX/gqa.jsonl" --out "$D1/cache_smoke"; rc=$?
    [ $rc -eq 0 ] && { /home/infres/yinwang/CS_QMI/env/bin/python scripts/r7_d1_train.py smoke --probe-records "$D1/smoke/probe/probe_records.jsonl" \
                         --cache "$D1/cache_smoke" --out "$D1/train_smoke" "$@"; rc=$?; };;
  cache) $E/bin/python scripts/r7_d1_cache.py "$@" --manifest-dir "$D1/data" --out "$D1/cache"; rc=$?;;
  *) echo "usage: bundle <id> | probe | timing | smoke | cache_smoke | d1_20_smoke | cache <fit_pca|encode --role R|questions>"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
