#!/usr/bin/env bash
#SBATCH --job-name=r7_e2_gpu
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 E2 (Qwen2.5-VL-7B on FIT_FULL) units in the vl_baselines env (torch 2.9 cu128, transformers 4.57.6).  The unit lines set --partition / --time
# (teacher bundles and quality shards L40S only; the cache any of RTX6000PRO / H100 / L40S; node51 / node60 always excluded).  The frozen R7-40
# scripts run unchanged, python -I; one host process per job; resumable (teacher per question, runner per configuration and item; the one-pass
# cache restarts).  First arg: bundle <f###|r###> | cache <PCA image count> | run <bench> <i/n> | smoke.  Positional arguments only (no --export).
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; E2=/home/infres/yinwang/CS_QMI/outputs/R7_E2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: E2 $(cat scripts/r7_e2_*.py tests/test_r7_e2.py slurm/r7_e2_* 2>/dev/null | sha256sum | cut -c1-16), frozen R7-40 $(cat scripts/r7_d1_40_*.py scripts/r7_d1_40c_*.py | sha256sum | cut -c1-16)"
case "$what" in
  bundle) [[ "${1:-}" =~ ^[fr][0-9]{3}$ ]] || { echo "bundle <f###|r###>"; exit 2; }; $Q/bin/python -I scripts/r7_d1_40_teacher.py bundle "$1" --manifest-dir "$E2/data" --out "$E2/teacher" --probe-report "$E2/smoke/probe/probe_report.json"; rc=$?;;
  cache) [[ "${1:-}" =~ ^[0-9]+$ ]] || { echo "cache <PCA image count>"; exit 2; }
         $Q/bin/python -I scripts/r7_d1_40_cache.py d1_40 --manifest-dir "$E2/data" --out "$E2/cache" --max-images "$1"; rc=$?;;
  run) case "${1:-}" in gqa_dev|textvqa_val|chartqa_val) ;; *) echo "run <gqa_dev|textvqa_val|chartqa_val> <i/n>"; exit 2;; esac
       $Q/bin/python -I scripts/r7_d1_40c_run.py run --sets-dir "$D1/d1_30/sets" --bench "$1" --configs-file "$E2/eval/run/configs_$1.txt" --shard "$2" \
         --out "$E2/eval/pred" --selectors-dir "$E2/selectors" --pca "$E2/cache/pca.pt"; rc=$?;;
  smoke) $Q/bin/python -I scripts/r7_e2_smoke.py gpu --out "$E2/smoke" "$@"; rc=$?;;     # probe + replay (L40S); before the freeze
  *) echo "usage: bundle <f###|r###> | cache <PCA image count> | run <bench> <i/n> | smoke"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
