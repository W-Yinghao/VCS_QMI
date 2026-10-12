#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_40_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-40 Qwen2.5-VL-7B-Instruct units in the vl_baselines env (torch 2.9 cu128 -> any of RTX6000PRO / H100 / L40S).  One host process per job
# (weights staged to node-local disk once), per-chunk appends to single-writer shard files, skip-if-done.  Python runs with -I (downloaded data
# and weights are read; the scripts insert their own directory).  Timing runs only on the reference GPU: sbatch --partition=H100
# --nodelist=nodesumo01.  First arg: bundle <id> | probe [s4 rule] [s1 rule] | timing [...] | smoke [tag] | cache_smoke | cache.  Positional arguments only.
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R40=$D1/r7_40
FX=/projects/EEG-foundation-model/yinghao/data/r7_fixtures
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm | wc -l)"
echo "code: $(cat scripts/r7_d1_40_*.py tests/test_r7_d1_40.py slurm/r7_d1_40_* | sha256sum | cut -c1-16) (sha256 of the R7-40 scripts, tests and wrappers)"
case "$what" in
  bundle) $Q/bin/python -I scripts/r7_d1_40_teacher.py bundle "$1" --manifest-dir "$D1/data" --out "$R40/teacher"; rc=$?;;
  probe) $Q/bin/python -I scripts/r7_d1_40_teacher.py probe --out "$R40/probe" --s4-rule "${1:-min}" --s1-rule "${2:-r720}"; rc=$?;;
  timing) $Q/bin/python -I scripts/r7_d1_40_teacher.py timing --out "$R40/timing" "$@"; rc=$?;;
  smoke)   # probe + throughput + memory probe (one load; the probe's S4 verdict is reported, the smoke continues), then the selector cache smoke.
           # With a tag (smoke <tag>): the teacher smoke only, into smoke_<tag> (a second GPU type; never the launch-probe or cache-smoke paths).
    tag=""; if [ -n "${1:-}" ] && [[ "$1" != -* ]]; then tag=$1; shift; fi
    if [ -n "$tag" ]; then $Q/bin/python -I scripts/r7_d1_40_teacher.py smoke --out "$R40/smoke_$tag" "$@"; rc=$?; [ $rc -eq 4 ] && rc=0
    else
      $Q/bin/python -I scripts/r7_d1_40_teacher.py smoke --out "$R40/smoke" "$@"; rc=$?; [ $rc -eq 4 ] && { echo "probe sentinels not all passing (see report)"; rc=0; }
      [ $rc -eq 0 ] && { $Q/bin/python -I scripts/r7_d1_40_cache.py smoke --fixture "$FX/gqa.jsonl" --out "$R40/cache_smoke"; rc=$?; }
    fi;;
  cache_smoke) $Q/bin/python -I scripts/r7_d1_40_cache.py smoke --fixture "$FX/gqa.jsonl" --out "$R40/cache_smoke"; rc=$?;;
  cache) $Q/bin/python -I scripts/r7_d1_40_cache.py d1_40 --manifest-dir "$D1/data" --out "$R40/cache"; rc=$?;;
  *) echo "usage: bundle <id> | probe [s4 rule] [s1 rule] | timing [...] | smoke [tag] | cache_smoke | cache"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
