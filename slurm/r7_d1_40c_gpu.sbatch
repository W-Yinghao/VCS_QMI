#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_40c_gpu
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-40c Qwen2.5-VL-7B-Instruct units in the vl_baselines env (torch 2.9 cu128 -> any of RTX6000PRO / H100 / L40S; the unit lines set
# --partition / --time, node51 / node60 always excluded).  One host process per job (weights staged to node-local disk once); single-writer
# shard files; resumable.  Timing runs only on the reference GPU, as ONE bundle job (the line pins --partition=H100 --nodelist=nodesumo01;
# the code refuses another GPU).  First arg: run <bench> <i/n> | official <bench> <i/n> | timing_bundle <in_bundle|report> |
# devteacher <b012|b013> | devrisk | smoke | throughput | anchor_smoke.  Positional arguments only.
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R40=$D1/r7_40; RC=$R40/eval_40c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm | wc -l)"
echo "code: $(cat scripts/r7_d1_40c_*.py tests/test_r7_d1_40c.py slurm/r7_d1_40c_* 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the R7-40c scripts, tests and wrappers)"
case "$what" in
  run) $Q/bin/python -I scripts/r7_d1_40c_run.py run --sets-dir "$D1/d1_30/sets" --bench "$1" --configs-file "$RC/configs/configs_$1.txt" --shard "$2" --out "$RC/pred"; rc=$?;;
  official) $Q/bin/python -I scripts/r7_d1_40c_run.py official --sets-dir "$D1/d1_30/sets" --bench "$1" --shard "$2" --out "$RC/pred"; rc=$?;;   # decision 15
  timing_bundle)   # ONE nodesumo01 job (owner rule 2026-10-13): Qwen's B* (in_bundle: frozen R7-40a procedure unless its report exists | report), then the 3 benchmarks
    case "${1:-}" in in_bundle|report) ;; *) echo "timing_bundle needs in_bundle | report"; exit 2;; esac
    $Q/bin/python -I scripts/r7_d1_40c_timing.py bundle --sets-dir "$D1/d1_30/sets" --configs-dir "$RC/configs" --out "$RC/timing" --b-star-mode "$1"; rc=$?;;
  devteacher)   # the sealed DEV bundles with the frozen R7-40a teacher (D40a-5: DEV deferred to 40c; --allow-dev only here)
    if [[ "$1" =~ ^b01[23]$ ]]; then $Q/bin/python -I scripts/r7_d1_40_teacher.py bundle "$1" --manifest-dir "$D1/data" --out "$R40/teacher" --allow-dev; rc=$?
    else echo "devteacher accepts only the DEV bundles b012 / b013"; rc=2; fi;;
  devrisk) $Q/bin/python -I scripts/r7_d1_40c_devrisk.py run --out "$RC/devrisk"; rc=$?;;
  smoke) $Q/bin/python -I scripts/r7_d1_40c_smoke.py gpu --out "$R40/smoke_40c" "$@"; rc=$?;;
  throughput) $Q/bin/python -I scripts/r7_d1_40c_smoke.py throughput --out "$R40/smoke_40c"; rc=$?;;
  anchor_smoke) $Q/bin/python -I scripts/r7_d1_40c_smoke.py anchor --out "$R40/smoke_40c"; rc=$?;;
  *) echo "usage: run <bench> <i/n> | official <bench> <i/n> | timing_bundle <in_bundle|report> | devteacher <b012|b013> | devrisk | smoke | throughput | anchor_smoke"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
