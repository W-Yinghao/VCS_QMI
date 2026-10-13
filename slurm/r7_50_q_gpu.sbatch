#!/usr/bin/env bash
#SBATCH --job-name=r7_50q_gpu
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-50 phase Q GPU units (addendum reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml, Q3): the R7-40c quality runner, unchanged
# (scripts/r7_d1_40c_run.py run; Qwen2.5-VL-7B-Instruct in the vl_baselines env, as slurm/r7_d1_40c_gpu.sbatch), pointed at the R7-50 roots.
# Before the runner, scripts/r7_50_q.py runner-args checks the unit (generation: sealed inputs unchanged, integrity gate passed, manifest_Q
# absent, configs_Q unedited, selector / PCA sha256 as in configs_Q/rows.json; integrity: this GPU's class = the R7-40c reference marker's, gate
# rows and S1 seed-0 / PCA sha256 as in the plan) and prints the runner's arguments; a refusal stops the unit before any model load.
# Every submitting line sets --partition / --time (L40S for generation, the reference class for integrity units); node51 / node60 excluded.
# Resumable per (configuration, item); single-writer shard files.  Positional arguments only (no --export).  First arg:
#   run <bench> <i/n> <root>            test generation shard: --sets-dir <root>/sealed/inputs --out <root>/sealed/pred_Q
#   integrity <root> <bench> <i/n>      full-scale gate shard (GQA DEV / ChartQA val): --sets-dir / --out under <root>/integrity_Q
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; PY="$Q/bin/python -I"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm | wc -l)"
echo "code: $(cat scripts/r7_50_q.py tests/test_r7_50_q.py slurm/r7_50_q_* 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the phase-Q script, tests and wrappers)"
runner() {   # $@: runner-args options; executes 'r7_d1_40c_run.py run' with the verified arguments
  local out; out=$($PY scripts/r7_50_q.py runner-args "$@") || { echo "guard refused: the unit does not run"; return 3; }
  read -r -a ARGV <<< "$out"; echo "runner: scripts/r7_d1_40c_run.py ${ARGV[*]}"
  $PY scripts/r7_d1_40c_run.py "${ARGV[@]}"; }
case "$what" in
  run)
    [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] || { echo "usage: run <bench> <i/n> <root>"; exit 2; }
    runner --root "$3" --bench "$1" --shard "$2"; rc=$?;;
  integrity)
    [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] || { echo "usage: integrity <root> <bench> <i/n>"; exit 2; }
    runner --root "$1" --bench "$2" --shard "$3" --integrity; rc=$?;;
  *) echo "usage: run <bench> <i/n> <root> | integrity <root> <bench> <i/n>"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
