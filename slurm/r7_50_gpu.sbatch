#!/usr/bin/env bash
#SBATCH --job-name=r7_50_gpu
#SBATCH --partition=H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-50 GPU units: LLaVA-1.5 host in the SparseVLM env (torch 2.1.2 + flash-attn 2.3.3, no Blackwell kernels -> H100 / L40S only; every
# submitting line repeats --partition / --exclude, integrity units pin the reference's GPU class).  Resumable per (row, item).  python -I.
# Positional arguments only (no --export).  First arg:
#   run <bench> <i/n> <root> [phase L|F] | sparsevlm <bench> <i/n> <root> | integrity <root> <phase> <unit> | smoke <S> | smoke-integrity <S> <name>
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; PY="$E/bin/python -I"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) git=$(git rev-parse --short HEAD 2>/dev/null)"
case "$what" in
  run)
    [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] || { echo "usage: run <bench> <i/n> <root> [phase]"; exit 2; }
    $PY scripts/r7_50_test.py run --root "$3" --bench "$1" --shard "$2" --phase "${4:-L}"; rc=$?;;
  sparsevlm)
    [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] || { echo "usage: sparsevlm <bench> <i/n> <root>"; exit 2; }
    $PY scripts/r7_50_test.py sparsevlm --root "$3" --bench "$1" --shard "$2"; rc=$?;;
  integrity)
    [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] || { echo "usage: integrity <root> <phase> <unit>"; exit 2; }
    $PY scripts/r7_50_test.py integrity-run --root "$1" --phase "$2" --unit "$3"; rc=$?;;
  smoke)
    S="$1"; [ -n "$S" ] || { echo "usage: smoke <smoke_root>"; exit 2; }
    $PY scripts/r7_50_smoke.py gpu --smoke "$S"; rc=$?
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py sparsevlm --root "$S/integrity" --bench gqa_testdev chartqa_test --shard 0/1 --rows 192 --limit 96; rc=$?; }
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py sparsevlm --root "$S/scored" --bench gqa_testdev chartqa_test --shard 0/1; rc=$?; };;
  smoke-integrity)
    S="$1"; N="$2"; [ -n "$S" ] && [ -n "$N" ] || { echo "usage: smoke-integrity <smoke_root> <integrity_root_name>"; exit 2; }
    $PY scripts/r7_50_smoke.py gpu-integrity --smoke "$S" --integrity-root "$N"; rc=$?
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py sparsevlm --root "$S/$N" --bench gqa_testdev chartqa_test --shard 0/1 --rows 192 --limit 96; rc=$?; };;
  *) echo "usage: run <bench> <i/n> <root> [phase] | sparsevlm <bench> <i/n> <root> | integrity <root> <phase> <unit> | smoke <S> | smoke-integrity <S> <name>"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
