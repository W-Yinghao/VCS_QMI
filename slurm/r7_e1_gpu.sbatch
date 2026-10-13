#!/usr/bin/env bash
#SBATCH --job-name=r7_e1_gpu
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:15:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 E1 GPU units: external Qwen2.5-VL-7B baselines (MMTok @8e26868 in envs/mmtok, HiPrune @8278100 in envs/hiprune), each as its official code ships
# it, batch 1, on R7-40c's validation sets / prompts / generation / sharding (reports/R7/R7_E1_EXTERNAL_PROTOCOL_DRAFT_20261013.yaml).  The unit lines
# set --partition / --time (L40S; node51 / node60 always excluded).  One job per (method, benchmark, shard): checkpoint staged to node-local disk once,
# then one fresh interpreter per fraction (0.125 / 0.25 / 0.5 / 0.75, + the in-env anchor f = 1 with 'anchor'); single-writer shard files; resumable;
# marker pred/<bench>/done_e1_<method>_shard<i>of<n>.json.  First arg: run <mmtok|hiprune> <bench> <i/n> [anchor] | smoke <mmtok|hiprune> [items]
# (R7-40c smoke fixtures: R7-20 GQA / TextVQA fixtures and the ChartQA TRAIN subset, never val / test; all fractions + anchor; own output dir)
# | smokeval <mmtok|hiprune> <fraction> [items] (coordinator's validation smoke: first items of GQA DEV shard 0/6 + ChartQA val shard 0/2 at one
# fraction into outputs/R7_E1/smoke/<method>, then a read-back with R7-40c's loader; no answer key read).
# Positional arguments only.  cwd = this repo (never the fetched official tree); python -I.
ENVS=/projects/EEG-foundation-model/yinghao/envs; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R40=$D1/r7_40; E1=$R40/eval_e1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: $(cat scripts/r7_e1_external.py tests/test_r7_e1_external.py slurm/r7_e1_* 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the E1 script, tests and wrappers)"
m="${1:-}"; case "$m" in mmtok|hiprune) PY=$ENVS/$m/bin/python;; *) echo "method must be mmtok | hiprune"; exit 2;; esac
[ -x "$PY" ] || { echo "no env $ENVS/$m"; exit 2; }
case "$what" in
  run)
    [ -n "${2:-}" ] && [[ "${3:-}" =~ ^[0-9]+/[0-9]+$ ]] || { echo "run <method> <bench> <i/n> [anchor]"; exit 2; }
    extra=(); [ "${4:-}" = anchor ] && extra=(--anchor)
    "$PY" -I scripts/r7_e1_external.py run --method "$m" --bench "$2" --shard "$3" --sets-dir "$D1/d1_30/sets" --out "$E1/pred" "${extra[@]}"; rc=$?;;
  smoke)
    rc=0; n="${2:-16}"; [[ "$n" =~ ^[0-9]+$ ]] || { echo "smoke <method> [items]"; exit 2; }
    for b in gqa_fixture textvqa_fixture chartqa_train_smoke; do
      "$PY" -I scripts/r7_e1_external.py run --method "$m" --bench "$b" --shard 0/1 --sets-dir "$R40/smoke_40c/sets64" --out "$E1/smoke/pred" --anchor --limit "$n" || rc=$((rc + 1))
    done;;
  smokeval)   # coordinator's smoke on validation items: the first <items> items of GQA DEV shard 0/6 and ChartQA val shard 0/2 at ONE fraction,
              # into outputs/R7_E1/smoke/<method>; then the read-back check with R7-40c's loader (no answer key read, nothing scored)
    f="${2:-}"; n="${3:-16}"; [[ "$f" =~ ^(0\.125|0\.25|0\.5|0\.75|1)$ && "$n" =~ ^[0-9]+$ ]] || { echo "smokeval <method> <fraction> [items]"; exit 2; }
    SM=/home/infres/yinwang/CS_QMI/outputs/R7_E1/smoke/$m; rc=0
    for bs in gqa_dev@0/6 chartqa_val@0/2; do
      "$PY" -I scripts/r7_e1_external.py run --method "$m" --bench "${bs%@*}" --shard "${bs#*@}" --sets-dir "$D1/d1_30/sets" --out "$SM" --fractions "$f" --limit "$n" || rc=$((rc + 1))
    done
    "$PY" -I scripts/r7_e1_external.py smokecheck --method "$m" --out "$SM" --sets-dir "$D1/d1_30/sets" --benches gqa_dev@0/6 chartqa_val@0/2 --fractions "$f" --limit "$n" || rc=$((rc + 1));;
  *) echo "usage: run <mmtok|hiprune> <bench> <i/n> [anchor] | smoke <mmtok|hiprune> [items] | smokeval <mmtok|hiprune> <fraction> [items]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
