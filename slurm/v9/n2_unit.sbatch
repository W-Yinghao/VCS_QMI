#!/usr/bin/env bash
#SBATCH --job-name=v9_n2
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2 (package v1 docs/05).  Positional MODE:
#   gate                      CPU-capable: tests/v9/test_n2.py + the fixture smoke (`n2_run.py smoke`) — code path only, never evidence
#   smoke-gpu                 the fixture smoke on a GPU (timing of the code path)
#   unit  <dataset> <rho> <method> <seed> [lr_mult]      one formal unit on the real precomp data (needs DATA_ROOT, VOCAB, HOST_LOCK; resumable; skips when done)
#   diag  <spec json>         A/B diagnostic: configs trained from scratch for warm-up + the first steps of the hard phase (n2_run.py diag)
#   group <units file>        several `unit` lines run back to back on one GPU (owner rule: no many short jobs); each line = "dataset rho method seed lr_mult"
# Environment: V9_SGR_PY (python of fork A's v9_sgr env; default = main env), DATA_ROOT (…/ncr_precomp), VOCAB (…/vocab), OUT_ROOT (default
# /projects/EEG-foundation-model/yinghao/outputs_v9/n2), HOST_LOCK (reports/V9/HOST_LOCK.yaml), FORMAL=1 to require a verified HOST_LOCK, TEST=1 to
# evaluate the selected checkpoint on the official test split (only when the frozen protocol says so).  Code is snapshotted at job start.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
MODE=${1:?mode}; shift
PY=${V9_SGR_PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; OUT_ROOT=${OUT_ROOT:-/projects/EEG-foundation-model/yinghao/outputs_v9/n2}; mkdir -p "$OUT_ROOT"
SNAP=$(mktemp -d /tmp/v9_n2_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/src/v9 $SNAP/scripts/v9 $SNAP/v9
cp -r src/v9/n2 $SNAP/src/v9/; cp src/v9/*.py $SNAP/src/v9/; cp scripts/v9/n2_run.py $SNAP/scripts/v9/; cp -r v9/package_v9_20261016 $SNAP/v9/   # whole src/v9 (quant.py needs its reference_* siblings)
export PYTHONPATH=$SNAP/src
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # HOST_LOCK 57366c5 / 3deb194: official evaluators (torch >= 2.6 weights_only default)
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/src/v9/n2/*.py $SNAP/scripts/v9/n2_run.py | sha256sum | cut -c1-16) mode=$MODE args=$*"
DEV=cuda; nvidia-smi >/dev/null 2>&1 || DEV=cpu
case "$MODE" in
  gate) $PY -m pytest -q tests/v9/test_n2.py -p no:cacheprovider 2>&1 | tail -5; rc=${PIPESTATUS[0]}
        $PY $SNAP/scripts/v9/n2_run.py smoke --out $OUT_ROOT/smoke_${SLURM_JOB_ID} --device $DEV --epochs 2 2>&1 | tail -60; rc=$((rc | ${PIPESTATUS[0]}));;
  smoke-gpu) $PY $SNAP/scripts/v9/n2_run.py smoke --out $OUT_ROOT/smoke_${SLURM_JOB_ID} --device $DEV --epochs 2 --n-images 200; rc=$?;;
  unit) : "${DATA_ROOT:?}" "${VOCAB:?}"; ds=$1; rho=$2; m=$3; seed=$4; lm=${5:-1}
        $PY $SNAP/scripts/v9/n2_run.py unit --dataset $ds --rho $rho --method $m --seed $seed --lr-mult $lm --data-root $DATA_ROOT --vocab $VOCAB --out-root $OUT_ROOT \
            --host-lock ${HOST_LOCK:-reports/V9/HOST_LOCK.yaml} --device $DEV ${FORMAL:+--formal} ${TEST:+--test}; rc=$?;;
  group) : "${DATA_ROOT:?}" "${VOCAB:?}"; rc=0
        while read -r ds rho m seed lm members; do [ -z "$ds" ] && continue; echo "[unit] $(date -u +%FT%TZ) $ds rho=$rho $m s$seed lr$lm members=${members:-$m}"
          $PY $SNAP/scripts/v9/n2_run.py unit --dataset $ds --rho $rho --method $m --seed $seed --lr-mult ${lm:-1} ${members:+--members $members} --data-root $DATA_ROOT --vocab $VOCAB --out-root $OUT_ROOT \
              --host-lock ${HOST_LOCK:-reports/V9/HOST_LOCK.yaml} --device $DEV ${FORMAL:+--formal} ${TEST:+--test}; r=$?; echo "[unit-exit] rc=$r"; [ $r -ne 0 ] && rc=$r
        done < "$1";;
  diag) : "${DATA_ROOT:?}" "${VOCAB:?}"                    # A/B diagnostic (main session 2026-10-17): $1 = spec json (configs + pre-registered rule)
        $PY $SNAP/scripts/v9/n2_run.py diag --spec $1 --data-root $DATA_ROOT --vocab $VOCAB --out-root $OUT_ROOT --host-lock ${HOST_LOCK:-reports/V9/HOST_LOCK.yaml} --device $DEV; rc=$?;;
  *) echo "unknown mode $MODE"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
