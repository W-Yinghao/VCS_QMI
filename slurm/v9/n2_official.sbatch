#!/usr/bin/env bash
#SBATCH --job-name=v9_n2_off
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2 wave 2 (NCR+A / L2RM+A through the PATCHED official code; package docs/05 §1, §6).  Positional MODE:
#   smoke                                                   real-data subset, anchors ON, 1 warm-up + 1 main epoch per baseline (scripts/v9/n2_official_smoke.py)
#   unit <ncr|l2rm> <dataset> <rho> <seed> <lr_mult> <tuning|formal>
#        export the V9 mapping of the shared stage <OUT_ROOT>/<dataset>/rho<rho>/s<seed> -> launch -> collect-dev (dev best rSum, tuning selection)
#        [-> collect: official evaluation on the official test split, formal only].  Unit dir: .../s<seed>/unit_<NCR_A|L2RM_A>_lr<lr_mult>.
#        Skips when done; a launch without rc 0 is moved aside and relaunched (NCR has no co-training resume: one run must fit one job).
# Code (driver, patches, HOST_LOCK) is snapshotted at job start; the official clones are copied per unit by the driver.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
MODE=${1:?mode}; shift
PY=/home/infres/yinwang/CS_QMI/env/bin/python; OUT_ROOT=${OUT_ROOT:-/projects/EEG-foundation-model/yinghao/outputs_v9/n2}
DATA_ROOT=${DATA_ROOT:-/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data}
SNAP=$(mktemp -d /tmp/v9_n2off_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/src/v9 $SNAP/scripts/v9 $SNAP/v9 $SNAP/reports/V9
cp -r src/v9/n2 $SNAP/src/v9/; cp src/v9/*.py $SNAP/src/v9/; cp scripts/v9/n2_official.py scripts/v9/n2_official_smoke.py $SNAP/scripts/v9/; cp -r v9/patches $SNAP/v9/; cp reports/V9/HOST_LOCK.yaml $SNAP/reports/V9/
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) mode=$MODE $*"
case "$MODE" in
  smoke) $PY $SNAP/scripts/v9/n2_official_smoke.py --out $OUT_ROOT/official_smoke_${SLURM_JOB_ID:-local} "$@"; rc=$?;;
  unit) b=$1; ds=$2; rho=$3; seed=$4; lm=$5; kind=$6; [ "$b" = ncr ] && meth=NCR_A || meth=L2RM_A
        SH=$OUT_ROOT/$ds/rho$rho/s$seed; U=$SH/unit_${meth}_lr$lm; mkdir -p $U; rc=0; echo "[unit] $(date -u +%FT%TZ) $b $ds rho=$rho s$seed lr$lm $kind -> $U"
        if [ -f $U/done.json ] && { [ "$kind" = tuning ] || [ -f $U/result.json ]; }; then echo "[skip] done"
        else
          [ -f $U/v9_observed_t2i.npy ] || $PY $SNAP/scripts/v9/n2_official.py export --shared $SH --data-root $DATA_ROOT/${ds}_precomp --out $U || rc=1
          if [ $rc -eq 0 ] && ! grep -q '"rc": 0' $U/launch_manifest.json 2>/dev/null; then
            $PY $SNAP/scripts/v9/n2_official.py launch --baseline $b --unit $U --dataset $ds --seed $seed --lr-mult $lm --rho $rho --data-root $DATA_ROOT; rc=$?
          fi
          [ $rc -eq 0 ] && { $PY $SNAP/scripts/v9/n2_official.py collect-dev --baseline $b --unit $U; rc=$?; }
          [ $rc -eq 0 ] && [ "$kind" = formal ] && { $PY $SNAP/scripts/v9/n2_official.py collect --baseline $b --unit $U --dataset $ds --data-root $DATA_ROOT; rc=$?; }
        fi
        echo "[unit-exit] rc=$rc";;
  *) echo "unknown mode $MODE"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
