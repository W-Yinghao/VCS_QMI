#!/usr/bin/env bash
#SBATCH --job-name=p156_sep
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P156 (SEP-1).  MODE=run (SEED required; BLOCKS optional, "+"-separated, never commas in --export) | repro | timing | gpugate (timing + repro).  OUT required.  Resumable: finished cells are skipped.
#   sbatch --job-name=p156_s100 --export=ALL,MODE=run,SEED=100,OUT=/home/infres/yinwang/CS_QMI/outputs/P156_sep slurm/p156_sep.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${MODE:?}"; : "${OUT:?}"
banner "P156 $MODE seed=${SEED:-NA} blocks=${BLOCKS:-all} -> $OUT"
case "$MODE" in
  run)    : "${SEED:?}"; rc=0                                    # SEED may be a "+"-separated list (e.g. 100+101): seeds run in order
          for s in ${SEED//+/ }; do "$PYTHON" scripts/p156_sep.py run --seed "$s" --out "$OUT" ${BLOCKS:+--blocks $BLOCKS} || rc=1; done ;;
  repro)  "$PYTHON" scripts/p156_sep.py repro --out "$OUT"; rc=$? ;;
  timing) "$PYTHON" scripts/p156_sep.py timing --out "$OUT"; rc=$? ;;
  gpugate) "$PYTHON" scripts/p156_sep.py timing --out "$OUT/timing"; r1=$?; "$PYTHON" scripts/p156_sep.py repro --out "$OUT/repro"; r2=$?
           echo "timing_rc=$r1 repro_rc=$r2"; rc=$(( r1 | r2 )) ;;
  *) echo "unknown MODE $MODE"; rc=2 ;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
