#!/usr/bin/env bash
#SBATCH --job-name=v10_t03_grid
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 T03 grid batch (decision 2026-10-17 9.2-9.4; protocol P2 t03; units from v10/protocol/T03_UNITS.csv).  CPU only, synthetic only.
#   slurm/v10_t03/submit.sh slurm/v10_t03/t03_grid.sbatch -- <CAL|formal> <processes> <threads>
#   CAL     the 252 tuning-suite units (data root 1017, algo 1017; need the 7 smoke units COMPLETE under the same T03 code hash),
#           then t03_run.py select (the 60 group selections -> selection/T03_CAL_SELECTION.json)
#   formal  main (600), algo_extra (120), capacity (240): need every CAL candidate of their (condition, method) COMPLETE under the same
#           T03 code hash (the runner BLOCKs otherwise)
# Each of <processes> processes runs every <processes>-th unit of each block (--shard k/P) with <threads> torch threads; COMPLETE units
# are kept, FAIL records are kept (no --retry-failed here), so a resubmission continues the batch.  Exit 0 only when every unit of
# the batch is COMPLETE (and, for CAL, the selection command ran).
# Layout basis: the smoke job 1043854 measured the 7 smoke units as 1 x 8, 4 x 2 and 8 x 1 threads on 8 CPUs (v10/T03/smoke/
# T03_BUDGET.json); 1-thread processes gave the lowest CPU cost, so the planned call is <processes> = the CPUs of the job, 1 thread:
#   slurm/v10_t03/submit.sh slurm/v10_t03/t03_grid.sbatch -- CAL 32 1        then, after the CAL selection,  -- formal 32 1
set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_t03
cd "$R" || exit 2
BATCH=${1:?usage: t03_grid.sbatch <CAL|formal> <processes> <threads>} P=${2:?processes} T=${3:?threads}
case "$BATCH" in CAL) BLOCKS="CAL"; SEEDS="1017" ;; formal) BLOCKS="main algo_extra capacity"; SEEDS="$(seq -s ' ' 2026101700 2026101709)" ;;
  *) echo "[v10_t03] bad batch $BATCH"; exit 2 ;; esac
export PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python PYTHONUNBUFFERED=1 PYTHONPATH="$R/src:$R/scripts/v10_t03:$R"
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" CUBLAS_WORKSPACE_CONFIG=:4096:8
export V10_T03_OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v10/T03
source "$R/slurm/v10_t03/t03_header.sh" || exit 2
t03_header cpu slurm/v10_t03/t03_grid.sbatch "grid:$BATCH:${P}x${T}" || exit 3
T0=$(date +%s)
"$PYTHON" scripts/v10_t03/t03_run.py units --check || exit 4
# shellcheck disable=SC2086
"$PYTHON" scripts/v10_t03/t03_run.py gen --root-seed $SEEDS || exit 4          # suites + package checker, sequentially (no race)
"$PYTHON" scripts/v10_t03/t03_run.py gate || exit 4                              # gates under the current T03 code hash
mkdir -p "$V10_T03_OUT_ROOT/logs/$SLURM_JOB_ID"
pids=()
for k in $(seq 0 $((P - 1))); do
  (
    rc=0
    for b in $BLOCKS; do "$PYTHON" scripts/v10_t03/t03_run.py run --block "$b" --shard "$k/$P" --threads "$T" || rc=1; done
    exit $rc
  ) > "$V10_T03_OUT_ROOT/logs/$SLURM_JOB_ID/shard_${k}of${P}.log" 2>&1 &
  pids+=($!)
done
rc_r=0
for p in "${pids[@]}"; do wait "$p" || rc_r=1; done
grep -h -E "^\[t03\] .* (COMPLETE|FAIL|BLOCKED)" "$V10_T03_OUT_ROOT/logs/$SLURM_JOB_ID"/shard_*.log | sort | uniq -c | sort -rn | head -5
"$PYTHON" scripts/v10_t03/t03_run.py status
rc_sel=NOT_RUN
if [ "$BATCH" = CAL ]; then "$PYTHON" scripts/v10_t03/t03_run.py select; rc_sel=$?; fi
echo "[v10_t03] t03_grid batch=${BATCH} layout=${P}x${T} units_rc=${rc_r} select_rc=${rc_sel} wall=$(( $(date +%s) - T0 ))s end=$(date -u +%FT%TZ)"
[ "$rc_r" -eq 0 ] && { [ "$rc_sel" = NOT_RUN ] || [ "$rc_sel" = 0 ]; } || exit 1
