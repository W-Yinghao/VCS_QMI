#!/usr/bin/env bash
#SBATCH --job-name=r6crema_p1
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 CREMA-D full run in ONE GPU job: the given families x seeds 0..N-1 (x folds 0-4 when MODE=cv5) sequentially.  Seed 0 of each
# family also measures the per-DEV-clip cost table (every fold).  A system whose result.json exists is skipped, so a rerun resumes; the
# aggregate runs at the end (it reports missing / incomplete systems).  Args: MODE (single | cv5; default cv5), N_SEEDS (default 10),
# FAMILIES (default "brier ce"; pass one family per job to run two jobs in parallel).  r6_av env (PyAV for in-process FLV decoding).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export PYTHON=/projects/EEG-foundation-model/yinghao/envs/r6_av/bin/python
MODE=${1:-cv5}; NS=${2:-10}; FAMS=${3:-brier ce}; OUTD=/home/infres/yinwang/CS_QMI/outputs/R6_GAIN/cremad; rc=0
if [ "$MODE" = cv5 ]; then FOLDS="0 1 2 3 4"; else FOLDS="-1"; fi
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) mode=$MODE seeds=$NS families=$FAMS start=$(date -u +%FT%TZ)"
for s in $(seq 0 $((NS - 1))); do
  for f in $FAMS; do
    for k in $FOLDS; do
      d="$OUTD/${f}_s$s"; [ "$k" != "-1" ] && d="${d}_f$k"
      if [ -f "$d/result.json" ]; then echo "skip $(basename "$d") (result.json exists)"; continue; fi
      extra=""; [ "$s" = 0 ] && extra="--costs"
      "$PYTHON" scripts/r6_gain_crema.py run --family "$f" --seed "$s" --fold "$k" $extra || rc=1
    done
  done
done
"$PYTHON" scripts/r6_gain_crema.py aggregate --mode "$MODE" || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
