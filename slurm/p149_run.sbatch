#!/usr/bin/env bash
#SBATCH --job-name=p149_run
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
#SBATCH --time=16:00:00
# P149 (FROZEN prereg) one fixture: build the two-view fixture (skipped if its manifest exists), then the 4 settings x 3 estimators x 3 seeds measurement.
#   sbatch slurm/p149_run.sbatch <cifar10|cifar100> <run_id>
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
DS=$1; RUN=$2; O=$OUTPUT_ROOT/P149_twoview; mkdir -p $O
[ -f $O/$RUN/manifest.json ] || "$PYTHON" scripts/p149_twoview.py fixture --dataset $DS --run $RUN --out $O || exit 1
"$PYTHON" scripts/p149_twoview.py measure --fixture $O/$RUN --out $O/measure_$RUN; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
