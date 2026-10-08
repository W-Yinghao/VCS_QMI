#!/usr/bin/env bash
#SBATCH --job-name=p149a1_run
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P149 add. 1 (FROZEN prereg): one encoder family = z-site on the existing seed-1 fixture, then the seed-2 fixture with h- and z-site measurements.
#   sbatch slurm/p149a1_run.sbatch <cifar10|cifar100> <seed-1 run> <seed-2 run>      (finished outputs are skipped)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
DS=$1; R1=$2; R2=$3; O1=$OUTPUT_ROOT/P149_twoview; O=$OUTPUT_ROOT/P149a1_twoview; mkdir -p $O; rc=0
m() { [ -f $O/$2.json ] && { echo "skip $2"; return 0; }; "$PYTHON" scripts/p149_twoview.py measure --fixture $1 --out $O/$2 --site $3 || rc=1; }
m $O1/$R1 measure_${R1}_z z
[ -f $O/$R2/manifest.json ] || "$PYTHON" scripts/p149_twoview.py fixture --dataset $DS --run $R2 --out $O || exit 1
m $O/$R2 measure_${R2} h
m $O/$R2 measure_${R2}_z z
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
