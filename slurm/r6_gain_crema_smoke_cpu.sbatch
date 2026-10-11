#!/usr/bin/env bash
#SBATCH --job-name=r6crema_smoke_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 CREMA-D CPU preparation + smoke: env check, probe (ffprobe sample, WAV headers), prep (decode every clip once with PyAV into
# outputs/R6_GAIN/cremad/cache), unit tests, CPU smoke run (brier, seed 0, tiny subsets, with per-clip costs), smoke aggregate.
# Python: the r6_av env (PyAV 19.0.1; reuses the main env's packages read-only).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export PYTHON=/projects/EEG-foundation-model/yinghao/envs/r6_av/bin/python OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; rc=0
"$PYTHON" -c "import av, torch, torchvision, scipy, numpy; print('av', av.__version__, 'torch', torch.__version__, 'torchvision', torchvision.__version__, 'numpy', numpy.__version__)" || rc=1
"$PYTHON" scripts/r6_gain_crema.py probe || rc=1
[ -f /home/infres/yinwang/CS_QMI/outputs/R6_GAIN/cremad/cache/meta.json ] || "$PYTHON" scripts/r6_gain_crema.py prep --workers 16 --decoder av || rc=1
"$PYTHON" -m unittest tests/test_r6_gain_crema.py -v || rc=1
"$PYTHON" scripts/r6_gain_crema.py run --family brier --seed 0 --smoke --cpu --costs || rc=1
"$PYTHON" scripts/r6_gain_crema.py aggregate --smoke --mode single || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
