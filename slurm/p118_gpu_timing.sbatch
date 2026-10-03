#!/usr/bin/env bash
#SBATCH --job-name=p118_gpu_timing
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# GPU timing for P117 / P118 full-size units (to choose GPU vs CPU): one P117 cell, one P118 truth repeat, one P118 visual cell-repeat.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P118_GATE; mkdir -p $G reports/P117_GATE; banner "P117/P118 GPU timing"
/usr/bin/env bash -c "time '$PYTHON' scripts/p117_cal.py --cells C2_gauss_pad:16384:0 --out-dir /home/infres/yinwang/CS_QMI/outputs/P117_GATE_timing_gpu" > reports/P117_GATE/timing_gpu.txt 2>&1; tail -5 reports/P117_GATE/timing_gpu.txt
/usr/bin/env bash -c "time '$PYTHON' scripts/p118_truth.py --conds weak --repeats 1 --out $G/truth_timing_gpu" > $G/truth_timing_gpu.txt 2>&1; tail -4 $G/truth_timing_gpu.txt | cut -c1-300
/usr/bin/env bash -c "time '$PYTHON' scripts/p118_visual.py --runs P107_AP3_views4_800ep_seed2 --cells colour_s0.1 --repeats 1 --out $G/visual_timing_gpu" > $G/visual_timing_gpu.txt 2>&1; tail -4 $G/visual_timing_gpu.txt | cut -c1-300
echo "exit=0 end=$(date -u +%FT%TZ)"
