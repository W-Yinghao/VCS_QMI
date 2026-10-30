#!/bin/bash
#SBATCH --job-name=r27s_gate
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/logs/%x_%j.out
# Re-run r27s_gate.py on an existing R27-S smoke directory (arm runs not repeated). Usage: r27s_gate_only.sbatch SMOKE_DIR
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage.sh
cd $D/code_v12
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u r27s_gate.py $1 /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy $DP
echo GATE_DONE $1
