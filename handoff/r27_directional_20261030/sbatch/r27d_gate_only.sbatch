#!/bin/bash
#SBATCH --job-name=r27d_gate
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030/logs/%x_%j.out
# Re-run gate/r27d_gate.py on an existing R27-D smoke directory (smoke arms not repeated). Usage: r27d_gate_only.sbatch SMOKE_DIR
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030
Q=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $Q/sbatch/stage.sh
cd $D/code_frozen
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u -B $D/gate/r27d_gate.py $1 \
  /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy $DP \
  /projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/smoke_s_1075954/s4_single_cap_vcs/state_last.pt \
  $D/code_frozen /home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030
cp $1/GPU_GATE.json $D/GPU_GATE_${SLURM_JOB_ID}.json
echo GATE_DONE $1
