#!/bin/bash
#SBATCH --job-name=r30v11_gate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/logs/%x_%j.out
# Re-run gate/r30v11_gate.py on an existing v1.1 smoke directory (smoke stages not repeated). Usage: r30v11_gate_only.sbatch SMOKE_DIR
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
cd $R30V11_CODE
set +e
$PYTHON_BIN -u -B $R30_ROOT/gate/r30v11_gate.py $1 $DP $R30V11_CODE $R30_ROOT/reference $R30V11_PKG
rc=$?
set -e
cp $1/GPU_GATE.json $R30_ROOT/GPU_GATE_V11_${SLURM_JOB_ID}.json
cp $1/GPU_GATE.json $R30_ROOT/GPU_GATE.json
echo GATE_DONE $1 rc=$rc
