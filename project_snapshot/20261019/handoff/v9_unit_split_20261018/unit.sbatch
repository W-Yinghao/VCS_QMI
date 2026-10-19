#!/usr/bin/env bash
#SBATCH --job-name=v9f_unit
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
TASK_ROOT=/home/infres/yinwang/CS_QMI/handoff/v9_unit_split_20261018
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8
cd "$TASK_ROOT"
record_exit() {
    RUN_RC=$?
    export RUN_RC FULL_GROUP="$1" UNIT_INDEX="$2"
    /home/infres/yinwang/CS_QMI/env/bin/python - <<'PY'
import os, time
from split_common import BUNDLE, write
write(BUNDLE/'units'/os.environ['FULL_GROUP']/os.environ['UNIT_INDEX']/('EXIT_'+os.environ['SLURM_JOB_ID']+'.json'),
      {'exit_code':int(os.environ['RUN_RC']),'job_id':os.environ.get('SLURM_JOB_ID'),
       'utc':time.strftime('%FT%TZ',time.gmtime())})
PY
}
trap 'record_exit "$1" "$2"' EXIT
trap 'exit 143' TERM
/home/infres/yinwang/CS_QMI/env/bin/python -u run_unit.py --group "$1" --unit-index "$2"
