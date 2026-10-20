#!/usr/bin/env bash
#SBATCH --job-name=v9n2m_seed
#SBATCH --partition=RTX6000PRO
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo 'usage: sbatch run_seed.sbatch PACKAGE_ROOT EXPERIMENT_JSON SEED PYTHON_EXECUTABLE' >&2
    exit 2
fi
N2M_PACKAGE="$1"
N2M_EXPERIMENT="$2"
N2M_SEED="$3"
N2M_PYTHON="$4"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
N2M_CHILD=''
N2M_WARNING_SEEN=0
forward_warning() {
    N2M_WARNING_SEEN=1
    if [ -n "$N2M_CHILD" ]; then kill -USR1 "$N2M_CHILD" 2>/dev/null || true; fi
}
forward_term() {
    if [ -n "$N2M_CHILD" ]; then kill -TERM "$N2M_CHILD" 2>/dev/null || true; fi
}
trap forward_warning USR1
trap forward_term TERM INT
"$N2M_PYTHON" -u "$N2M_PACKAGE/run_n2m.py" run --experiment "$N2M_EXPERIMENT" --seed "$N2M_SEED" &
N2M_CHILD=$!
if [ "$N2M_WARNING_SEEN" -eq 1 ]; then forward_warning; fi
# A trapped signal interrupts bash wait before the Python child finishes. Keep
# waiting for the child so its complete-epoch checkpoint and attempt receipt land.
set +e
while true; do
    wait "$N2M_CHILD"
    N2M_RC=$?
    if ! kill -0 "$N2M_CHILD" 2>/dev/null; then break; fi
done
set -e
exit "$N2M_RC"
