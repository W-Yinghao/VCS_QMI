#!/usr/bin/env bash
#SBATCH --job-name=v81_fork3_paramaudit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8.1 fork 3 parameter-level audit (scripts/v81/v81_fork3_param_audit.py): RECONSTRUCTED_INIT per seed vs best.pt / resume.pt of the 24 formal units,
# frozen-tensor identity, AdamW step counts vs iterations (GradScaler skips), scaler state.  CPU only, read-only on outputs_v8.
#   sbatch slurm/v81/v81_fork3_param_audit.sbatch [--units V8-FIT-U-CS-s0 ...]
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v81 || exit 2; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SNAP=$(mktemp -d /tmp/v81_paramaudit_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v81 $SNAP/scripts/
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v81/v81_fork3_param_audit.py $SNAP/scripts/v81/v8_src/v8/*.py $SNAP/scripts/v81/v8_src/v7/*.py | sha256sum | cut -c1-16) args='$*'"
$PY -m pytest -q -p no:cacheprovider tests/v81/test_fork3.py 2>&1 | tail -3; trc=${PIPESTATUS[0]}; echo "tests_rc=$trc"
[ $trc -ne 0 ] && { echo "exit=$trc end=$(date -u +%FT%TZ)"; exit $trc; }
OUT=/home/infres/yinwang/CS_QMI/ssl_pilot_v81/reports/V8_1_DIAGNOSTICS/results/param_audit.json
[ $# -gt 0 ] && OUT=/projects/EEG-foundation-model/yinghao/outputs_v81/fork3_param_audit_partial_${SLURM_JOB_ID}.json
if [ $# -gt 0 ]; then ARGS=(--units "$@"); else ARGS=(); fi
$PY $SNAP/scripts/v81/v81_fork3_param_audit.py --out $OUT "${ARGS[@]}"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
