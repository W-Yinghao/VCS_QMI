#!/usr/bin/env bash
#SBATCH --job-name=v10e_audit
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 EPFL activation gates 1-4 (decision 2026-10-17 section 6.2 items 1-4): CPU only, no GPU, no registration.
# Reads all 954 EPFL files on the V10 path (required by item 1), the REG-P3 manifest / cache / audit and the V9-G2 unit records; writes only
# under V10_EPFL_OUT (records + dataset-derived images; images never enter git).  Submit from a clean, committed tree:
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl && V10_EPFL_EXPECT_COMMIT=$(git rev-parse HEAD) sbatch --export=ALL slurm/v10_epfl/audit_cpu.sbatch [run --gates 1,2,3,4 | sheets [--extra-pairs F]]
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
OUT=${V10_EPFL_OUT:-/projects/EEG-foundation-model/yinghao/outputs_v10/epfl_activation}
commit=$(git -C "$R" rev-parse HEAD)
dirty=$(git -C "$R" status --porcelain -- src/v10/epfl scripts/v10_epfl slurm/v10_epfl src/v9/reg | wc -l)
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} commit=${commit} expect=${V10_EPFL_EXPECT_COMMIT:-UNSET} dirty=${dirty} out=${OUT}"
if [ "$dirty" -ne 0 ]; then echo "[v10e] REFUSED: uncommitted changes under the audit code"; git -C "$R" status --porcelain -- src/v10/epfl scripts/v10_epfl slurm/v10_epfl src/v9/reg; exit 1; fi
if [ -z "${V10_EPFL_EXPECT_COMMIT:-}" ] || [ "$commit" != "$V10_EPFL_EXPECT_COMMIT" ]; then echo "[v10e] REFUSED: HEAD ${commit} != submitted ${V10_EPFL_EXPECT_COMMIT:-UNSET}"; exit 1; fi
cd "$R"
set +e
cmd=${1:-run}; [ $# -gt 0 ] && shift
"$PYTHON" -I scripts/v10_epfl/audit.py "$cmd" --out "$OUT" "$@"
rc=$?
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
