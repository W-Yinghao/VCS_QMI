#!/usr/bin/env bash
# V10 job submission under the standing queue rules, with a code pin.
#   slurm/v10/submit.sh slurm/v10/<job>.sbatch [extra sbatch options, e.g. -p L40S,RTX6000PRO,H100]
# Refuses when: code paths have uncommitted changes; the script is not tracked; V10 already holds 2 queued jobs; the normal-QOS
# count is 29 or more (a new job would exceed 29; MaxSubmitJobsPU is 30).  Exports V10_EXPECT_CODE (hash of src/v10 + scripts/v10
# + slurm/v10) and V10_SUBMIT_COMMIT; slurm/v10/v10_header.sh refuses to run if that code changed before the job starts.
# Moving a PENDING V10 job to other GPU types: scontrol update JobId=<own pending job> Partition=L40S,RTX6000PRO,H100
# (own job only) - never submit a third V10 job instead.
set -euo pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10
script=${1:?usage: slurm/v10/submit.sh slurm/v10/<job>.sbatch [sbatch options]}
shift
cd "$R"
dirty=$(git status --porcelain -- src/v10 scripts/v10 slurm/v10 tests/v10 src/v9/reg | wc -l)
if [ "$dirty" -ne 0 ]; then
  echo "[submit] REFUSED: uncommitted changes:"; git status --porcelain -- src/v10 scripts/v10 slurm/v10 tests/v10 src/v9/reg; exit 1
fi
git ls-files --error-unmatch "$script" >/dev/null 2>&1 || { echo "[submit] REFUSED: $script is not tracked"; exit 1; }
n_normal=$(squeue -u yinwang -h -o %q | grep -c normal || true)
n_v10=$(squeue -u yinwang -h -o %j | grep -c '^v10_' || true)
echo "[submit] $(date -u +%FT%TZ) normal-QOS jobs=${n_normal} V10 jobs=${n_v10}"
if [ "$n_v10" -ge 2 ]; then echo "[submit] REFUSED: V10 already holds ${n_v10} queued jobs (max 2)"; exit 1; fi
if [ "$n_normal" -ge 29 ]; then echo "[submit] REFUSED: normal-QOS count ${n_normal}; a new job would exceed 29"; exit 1; fi
pin=$(git ls-files src/v10 scripts/v10 slurm/v10 | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
commit=$(git rev-parse HEAD)
echo "[submit] commit=${commit} pin=${pin} script=${script} extra=$*"
sbatch --export=ALL,V10_EXPECT_CODE="${pin}",V10_SUBMIT_COMMIT="${commit}" "$@" "$script"
