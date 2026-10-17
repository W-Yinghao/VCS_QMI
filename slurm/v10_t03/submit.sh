#!/usr/bin/env bash
# V10 T03 job submission under the standing queue rules, with a code pin (the T03 equivalent of slurm/v10/submit.sh; T03 runs in its
# own worktree /home/infres/yinwang/CS_QMI/ssl_pilot_v10_t03, branch v10/t03, decision 2026-10-17 section 10).
#   slurm/v10_t03/submit.sh slurm/v10_t03/<job>.sbatch [extra sbatch options] [-- <script arguments>]
#   e.g. slurm/v10_t03/submit.sh slurm/v10_t03/t03_smoke.sbatch
# Refuses when: pinned paths have uncommitted changes; the script is not tracked; ALL V10 jobs together (job names starting with v10_,
# BF and T03 alike) already hold 2 queued jobs; the normal-QOS count is 29 or more (a new job would exceed 29).  Exports
# V10_T03_EXPECT_CODE (pin = src/v10 + scripts/v10_t03 + slurm/v10_t03 + the T03 inputs) and V10_T03_SUBMIT_COMMIT;
# slurm/v10_t03/t03_header.sh refuses to run if any of it changed before the job starts.
# Arguments after "--" go to the batch script; everything before goes to sbatch.  Job names must start with v10_ (counted above).
set -euo pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_t03
PIN_PATHS="src/v10 scripts/v10_t03 slurm/v10_t03 v10/protocol/T03_UNITS.csv v10/package_v10_server_v1_20261016/synthetic/src/v10_synthetic.py v10/package_v10_server_v1_20261016/synthetic/tests/check_synthetic.py"
DIRTY_PATHS="$PIN_PATHS tests/v10_t03 src/v9/reg"
script=${1:?usage: slurm/v10_t03/submit.sh slurm/v10_t03/<job>.sbatch [sbatch options] [-- script arguments]}
shift
opts=()
sargs=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; sargs=("$@"); break; fi
  opts+=("$1"); shift
done
cd "$R"
# shellcheck disable=SC2086
dirty=$(git status --porcelain -- $DIRTY_PATHS | wc -l)
if [ "$dirty" -ne 0 ]; then
  echo "[submit_t03] REFUSED: uncommitted changes:"; git status --porcelain -- $DIRTY_PATHS; exit 1
fi
git ls-files --error-unmatch "$script" >/dev/null 2>&1 || { echo "[submit_t03] REFUSED: $script is not tracked"; exit 1; }
name=$(sed -n 's/^#SBATCH --job-name=//p' "$script" | head -1)
case "$name" in v10_*) ;; *) echo "[submit_t03] REFUSED: job name '$name' must start with v10_ (V10 queue count)"; exit 1 ;; esac
n_normal=$(squeue -u yinwang -h -o %q | grep -c normal || true)
n_v10=$(squeue -u yinwang -h -o %j | grep -c '^v10_' || true)
echo "[submit_t03] $(date -u +%FT%TZ) normal-QOS jobs=${n_normal} V10 jobs (all v10_*)=${n_v10}"
if [ "$n_v10" -ge 2 ]; then echo "[submit_t03] REFUSED: V10 already holds ${n_v10} queued jobs (max 2, BF and T03 together)"; exit 1; fi
if [ "$n_normal" -ge 29 ]; then echo "[submit_t03] REFUSED: normal-QOS count ${n_normal}; a new job would exceed 29"; exit 1; fi
# shellcheck disable=SC2086
pin=$(git ls-files $PIN_PATHS | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
commit=$(git rev-parse HEAD)
echo "[submit_t03] commit=${commit} pin=${pin} script=${script} sbatch_options=${opts[*]:-} script_args=${sargs[*]:-}"
sbatch --export=ALL,V10_T03_EXPECT_CODE="${pin}",V10_T03_SUBMIT_COMMIT="${commit}" ${opts[@]+"${opts[@]}"} "$script" ${sargs[@]+"${sargs[@]}"}
