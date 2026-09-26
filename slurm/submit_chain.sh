#!/usr/bin/env bash
# Submit one unit as a chain of N dependent run_unit jobs (afterany), so a run longer than the 23 h walltime resumes from last.pt.
# Each link resumes if the previous one was stopped at the walltime (STOPPED_BUDGET) and only re-evaluates if training is already COMPLETED.
# usage: slurm/submit_chain.sh <run_id> <config> <final_ckpt> <n_links>   -> prints the job ids (space separated)
set -euo pipefail
RUN_ID=$1; CFG=$2; FINAL=$3; N=${4:-2}
REPO=/home/infres/yinwang/CS_QMI/ssl_pilot
# owner (2026-09-26): runs that need more than one 23 h walltime go to the fastest GPUs first (H100 / RTX6000PRO); override with PARTITION=...
PART=${PARTITION:-H100,RTX6000PRO}
ids=(); dep=""
for i in $(seq 1 "$N"); do
  name="${RUN_ID#P*_vcs_}"; [ "$i" -gt 1 ] && name="${name}_c$i"
  if [ -z "$dep" ]; then
    id=$(sbatch --parsable --partition="$PART" --job-name="$name" --export=ALL,CFG="$CFG",RUN_ID="$RUN_ID",FINAL_CKPT="$FINAL" "$REPO/slurm/run_unit.sbatch" | cut -d';' -f1)
  else
    id=$(sbatch --parsable --partition="$PART" --dependency=afterany:$dep --job-name="$name" --export=ALL,CFG="$CFG",RUN_ID="$RUN_ID",FINAL_CKPT="$FINAL" "$REPO/slurm/run_unit.sbatch" | cut -d';' -f1)
  fi
  ids+=("$id"); dep=$id
done
echo "${ids[*]}"
