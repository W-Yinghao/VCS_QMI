#!/usr/bin/env bash
# Submit P75 units (all, or the "METHOD SEED" pairs given as arguments, e.g. `submit_P75.sh vcs 0 simclr 0`).
# Verifies a numeric job id per submission (the shared 30-job cap silently rejects) and appends to reports/job_ids.json by hand.
cd /home/infres/yinwang/CS_QMI/ssl_pilot
if [ $# -gt 0 ]; then mapfile -t U < <(printf '%s %s\n' "$@" | xargs -n2); else mapfile -t U < <(grep -v '^#' s2_solo_learn/units_P75.txt); fi
for u in "${U[@]}"; do
  set -- $u; m=$1; s=$2
  id=$(sbatch --parsable ${NICE:+--nice=$NICE} --job-name="s2_solo_${m}_1000ep_seed${s}" --export=ALL,METHOD=$m,SEED=$s slurm/s2_solo_unit.sbatch)
  if [[ "$id" =~ ^[0-9]+$ ]]; then echo "$m seed$s -> $id"; else echo "$m seed$s REJECTED: $id"; fi
done
