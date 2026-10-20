#!/usr/bin/env bash
# Submit one N2M GPU job through the package's own sbatch template, without editing the package:
#   submit_n2m.sh seed 901         -> scripts/run_seed.sbatch (40-epoch arms of that seed, frozen order)
#   submit_n2m.sh loss 901         -> scripts/export_losses.sbatch (fixed-student diagnostic, 8 batches)
#   submit_n2m.sh seed 901 1048432 -> same, held with --dependency=afterok:<prepare job> (automatic start after the freeze;
#                                     the experiment-exists check is replaced by "the prepare job is queued")
# Command-line overrides only: --job-name n2m_s<seed> / n2m_loss<seed> (no v9 prefix: the N2F closeout controllers
# wait while any v9* job is live), --partition from env.sh (one frozen environment for all arms), --output to slurm_logs.
# Refuses when: experiment.json is missing or not frozen (or, with a prepare job id, that job is not queued); a job of the
# same name is already queued/running; my normal-QOS jobs would exceed 29; my pending normal-QOS GPU jobs would exceed 8 or
# running + pending would exceed 16 (the project dispatch caps: pending 8 / live 16 / normal 29; Slurm itself runs at most 8).
set -euo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/server/env.sh
kind="${1:?seed|loss}"; seed="${2:?901|902|903}"; after="${3:-}"
case "$seed" in 901|902|903) ;; *) echo "bad seed $seed" >&2; exit 2 ;; esac
dep=()
if [ -n "$after" ]; then
  [ -n "$(squeue -h -j "$after" -o %j 2>/dev/null | grep -x n2m_prepare)" ] || { echo "REFUSED: prepare job $after is not queued" >&2; exit 3; }
  dep=(--dependency="afterok:$after")
else
  [ -f "$N2M_OUT/experiment.json" ] && [ -f "$N2M_OUT/EXPERIMENT.sha256" ] || { echo "REFUSED: experiment not frozen" >&2; exit 3; }
fi
case "$kind" in
  seed) name="n2m_s$seed"; tmpl="$N2M_PACKAGE/scripts/run_seed.sbatch"; args=("$N2M_PACKAGE" "$N2M_OUT/experiment.json" "$seed" "$N2M_PYTHON") ;;
  loss) name="n2m_loss$seed"; tmpl="$N2M_PACKAGE/scripts/export_losses.sbatch"
        csv="$N2M_OUT/diagnostics/s$seed/fixed_losses.csv"
        [ -e "$csv" ] && { echo "REFUSED: $csv exists" >&2; exit 3; }
        args=("$N2M_PACKAGE" "$N2M_OUT/experiment.json" "$seed" "$csv" "$N2M_PYTHON") ;;
  *) echo "bad kind $kind" >&2; exit 2 ;;
esac
q=$(squeue -u "$USER" -h -o "%j|%q|%b" 2>/dev/null) || { echo "REFUSED: squeue failed" >&2; exit 4; }
[ -n "$q" ] || { echo "REFUSED: empty squeue output (cannot be trusted)" >&2; exit 4; }
echo "$q" | cut -d'|' -f1 | grep -qx "$name" && { echo "REFUSED: $name already queued/running" >&2; exit 5; }
n_normal=$(echo "$q" | awk -F'|' '$2=="normal"' | wc -l)
n_gpu=$(echo "$q" | awk -F'|' '$2=="normal" && $3 ~ /gpu/' | wc -l)
n_gpu_pd=$(squeue -u "$USER" -h -t PD -o "%q|%b" | awk -F'|' '$1=="normal" && $2 ~ /gpu/' | wc -l)
[ "$n_normal" -lt 29 ] || { echo "REFUSED: normal-QOS jobs $n_normal >= 29" >&2; exit 6; }
[ "$n_gpu_pd" -lt 8 ] || { echo "REFUSED: pending normal-QOS GPU jobs $n_gpu_pd >= 8" >&2; exit 6; }
[ "$n_gpu" -lt 16 ] || { echo "REFUSED: normal-QOS GPU jobs (running + pending) $n_gpu >= 16" >&2; exit 6; }
mkdir -p "$N2M_OUT/submissions"
jid=$(sbatch --parsable "${dep[@]}" --job-name="$name" --partition="$N2M_PARTITION" --output="$N2M_LOGS/%x_%j.out" "$tmpl" "${args[@]}")
printf '{"utc":"%s","job":"%s","name":"%s","kind":"%s","seed":%s,"template":"%s","partition":"%s","afterok":"%s"}\n' \
  "$(date -u +%FT%TZ)" "$jid" "$name" "$kind" "$seed" "$tmpl" "$N2M_PARTITION" "$after" >> "$N2M_OUT/submissions/SUBMISSIONS.jsonl"
echo "$jid"
