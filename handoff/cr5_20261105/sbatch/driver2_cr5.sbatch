#!/usr/bin/env bash
# CR5 follow-up driver: driver 1089911 stopped when the 4th group submission hit QOSMaxSubmitJobPerUserLimit.
# Retries that one submission until accepted, then waits for all 4 groups and summarizes. No --dependency.
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr5_20261105/logs/%x_%j.out
set -uo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr5_20261105/sbatch/common_env.sh
T=$CR5_TREE; JS=$1; log(){ echo "$(date -u +%FT%TZ) $*"; }
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
cd $T/sbatch
until j=$(sbatch --parsable --job-name=cr5_CC152K_RE_s1003 group_cr5.sbatch CC152K_RECOVERED_20261104 1003 2>/dev/null); do sleep 120; done
printf "$j\tcr5_CC152K_RE_s1003\tGROUP\tGPU-any\t$(date -u +%FT%TZ)\tdriver2: matched group CC152K_RECOVERED_20261104 seed 1003 (retried after submit limit)\n" >> $T/SUBMITTED_CR5.tsv
JS="$JS,$j"; log "submitted $j; waiting for $JS"
while squeue -h -j "$JS" 2>/dev/null | grep -q .; do sleep 60; done
log "groups complete: $(grep -h GROUP_COMPLETE $T/logs/cr5_*_s100*_*.out | wc -l)/4"
cd $T/code && $CR5_PY -I -B site/summarize_cr5.py > $T/logs/summarize_cr5_last.txt 2>&1 || log "summarize failed"
log "DRIVER2_DONE"; echo "EXIT_OK $(date -u +%FT%TZ)"
