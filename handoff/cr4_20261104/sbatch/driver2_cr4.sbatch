#!/usr/bin/env bash
# CR4_A1 post driver (CPU): wait for the 3 group jobs -> audit(after) -> summarize. No --dependency; stops on failure.
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr4_20261104/sbatch/common_env.sh
T=$CR4_TREE; GJ=$1; log(){ echo "$(date -u +%FT%TZ) $*"; }
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
while squeue -h -j "$GJ" 2>/dev/null | grep -q .; do sleep 60; done
for c in Flickr60 CC152K_RECOVERED_20261104 Flickr00; do grep -qh "GROUP_COMPLETE $c" $T/logs/cr4a1_*.out || { log "group $c not complete: stop"; exit 4; }; done
cd $T/sbatch
AA=$(sbatch --parsable --job-name=cr4_audit_after --nice=0 --time=02:00:00 gpu_py.sbatch '$CR4_PY -I -B site/audit_cr4.py --when after')
printf "$AA\tcr4_audit_after\tAUDIT_AFTER\t-\t$(date -u +%FT%TZ)\tdriver2: read-only audit after training\n" >> $T/SUBMITTED_CR4.tsv
while squeue -h -j "$AA" 2>/dev/null | grep -q .; do sleep 60; done
cd $T/code && $CR4_PY -I -B site/summarize_cr4.py > $T/logs/summarize_cr4_last.txt 2>&1 || log "summarize failed"
log "DRIVER2_DONE"; echo "EXIT_OK $(date -u +%FT%TZ)"
