#!/usr/bin/env bash
# CR5 driver (CPU): acceptance ALL_PASS -> freeze -> dry-run -> 4 group jobs -> summarize. No --dependency; stops on failure.
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr5_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr5_20261105/sbatch/common_env.sh
T=$CR5_TREE; ACC=$1; log(){ echo "$(date -u +%FT%TZ) $*"; }
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
while squeue -h -j $ACC 2>/dev/null | grep -q .; do sleep 60; done
AF=$T/acceptance/ACCEPTANCE_CR5_${ACC}.json; [ -f $AF ] || { log "acceptance file missing: stop"; exit 2; }
$CR5_PY -I -c "import json,sys;sys.exit(0 if json.load(open('$AF'))['all_pass'] else 3)" || { log "ACCEPTANCE NOT ALL_PASS: stop"; exit 3; }
[ -f $T/PROTOCOL_FROZEN_CR5.json ] || $CR5_PY -I -B site/freeze_cr5.py $AF
log "frozen $(sha256sum $T/PROTOCOL_FROZEN_CR5.json)"
for c in Flickr60 CC152K_RECOVERED_20261104; do for s in 1002 1003; do $CR5_PY -I -B site/cr5_group.py --condition $c --seed $s --dry-run; done; done
cd $T/sbatch; JS=""
for c in Flickr60 CC152K_RECOVERED_20261104; do for s in 1002 1003; do
  j=$(sbatch --parsable --job-name=cr5_${c:0:9}_s$s group_cr5.sbatch $c $s); JS="$JS,$j"
  printf "$j\tcr5_${c:0:9}_s$s\tGROUP\tGPU-any\t$(date -u +%FT%TZ)\tdriver: matched group $c seed $s\n" >> $T/SUBMITTED_CR5.tsv
done; done; JS=${JS#,}; log "submitted groups $JS"
while squeue -h -j "$JS" 2>/dev/null | grep -q .; do sleep 60; done
n=$(grep -h "GROUP_COMPLETE" $T/logs/cr5_*_s100*_*.out | wc -l); log "groups complete: $n/4"
cd $T/code && $CR5_PY -I -B site/summarize_cr5.py > $T/logs/summarize_cr5_last.txt 2>&1 || log "summarize failed"
log "DRIVER_DONE"; echo "EXIT_OK $(date -u +%FT%TZ)"
