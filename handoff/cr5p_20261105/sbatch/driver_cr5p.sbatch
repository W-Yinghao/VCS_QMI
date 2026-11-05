#!/usr/bin/env bash
# CR5_P1105 driver (CPU): package checks (scratch) -> freeze -> dry-run -> submit 4 group jobs with retry (per-user submit limit) -> wait -> summarize.
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr5p_20261105/logs/%x_%j.out
set -uo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr5p_20261105/sbatch/common_env.sh
T=$CR5P_TREE; log(){ echo "$(date -u +%FT%TZ) $*"; }
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
S=$CR5P_OUT/pkgcheck_$SLURM_JOB_ID; mkdir -p $S; cp -a $T/package/. $S/
( cd $S && export PATH=$(dirname $CR5P_PY):$PATH && bash run_checks.sh ) > $S/RUN_CHECKS.log 2>&1 && log "package run_checks PASS: $(grep -E 'passed|PASS' $S/RUN_CHECKS.log | tail -2 | tr '\n' ' ')" || { log "package run_checks FAILED (see $S/RUN_CHECKS.log): stop"; exit 2; }
[ -f $T/PROTOCOL_FROZEN_CR5_P1105.json ] || $CR5P_PY -I -B site/freeze_cr5p.py || { log "freeze failed: stop"; exit 3; }
log "frozen $(sha256sum $T/PROTOCOL_FROZEN_CR5_P1105.json)"
for c in Flickr60 CC152K_RECOVERED_20261104; do for s in 2002 2003; do $CR5P_PY -I -B site/cr5p_group.py --condition $c --seed $s --dry-run || { log "dry-run failed: stop"; exit 4; }; done; done
cd $T/sbatch; JS=""
for c in Flickr60 CC152K_RECOVERED_20261104; do for s in 2002 2003; do
  until j=$(sbatch --parsable --job-name=cr5p_${c:0:9}_t$s group_cr5p.sbatch $c $s 2>/dev/null); do sleep 120; done
  JS="$JS,$j"; printf "$j\tcr5p_${c:0:9}_t$s\tGROUP\tGPU-any\t$(date -u +%FT%TZ)\tdriver: matched group $c student seed $s\n" >> $T/SUBMITTED_CR5P.tsv; log "submitted $j $c $s"
done; done; JS=${JS#,}
while squeue -h -j "$JS" 2>/dev/null | grep -q .; do sleep 120; done
log "groups complete: $(grep -h GROUP_COMPLETE $T/logs/cr5p_*_t200*_*.out | wc -l)/4"
cd $T/code && $CR5P_PY -I -B site/summarize_cr5p.py > $T/logs/summarize_cr5p_last.txt 2>&1 || log "summarize failed"
log "DRIVER_DONE"; echo "EXIT_OK $(date -u +%FT%TZ)"
