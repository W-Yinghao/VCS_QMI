#!/usr/bin/env bash
# CR4 driver (CPU): acceptance ALL_PASS -> freeze -> audit(before) + 3 group jobs -> audit(after) -> summarize. No --dependency; stops on any failure.
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr4_20261104/sbatch/common_env.sh
T=$CR4_TREE; ACC_JOB=$1; log(){ echo "$(date -u +%FT%TZ) $*"; }
sub(){ local name=$1 stage=$2 note=$3; shift 3; local j; j=$(sbatch --parsable --job-name=$name "$@"); printf "$j\t$name\t$stage\t-\t$(date -u +%FT%TZ)\tdriver: $note\n" >> $T/SUBMITTED_CR4.tsv; echo $j; }
waitjobs(){ while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 60; done; }
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
waitjobs $ACC_JOB
ACC=$T/acceptance/ACCEPTANCE_CR4_${ACC_JOB}.json
[ -f $ACC ] || { log "acceptance file missing"; exit 2; }
$CR4_PY -I -c "import json,sys;sys.exit(0 if json.load(open('$ACC'))['all_pass'] else 3)" || { log "ACCEPTANCE NOT ALL_PASS: stop"; exit 3; }
[ -f $T/PROTOCOL_FROZEN_CR4.json ] || $CR4_PY -I -B site/freeze_cr4.py $ACC
log "frozen $(sha256sum $T/PROTOCOL_FROZEN_CR4.json)"
cd $T/sbatch
AB=$(sub cr4_audit_before AUDIT_BEFORE "read-only CE decomposition/grad audit before training" --time=02:00:00 gpu_py.sbatch '$CR4_PY -I -B site/audit_cr4.py --when before')
G1=$(sub cr4_Flickr60 GROUP "matched group Flickr60" group_cr4.sbatch Flickr60)
G2=$(sub cr4_CC152K_RE GROUP "matched group CC152K_RECOVERED_20261104" group_cr4.sbatch CC152K_RECOVERED_20261104)
G3=$(sub cr4_Flickr00 GROUP "matched group Flickr00" group_cr4.sbatch Flickr00)
log "submitted audit-before $AB groups $G1 $G2 $G3"
waitjobs "$G1,$G2,$G3"
for g in Flickr60 CC152K_RECOVERED_20261104 Flickr00; do grep -q "GROUP_COMPLETE $g" $T/logs/cr4_*_*.out || { log "group $g not complete: stop"; exit 4; }; done
AA=$(sub cr4_audit_after AUDIT_AFTER "read-only CE decomposition/grad audit after training" --time=02:00:00 gpu_py.sbatch '$CR4_PY -I -B site/audit_cr4.py --when after')
waitjobs "$AB,$AA"
cd $T/code && $CR4_PY -I -B site/summarize_cr4.py > $T/logs/summarize_cr4_last.txt 2>&1 || true
log "DRIVER_DONE"
echo "EXIT_OK $(date -u +%FT%TZ)"
