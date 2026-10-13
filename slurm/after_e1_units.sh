#!/usr/bin/env bash
# One-off (coordinator, 2026-10-13; R7-E1 freeze: stage 2 + D-E1-7): when all 24 E1 unit markers exist (MMTok 12 + HiPrune 12) and no r7e1_ GPU
# job is queued, run the frozen completeness step (slurm/r7_e1_cpu.sbatch complete anchor) and, if it says complete, the frozen readout
# (slurm/r7_readout_e1_cpu.sbatch readout) once.  The readout refuses without the freeze key and never overwrites.
L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/after_e1_units.log; E=/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40/eval_e1; P=$E/pred
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$"
n_markers() { ls $P/gqa_dev/done_e1_*_shard*.json $P/textvqa_val/done_e1_*_shard*.json $P/chartqa_val/done_e1_*_shard*.json 2>/dev/null | wc -l; }
until [ "$(n_markers)" -ge 24 ] && [ -z "$(squeue -h -u "$USER" -o '%j' | grep '^r7e1_')" ]; do sleep 300; done
say "all 24 unit markers present, no E1 job queued"; cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
j=$(sbatch --parsable --wait --job-name=r7e1_complete slurm/r7_e1_cpu.sbatch complete anchor); say "completeness job ${j%%;*} ended (rc $?)"
for i in $(seq 1 24); do [ -f $E/completeness_e1.json ] && break; sleep 10; done
if grep -q '"complete": true' $E/completeness_e1.json 2>/dev/null; then
  j=$(sbatch --parsable --wait --job-name=r7e1_readout slurm/r7_readout_e1_cpu.sbatch readout); say "readout job ${j%%;*} ended (rc $?)"
else say "completeness not true: readout NOT run (see $E/completeness_e1.json)"; fi
say "done"
