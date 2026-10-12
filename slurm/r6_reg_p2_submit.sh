#!/usr/bin/env bash
# R6 REG-P2 driver (login node, after the protocol is frozen: nohup setsid bash slurm/r6_reg_p2_submit.sh >/dev/null 2>&1 &  — stop by PID).
# Submits slurm/r6_reg_p2_feed.txt.  No afterok chains (a finished dependency is purged after MinJobAge and sbatch then rejects it — the REG-P1
# driver failure of 2026-10-11): registrations wait for perturbations.json; critic / CoMIR registrations wait for REG-P1's train.json / comir.json
# of the same run; the aggregate waits until every registration job has left the queue.  Each submission waits below CAP; an empty job id, a
# perturb job that ends without its file, or a REG-P1 file still missing after WAIT_H hours stops the driver (logged).  COMIR_WTE=0 skips the
# CoMIR warp-then-encode sensitivity row (queued last).
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; CAP=${CAP:-27}; WAIT_H=${WAIT_H:-72}
O=/home/infres/yinwang/CS_QMI/outputs; P1=${P1_ROOT:-$O/R6_REG}; P2=${P2_ROOT:-$O/R6_REG_P2}; LOG=${P2_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r6_reg_p2_submit.log}; IDS=""
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
sub() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep 120; done
  ID=$(sbatch --parsable "$@") || ID=""; [ -n "$ID" ] || { say "STOP: sbatch failed: $*"; exit 1; }; say "$ID $*"; }
inq() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
gate() { local n=0; until [ -f "$1" ]; do sleep 300; n=$((n + 1)); [ "$n" -ge $((WAIT_H * 12)) ] && { say "STOP: $1 missing after $WAIT_H h"; exit 1; }; done; }
say "start (REG-P1 root $P1, CAP $CAP)"; [ -f "$P1/manifest.json" ] || { say "STOP: REG-P1 manifest missing"; exit 1; }
sub --job-name=r6reg2_perturb slurm/r6_reg_p2_cpu.sbatch perturb; while inq "$ID"; do sleep 60; done
[ -f "$P2/perturbations.json" ] || { say "STOP: perturb job $ID ended without perturbations.json"; exit 1; }
for m in mi nmi; do sub --job-name=r6reg2_reg_$m slurm/r6_reg_p2_gpu.sbatch register --method $m; IDS="$IDS $ID"; done
sub --job-name=r6reg2_reg_kernel slurm/r6_reg_p2_gpu.sbatch register --method kernel --seed 0; IDS="$IDS $ID"
for s in 0 1 2; do
  for o in vcs logistic infonce; do
    gate "$P1/runs/${o}_s$s/train.json"; sub --job-name=r6reg2_reg_${o}_s$s slurm/r6_reg_p2_gpu.sbatch register --method $o --seed $s; IDS="$IDS $ID"
  done
  gate "$P1/runs/comir_s$s/comir.json"; sub --job-name=r6reg2_reg_comir_s$s slurm/r6_reg_p2_gpu.sbatch register --method comir --seed $s; IDS="$IDS $ID"
done
if [ "${COMIR_WTE:-1}" = 1 ]; then                                     # sensitivity row (DEV only), queued last; about 1.5 h per seed on one GPU
  for s in 0 1 2; do sub --job-name=r6reg2_reg_comirwte_s$s --time=12:00:00 slurm/r6_reg_p2_gpu.sbatch register --method comir_wte --seed $s; IDS="$IDS $ID"; done
fi
for j in $IDS; do while inq "$j"; do sleep 300; done; done
sub --job-name=r6reg2_aggregate slurm/r6_reg_p2_aggregate_cpu.sbatch; say "done; aggregate $ID"
