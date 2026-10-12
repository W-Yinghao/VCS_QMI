#!/usr/bin/env bash
# R6 REG-P3 driver (login node, after the protocol is frozen: nohup setsid bash slurm/r6_reg_p3_submit.sh >/dev/null 2>&1 &  — stop by PID).
# Submits slurm/r6_reg_p3_feed.txt.  No afterok chains (a finished dependency is purged after MinJobAge and sbatch then rejects it — the REG-P1
# driver failure of 2026-10-11): every job is gated on the files it reads (training / perturb on manifest.json; registrations on perturbations.json
# and on the run's train.json / comir.json).  A job that leaves the queue without its file stops the driver at once, as do a file still missing
# after WAIT_H hours and an empty job id (all logged).  Each submission waits until my normal-QOS job count is below CAP.  A unit whose output
# already exists is skipped, so the driver can be restarted after a stop.  Short units are bundled (the three critics of a seed in one training
# job, the three CoMIR seeds in one job, MI + NMI, the CoMIR registrations of the three seeds).  Data: EPFL RGB-NIR (prepare --dataset epfl; the
# Zurich path is not run).  The CoMIR warp-then-encode row is not run in REG-P3 (coordinator, 2026-10-12).  The aggregate runs once every
# registration job has left the queue.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; CAP=${CAP:-27}; WAIT_H=${WAIT_H:-72}
O=/home/infres/yinwang/CS_QMI/outputs; P3=${P3_ROOT:-$O/R6_REG_P3}; LOG=${P3_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r6_reg_p3_submit.log}; IDS=""; ID=""
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
sub() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep 120; done
  ID=$(sbatch --parsable "$@") || ID=""; [ -n "$ID" ] || { say "STOP: sbatch failed: $*"; exit 1; }; say "$ID $*"; }
inq() { [ -n "$1" ] && squeue -h -j "$1" 2>/dev/null | grep -q .; }
need() { local f=$1 j=$2 n=0                                      # wait for file f (written by job j); stop if j left the queue without it
  until [ -f "$f" ]; do
    if [ -n "$j" ] && ! inq "$j"; then sleep 60; [ -f "$f" ] && return 0; say "STOP: job $j ended without $f"; exit 1; fi
    sleep 120; n=$((n + 1)); [ "$n" -ge $((WAIT_H * 30)) ] && { say "STOP: $f missing after $WAIT_H h"; exit 1; }
  done; }
R="--root $P3"; say "start (root $P3, CAP $CAP, dataset epfl)"
# stage 0 (CPU): checks / exclusions / units / roles / windows / cache (≈ 11 min), then the rigid perturbations
if [ -f "$P3/manifest.json" ]; then say "skip prepare (manifest.json exists)"; else sub --job-name=r6reg3_prepare slurm/r6_reg_p3_cpu.sbatch prepare --dataset epfl $R; need "$P3/manifest.json" "$ID"; fi
PJ=""; if [ ! -f "$P3/perturbations.json" ]; then sub --job-name=r6reg3_perturb slurm/r6_reg_p3_cpu.sbatch perturb $R; PJ=$ID; fi
# stage 1 (GPU): training on FIT — the three critics of a seed in one job (REG-P1 recipe, 3 000 steps each); CoMIR seeds 0–2 in one job
declare -A TJ; for s in 0 1 2; do
  if [ -f "$P3/runs/vcs_s$s/train.json" ] && [ -f "$P3/runs/logistic_s$s/train.json" ] && [ -f "$P3/runs/infonce_s$s/train.json" ]; then say "skip training seed $s"; TJ[$s]=""
  else sub --job-name=r6reg3_train_s$s slurm/r6_reg_p3_gpu.sbatch train --objective vcs,logistic,infonce --seed $s $R; TJ[$s]=$ID; fi; done
CJ=""; if [ -f "$P3/runs/comir_s0/comir.json" ] && [ -f "$P3/runs/comir_s1/comir.json" ] && [ -f "$P3/runs/comir_s2/comir.json" ]; then say "skip CoMIR training"
else sub --job-name=r6reg3_comir slurm/r6_reg_p3_gpu.sbatch comir --seed 0,1,2 $R; CJ=$ID; fi
need "$P3/perturbations.json" "$PJ"
# stage 2 (GPU): registrations — deterministic references, then each critic run after its training, then CoMIR
reg() { local name=$1 out=$2; shift 2; if [ -f "$P3/results/$out" ]; then say "skip $name ($out exists)"; else sub --job-name=$name "$@"; IDS="$IDS $ID"; fi; }
reg r6reg3_reg_mi_nmi nmi.json slurm/r6_reg_p3_gpu.sbatch register --method mi,nmi $R
reg r6reg3_reg_kernel kernel_s0.json slurm/r6_reg_p3_gpu.sbatch register --method kernel --seed 0 $R
for s in 0 1 2; do for o in vcs logistic infonce; do
  need "$P3/runs/${o}_s$s/train.json" "${TJ[$s]}"; reg r6reg3_reg_${o}_s$s ${o}_s$s.json slurm/r6_reg_p3_gpu.sbatch register --method $o --seed $s $R; done; done
for s in 0 1 2; do need "$P3/runs/comir_s$s/comir.json" "$CJ"; done
reg r6reg3_reg_comir comir_s2.json slurm/r6_reg_p3_gpu.sbatch register --method comir --seed 0,1,2 $R
for j in $IDS; do while inq "$j"; do sleep 300; done; done
sub --job-name=r6reg3_aggregate slurm/r6_reg_p3_aggregate_cpu.sbatch $R; say "done; aggregate $ID"
