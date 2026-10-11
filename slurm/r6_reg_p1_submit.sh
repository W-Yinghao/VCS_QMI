#!/usr/bin/env bash
# R6 REG-P1 full-pilot driver (login node; run: nohup setsid bash slurm/r6_reg_p1_submit.sh >/dev/null 2>&1 &  — stop by PID).  Submits every job of
# slurm/r6_reg_p1_feed.txt in dependency order (afterok), waiting before each submission until my normal-QOS job count is below CAP (27).
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; CAP=${CAP:-27}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/r6_reg_p1_submit.log
sub() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep 120; done; id=$(sbatch --parsable "$@"); echo "$(date -u +%FT%TZ) $id $*" >> "$LOG"; echo "$id"; }
dep() { echo "--dependency=afterok:$1 --kill-on-invalid-dep=yes"; }
P1=$(sub --job-name=r6reg_prepare slurm/r6_reg_cpu.sbatch prepare); P2=$(sub $(dep $P1) --job-name=r6reg_perturb slurm/r6_reg_cpu.sbatch perturb)
REGS=""
for m in mi nmi; do REGS="$REGS:$(sub $(dep $P2) --job-name=r6reg_reg_$m slurm/r6_reg_gpu.sbatch register --method $m)"; done
REGS="$REGS:$(sub $(dep $P2) --job-name=r6reg_reg_kernel slurm/r6_reg_gpu.sbatch register --method kernel --seed 0)"
for s in 0 1 2; do
  for o in vcs logistic infonce; do T=$(sub $(dep $P2) --job-name=r6reg_train_${o}_s$s slurm/r6_reg_gpu.sbatch train --objective $o --seed $s)
    REGS="$REGS:$(sub $(dep $T) --job-name=r6reg_reg_${o}_s$s slurm/r6_reg_gpu.sbatch register --method $o --seed $s)"; done
  C=$(sub $(dep $P2) --job-name=r6reg_comir_s$s slurm/r6_reg_gpu.sbatch comir --seed $s)
  REGS="$REGS:$(sub $(dep $C) --job-name=r6reg_reg_comir_s$s slurm/r6_reg_gpu.sbatch register --method comir --seed $s)"
done
A=$(sub --dependency=afterok$REGS --kill-on-invalid-dep=yes slurm/r6_reg_aggregate_cpu.sbatch); echo "$(date -u +%FT%TZ) done; aggregate $A" >> "$LOG"
