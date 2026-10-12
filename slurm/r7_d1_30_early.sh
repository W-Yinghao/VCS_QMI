#!/usr/bin/env bash
# Execution note (2026-10-12, coordinator): before D1-20 completes, run the D1-30 rows that do not depend on the selectors with the FROZEN
# code: (1) the non-selector quality rows (full, full_ocr, grid / random / qsim at 4 budgets), one job per bench with shard 0/1 (a denominator
# the driver never uses, so its own markers and all selector rows are unaffected; one writer per prediction file); (2) the driver's own 7
# SparseVLM lines, chained per bench (afterok) so only one shard writes a bench's files at a time — the driver later skips them by marker.
# Timing and DEV risk stay in the driver.  Waits for the sets build (job $1).  Stop by PID.  Log: slurm_logs/r7_d1_30_early.log.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; B=$1; CAP=${CAP:-27}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_30_early.log
O=/home/infres/yinwang/CS_QMI/outputs/R7_D1/d1_30; say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }; say "start pid $$ build $B"
while squeue -h -j "$B" 2>/dev/null | grep -q .; do sleep 30; done
[ -f "$O/sets/d1_30_sets.json" ] || { say "STOP: sets/d1_30_sets.json missing after build $B"; exit 1; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep 60; done; }
sub() { waitcap; ID=$(sbatch --parsable "$@") || ID=""; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id: $*"; exit 1; }; say "$ID $*"; }
ENVX="cd /home/infres/yinwang/CS_QMI/ssl_pilot && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 && /projects/EEG-foundation-model/yinghao/envs/sparsevlm/bin/python scripts/r7_d1_30_run.py --sets-dir $O/sets"
for b in gqa_dev textvqa_val chartqa_val; do
  sub --nice=500 --job-name=r7d30e_q_${b} --partition=H100,L40S --exclude=node51,node52,node60 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
      --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out --wrap "$ENVX --bench $b --configs-file $O/configs_early/nonselector_${b}.txt --shard 0/1 --out $O/pred; echo exit=\$?"
done
for b in gqa_dev textvqa_val; do
  prev=""; for s in 0/3 1/3 2/3; do dep=""; [ -n "$prev" ] && dep="--dependency=afterok:$prev --kill-on-invalid-dep=yes"
    sub --nice=500 $dep --job-name=r7d30v_${b}_${s%%/*} --partition=H100,L40S --exclude=node51,node52,node60 slurm/r7_d1_30_gpu.sbatch sparsevlm $b $s; prev=$ID; done
done
sub --nice=500 --job-name=r7d30v_chartqa_val_0 --partition=H100,L40S --exclude=node51,node52,node60 slurm/r7_d1_30_gpu.sbatch sparsevlm chartqa_val 0/1
say "done: early D1-30 rows submitted"
