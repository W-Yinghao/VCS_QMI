#!/usr/bin/env bash
# One-off (coordinator, 2026-10-13; R7-50 notes r7_50_7 / r7_50_8): when D1-FULL's D1-30-style evaluation predictions for gqa_dev (6 shard markers)
# and chartqa_val (2 shard markers) are complete — the phase-F integrity gate's references (decision F2) — launch the frozen R7-50 driver for
# phase F with the D1-FULL paths.  D1-FULL's timing files are gated by the driver itself just before the manifest (note r7_50_8).  Never
# launches while another R7-50 driver process exists.
L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/after_d1full_phaseF.log; DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL; P=$DF/d1_30/pred
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$ (trigger: D1-FULL gqa_dev 6/6 + chartqa_val 2/2 markers)"
until [ "$(ls $P/gqa_dev/done_shard*of6.json 2>/dev/null | wc -l)" -ge 6 ] && [ "$(ls $P/chartqa_val/done_shard*of2.json 2>/dev/null | wc -l)" -ge 2 ]; do sleep 300; done
say "D1-FULL gqa_dev / chartqa_val evaluation complete"
while ps -u "$USER" -o cmd= | grep -qE "^bash slurm/r7_50_submit\.sh"; do sleep 300; done
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
PHASE=F F_LR_CHOICE=$DF/selectors/lr_choice.json F_SELECTORS=$DF/selectors F_PCA=$DF/cache/pca.pt F_INTEGRITY_REF=$P \
  F_TIMING_DIR=$DF/d1_30/timing nohup setsid bash slurm/r7_50_submit.sh >/dev/null 2>&1 &
say "launched the R7-50 phase-F driver"
