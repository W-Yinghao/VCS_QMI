#!/usr/bin/env bash
# One-off (coordinator, 2026-10-13; R7-50 note r7_50_7): when the D1-FULL driver logs its done line (all evaluation shards and its timing bundle
# complete), launch the frozen R7-50 driver for phase F with the D1-FULL paths.  Never launches while another R7-50 driver process exists.
L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/after_d1full_phaseF.log; DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }; say "start pid $$"
until grep -q "done: D1-FULL evaluation complete" $L/r7_d1full_submit.log 2>/dev/null; do sleep 300; done
say "D1-FULL done line present"
while ps -u "$USER" -o cmd= | grep -qE "^bash slurm/r7_50_submit\.sh"; do sleep 300; done
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
PHASE=F F_LR_CHOICE=$DF/selectors/lr_choice.json F_SELECTORS=$DF/selectors F_PCA=$DF/cache/pca.pt F_INTEGRITY_REF=$DF/d1_30/pred \
  F_TIMING_DIR=$DF/d1_30/timing nohup setsid bash slurm/r7_50_submit.sh >/dev/null 2>&1 &
say "launched the R7-50 phase-F driver"
