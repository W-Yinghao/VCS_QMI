#!/usr/bin/env bash
# feeds VL3 add. 2 once the smoke 1032907 finished with exit=0 and wrote the suffixed output
L=/home/infres/yinwang/CS_QMI/slurm_logs; f=$L/vl3_smoke_e20_1032907.out
until [ -f $f ] && grep -q "^exit=" $f; do sleep 60; done
grep -q "^exit=0" $f && [ -f /home/infres/yinwang/CS_QMI/outputs/VL3/refcocog_clip_b16_vcs_s0_e20_smoke.json ] || { echo "$(date -u +%FT%TZ) VL3 add. 2 smoke failed - not fed" >> $L/feed_normal.log; exit 1; }
cd /home/infres/yinwang/CS_QMI/ssl_pilot && bash slurm/feed_normal.sh slurm/vl3_add2_feed.txt
