#!/usr/bin/env bash
# VL3 add. 6: feed the other 11 jobs once the first (1033440) exits 0 with gate X passing for both targets.  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; L=/home/infres/yinwang/CS_QMI/slurm_logs; f=$L/vl3add6_refcocog_clip_b16_1033440.out
until [ -f $f ] && grep -q "^exit=" $f; do sleep 60; done
if grep -q "^exit=0" $f && ! grep -q "FAIL" $f && [ "$(grep -c 'gate X pass' $f)" -eq 2 ]; then bash slurm/feed_normal.sh slurm/vl3_add6_feed.txt
else echo "$(date -u +%FT%TZ) VL3 add. 6: first job 1033440 failed - not fed" >> $L/feed_normal.log; fi
