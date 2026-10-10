#!/usr/bin/env bash
# VL3 add. 5: feed the other 11 jobs once the first (1033390) exits 0 with every gate S passing.  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; L=/home/infres/yinwang/CS_QMI/slurm_logs; f=$L/vl3add5_refcocog_clip_b16_1033390.out
until [ -f $f ] && grep -q "^exit=" $f; do sleep 60; done
if grep -q "^exit=0" $f && ! grep -q "FAIL" $f && [ "$(grep -c 'gate S.*pass' $f)" -eq 9 ]; then bash slurm/feed_normal.sh slurm/vl3_add5_feed.txt
else echo "$(date -u +%FT%TZ) VL3 add. 5: first job 1033390 failed - not fed" >> $L/feed_normal.log; fi
