#!/usr/bin/env bash
# feeds the VL3 add. 1 remainder once the Large smoke (job 1032663) has finished with exit=0; aborts if it failed
L=/home/infres/yinwang/CS_QMI/slurm_logs; f=$L/vl3_smoke_large_1032663.out
until [ -f $f ] && grep -q "^exit=" $f; do sleep 60; done
grep -q "^exit=0" $f || { echo "$(date -u +%FT%TZ) VL3 add. 1 smoke failed - remainder NOT fed" >> $L/feed_normal.log; exit 1; }
cd /home/infres/yinwang/CS_QMI/ssl_pilot && bash slurm/feed_normal.sh slurm/vl3_add1_stage1_rest.txt
