#!/usr/bin/env bash
# feeds VL3 add. 4 once the add. 3 feeder (PID given) has submitted all its lines
while kill -0 "$1" 2>/dev/null; do sleep 120; done
cd /home/infres/yinwang/CS_QMI/ssl_pilot && bash slurm/feed_normal.sh slurm/vl3_add4_feed.txt
