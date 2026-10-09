#!/usr/bin/env bash
# waits for the VL1-15 downloads, then feeds the GPU jobs
L=/home/infres/yinwang/CS_QMI/slurm_logs
until [ "$(grep -c '^done' $L/vl1_15_downloads.log 2>/dev/null)" = 3 ] && grep -q '^ok' $L/vl1_15_clipL_download.log 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) VL1-15 downloads complete, starting feeder" >> $L/feed_normal.log
cd /home/infres/yinwang/CS_QMI/ssl_pilot && bash slurm/feed_normal.sh slurm/vl1_15_feed.txt
