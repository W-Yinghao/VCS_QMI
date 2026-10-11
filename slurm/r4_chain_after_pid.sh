#!/usr/bin/env bash
# Wait until process $1 is gone, then feed lines file $2 with CAP=${CAP:-27}.  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; while kill -0 "$1" 2>/dev/null; do sleep 120; done; CAP=${CAP:-27} bash slurm/feed_normal.sh "$2"
