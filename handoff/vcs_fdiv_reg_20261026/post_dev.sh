#!/usr/bin/env bash
# Run the frozen analysis once all 5 DEV jobs have COMPLETE records (touch STOP_POST_DEV to stop).
D=/home/infres/yinwang/CS_QMI/handoff/vcs_fdiv_reg_20261026; PY=/home/infres/yinwang/CS_QMI/env/bin/python
until [ "$(ls $D/runs/dev_*/COMPLETE.json 2>/dev/null | wc -l)" -eq 5 ]; do [ -e $D/STOP_POST_DEV ] && exit 0; sleep 120; done
$PY -I -B $D/run.py analyze >> $D/chain.log 2>&1 && echo "$(date -u +%FT%TZ) ANALYZE done" >> $D/chain.log || echo "$(date -u +%FT%TZ) analyze failed" >> $D/chain.log
