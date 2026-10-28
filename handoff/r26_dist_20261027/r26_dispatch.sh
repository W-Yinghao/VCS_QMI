#!/usr/bin/env bash
# R26 dispatcher (DESIGN §3-4): only technical prerequisites gate submission; no result is ever read here.
#   1. calib per dataset when its s1001 prefix exists and the GPU smoke GATE passed -> SCALE_<ds>.json
#   2. stage A feeder (TAG r26_flickr)  when SCALE_flickr ok + f40/f60 s1001 prefixes exist
#   3. stage B feeder (TAG r26_flickrB) when SCALE_flickr ok + the four seed1002/1003 Flickr prefixes exist
#   4. stage C feeder (TAG r26_cross)   when stage A+B feeders started + SCALE_coco/cc152k ok + all six prefixes exist; C lines nice 60 so Flickr runs first
# Write STOP_DISPATCH to exit. Log: dispatch.log. Run with setsid on the login node.
D=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027
W=/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/deps_watch
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
GATE=$1   # path to the passed smoke GATE.json
TF=$2; TC=$3; TCC=$4   # --time for Flickr / COCO / CC152K full runs
log() { echo "$(date -u +%FT%TZ) $*" >> $D/dispatch.log; }
pfx() { $PY -c "import json;print(json.load(open('$D/CELLS.json'))['$1']['prefix'])"; }
scale_ok() { [ -f $D/SCALE_$1.json ] && $PY -c "import json,sys;c=json.load(open('$D/SCALE_$1.json'))['c_J'];sys.exit(0 if c and all(v is not None for v in c.values()) else 1)"; }
all_exist() { for c in "$@"; do [ -f "$(pfx $c)" ] || return 1; done; }
calib_sub() { # ds cell
  [ -f $D/CALIB_SUBMITTED_$1 ] && return
  J=$(cd /home/infres/yinwang/CS_QMI/slurm_logs && sbatch --parsable --job-name=r26_calib_$1 --time=01:30:00 --nice=0 $D/sbatch/r26_unit.sbatch calib $2)
  [ -n "$J" ] && { echo $J > $D/CALIB_SUBMITTED_$1; printf "%s\tr26_calib_%s\t%s\tcalib %s\n" "$(date -u +%FT%TZ)" $1 $J $2 >> $D/SUBMITTED.tsv; log "calib $1 -> $J"; }
}
feed() { # stage tag
  [ -f $D/FEED_STARTED_$1 ] && return
  $PY $D/code/make_lines.py $1 $TF $TC $TCC >> $D/dispatch.log
  (cd $W && setsid nohup bash ./app_lines_feed.sh $D/LINES_$1.txt $2 18 0 >/dev/null 2>&1 < /dev/null &)
  date -u +%FT%TZ > $D/FEED_STARTED_$1; log "stage $1 feeder started TAG $2"
}
n_tsv() { [ -f $W/APPS_$1_SUBMITTED.tsv ] && awk -F'\t' '$3 ~ /^[0-9]+$/' $W/APPS_$1_SUBMITTED.tsv | cut -f2 | sort -u | wc -l || echo 0; }
$PY -c "import json,sys;sys.exit(0 if json.load(open('$GATE'))['all_pass'] else 1)" || { log "gate $GATE not passed: exit"; exit 1; }
log "dispatch start (pid $$) gate=$GATE times $TF/$TC/$TCC"
while [ ! -e $D/STOP_DISPATCH ]; do
  all_exist f40_s1001 && calib_sub flickr f40_s1001
  all_exist c40_s1001 && calib_sub coco c40_s1001
  all_exist cc_s1001 && calib_sub cc152k cc_s1001
  scale_ok flickr && all_exist f40_s1001 f60_s1001 && feed A r26_flickr
  scale_ok flickr && all_exist f40_s1002 f60_s1002 f40_s1003 f60_s1003 && feed B r26_flickrB
  if [ -f $D/FEED_STARTED_A ] && [ -f $D/FEED_STARTED_B ] \
     && scale_ok coco && scale_ok cc152k && all_exist c40_s1001 c40_s1002 c40_s1003 cc_s1001 cc_s1002 cc_s1003; then
    feed C r26_cross
  fi
  [ -f $D/FEED_STARTED_C ] && { log "all three feeders started: exit"; exit 0; }
  sleep 120
done
log "STOP_DISPATCH: exit"
