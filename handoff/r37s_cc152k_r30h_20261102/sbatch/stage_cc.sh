# sourced: stage ONLY the CC152K train/dev files (no TEST file is ever copied or linked) into node-local TMPDIR, sets DP.
# Without local space, a directory of symlinks to the same train/dev files is used (as R27-A1X stage_ds.sh).
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/cc152k_precomp
FILES="train_ims.npy dev_ims.npy train_caps.tsv dev_caps.tsv"
DP=${TMPDIR:-/tmp}/r37s_$SLURM_JOB_ID; LOC=$DP/cc152k_precomp; mkdir -p $LOC; trap 'rm -rf ${TMPDIR:-/tmp}/r37s_$SLURM_JOB_ID' EXIT
NEED=$(cd $SRC && du -cb $FILES | tail -1 | cut -f1); AVAIL=$(df --output=avail -B1 "${TMPDIR:-/tmp}" | tail -1)
t0=$(date +%s)
if [ "$AVAIL" -gt $(( NEED + 20*1024*1024*1024 )) ]; then
  for f in $FILES; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
  echo "staged cc152k_precomp ($NEED bytes) in $(( $(date +%s)-t0 ))s: $(ls $LOC | tr '\n' ' ')"
else
  for f in $FILES; do ln -s $SRC/$f $LOC/$f; done
  echo "no local space ($AVAIL < $NEED+20G): symlinked train/dev only: $(ls $LOC | tr '\n' ' ')"
fi
