# sourced with DATA_NAME set: stage ONLY train/dev files of $DATA_NAME into node-local TMPDIR (no TEST file is ever
# copied or linked), sets DP. Without local space, a directory of symlinks to the same train/dev files is used.
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/$DATA_NAME
case $DATA_NAME in cc152k_precomp) FILES="train_ims.npy dev_ims.npy train_caps.tsv dev_caps.tsv";; *) FILES="train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt";; esac
DP=${TMPDIR:-/tmp}/r27x_$SLURM_JOB_ID; LOC=$DP/$DATA_NAME; mkdir -p $LOC; trap 'rm -rf ${TMPDIR:-/tmp}/r27x_$SLURM_JOB_ID' EXIT
NEED=$(cd $SRC && du -cb $FILES | tail -1 | cut -f1); AVAIL=$(df --output=avail -B1 "${TMPDIR:-/tmp}" | tail -1)
t0=$(date +%s)
if [ "$AVAIL" -gt $(( NEED + 20*1024*1024*1024 )) ]; then
  for f in $FILES; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
  echo "staged $DATA_NAME ($NEED bytes) in $(( $(date +%s)-t0 ))s: $(ls $LOC | tr '\n' ' ')"
else
  for f in $FILES; do ln -s $SRC/$f $LOC/$f; done
  echo "no local space ($AVAIL < $NEED+20G): symlinked train/dev only: $(ls $LOC | tr '\n' ' ')"
fi
