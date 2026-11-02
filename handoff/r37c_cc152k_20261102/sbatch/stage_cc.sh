# sourced: stage CC152K train/dev (never test) to node-local disk if there is room (else mmap from NFS); sets DP
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/cc152k_precomp
FILES="train_ims.npy dev_ims.npy train_caps.tsv dev_caps.tsv"
NEED=$(cd $SRC && du -cb $FILES | tail -1 | cut -f1); TD=${TMPDIR:-/tmp}; AVAIL=$(df --output=avail -B1 "$TD" | tail -1)
if [ "$AVAIL" -gt $(( NEED + 20*1024*1024*1024 )) ]; then
  LOC=$TD/r37c_${SLURM_JOB_ID}_$$; mkdir -p $LOC/cc152k_precomp; trap 'rm -rf '"$LOC" EXIT
  t0=$(date +%s); for f in $FILES; do cp $SRC/$f $LOC/cc152k_precomp/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/cc152k_precomp/$f) ]; done
  DP=$LOC; echo "staged $NEED bytes to $LOC in $(( $(date +%s)-t0 ))s"
else
  DP=$(dirname $SRC); echo "no local space ($AVAIL < $NEED+20G): reading $SRC via mmap"
fi
