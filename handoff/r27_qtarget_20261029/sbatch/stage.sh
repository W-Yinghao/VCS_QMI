# sourced: stage Flickr train/dev to node-local TMPDIR (no test files), sets DP
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r27_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r27_$SLURM_JOB_ID' EXIT
t0=$(date +%s)
for f in train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
DP=$TMPDIR/r27_$SLURM_JOB_ID; echo "staged in $(( $(date +%s)-t0 ))s"
