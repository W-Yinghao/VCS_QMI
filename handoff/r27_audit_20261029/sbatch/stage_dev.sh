# sourced: stage Flickr DEV split only (no train/test) to node-local TMPDIR, sets DP
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r27r_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r27r_$SLURM_JOB_ID' EXIT
for f in dev_ims.npy dev_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
DP=$TMPDIR/r27r_$SLURM_JOB_ID
