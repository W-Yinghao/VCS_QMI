# sourced: stage Flickr train_ims.npy + dev_ims.npy (no TEST, no captions: captions are pre-tokenized in role files) -> $IMS
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
IMS=${TMPDIR:-/tmp}/nrcs_$SLURM_JOB_ID; mkdir -p $IMS; trap 'rm -rf $IMS' EXIT
t0=$(date +%s)
for f in train_ims.npy dev_ims.npy; do cp $SRC/$f $IMS/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $IMS/$f) ] || { echo "stage failed $f"; exit 1; }; done
echo "staged $(ls $IMS) in $(( $(date +%s)-t0 ))s"
