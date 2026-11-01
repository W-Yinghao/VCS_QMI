#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,L40S,A100,A40
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101/logs/%x_%j.out
# Usage: r31_smoke.sbatch flickr|cc  -- every config of that dataset: real config, 20 iterations per warmup/co-train
# epoch, 1 co-train epoch (eval_train and DEV validation are full). flickr also checks resume (state_last -> epoch 1).
set -euo pipefail
WHICH=$1
D=/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101
NCR=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
case $WHICH in
  flickr) DATA=f30k_precomp; FILES="train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt"
          UNITS="f20_s1001:H0_codecfg f40_s1001:H0_default f40_s1001:H0_default_i2t_t2i";;
  cc)     DATA=cc152k_precomp; FILES="train_ims.npy dev_ims.npy train_caps.tsv dev_caps.tsv"
          UNITS="cc_s1001:H0_default cc_s1001:H0_default_i2t_t2i";;
esac
LOC=${TMPDIR:-/tmp}/r31s_$SLURM_JOB_ID; mkdir -p $LOC/$DATA; trap 'rm -rf '"$LOC" EXIT
t0=$(date +%s); for f in $FILES; do cp $NCR/data/$DATA/$f $LOC/$DATA/; [ $(stat -c %s $NCR/data/$DATA/$f) = $(stat -c %s $LOC/$DATA/$f) ]; done
echo "staged in $(( $(date +%s)-t0 ))s"
export R31_DP=$LOC
for u in $UNITS; do
  c=${u%%:*}; a=${u##*:}; rm -rf $O/smoke/$c/$a
  t1=$(date +%s); bash $D/sbatch/r31_unit.sh $c $a smoke; echo "SMOKE_UNIT_OK $c $a $(( $(date +%s)-t1 ))s"
done
if [ $WHICH = flickr ]; then   # resume check: continue f20 codecfg from its state_last.pt to co-train epoch 1
  S=$O/smoke/f20_s1001/H0_codecfg; mv $S/DONE.json $S/DONE_epoch0.json; cp $S/EPOCHS.json $S/EPOCHS_epoch0.json
  R31_SMOKE_EPOCHS=2 bash $D/sbatch/r31_unit.sh f20_s1001 H0_codecfg smoke; echo SMOKE_RESUME_RAN
fi
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B $D/tools/check_smoke.py $WHICH $SLURM_JOB_ID
