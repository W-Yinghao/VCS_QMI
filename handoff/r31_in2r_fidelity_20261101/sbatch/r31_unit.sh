#!/bin/bash
# R31 native IN2R fidelity unit.  r31_unit.sh CELL ARM [formal|smoke]
#   CELL f20_s100X (Flickr 20%, OFFICIAL NCR index) | f40_s100X (Flickr 40%, R25 map) | cc_s100X (CC152K, real noise)
#   ARM  H0_codecfg          R25 code-version config (= r25_run.py defaults: warmup 5, K 5, i2t+t2i on)   [f20 only]
#        H0_default          author run.py defaults: warmup 1, K 1, i2t/t2i off                          [f40, cc]
#        H0_default_i2t_t2i  author run.py defaults + --i2t --t2i: warmup 1, K 1, i2t/t2i on             [f40, cc]
# Code = byte-identical copy of R25 code/ (incl. the native-fixed guards). Trains from scratch (no shared prefix).
# Resume: if <out>/state_last.pt exists (preemption/requeue or timeout continuation) the run resumes from it;
# every attempt is appended to <out>/ATTEMPTS.tsv. DONE.json present -> nothing to do.
# Env (smoke only): R31_SMOKE_EPOCHS (default 1), R31_DP (pre-staged data root).
set -euo pipefail
CELL=$1; ARM=$2; MODE=${3:-formal}
D=/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
NCR=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data
SEED=${CELL##*_s}
case $CELL in
  f20_s*) DATA=f30k_precomp; NR=0.2; NF=$NCR/noise_index/f30k_precomp_0.2.npy; OKARMS=" H0_codecfg ";;
  f40_s*) DATA=f30k_precomp; NR=0.4; NF=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy; OKARMS=" H0_default H0_default_i2t_t2i ";;
  cc_s*)  DATA=cc152k_precomp; NR=0; NF=none; OKARMS=" H0_default H0_default_i2t_t2i ";;
  *) echo "bad cell $CELL"; exit 2;;
esac
case "$OKARMS" in *" $ARM "*) ;; *) echo "arm $ARM not in design for $CELL"; exit 2;; esac
case $ARM in
  H0_codecfg)         A="";;
  H0_default)         A="--warmup_epoch 1 --k 1 --no_i2t --no_t2i";;
  H0_default_i2t_t2i) A="--warmup_epoch 1 --k 1";;
esac
case $SEED in 1001|1002|1003) ;; *) echo "bad seed $SEED"; exit 2;; esac
if [ "$MODE" = smoke ]; then OUT=$O/smoke/$CELL/$ARM; EXTRA="--max_iters 20 --num_epochs ${R31_SMOKE_EPOCHS:-1} --log_step 10"
elif [ "$MODE" = formal ]; then OUT=$O/runs/$CELL/$ARM; EXTRA=""
else echo "bad mode $MODE"; exit 2; fi
mkdir -p $OUT
test -f $OUT/DONE.json && { echo "already DONE $OUT"; exit 0; }

# ---- data: stage FIT/DEV files (never TEST) to node-local disk if there is room, else mmap from NFS
if [ -n "${R31_DP:-}" ]; then DP=$R31_DP
else
  SRC=$NCR/data/$DATA
  case $DATA in cc152k_precomp) FILES="train_ims.npy dev_ims.npy train_caps.tsv dev_caps.tsv";; *) FILES="train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt";; esac
  NEED=$(cd $SRC && du -cb $FILES | tail -1 | cut -f1); TD=${TMPDIR:-/tmp}; AVAIL=$(df --output=avail -B1 "$TD" | tail -1)
  if [ "$AVAIL" -gt $(( NEED + 20*1024*1024*1024 )) ]; then
    LOC=$TD/r31_${SLURM_JOB_ID}_$$; mkdir -p $LOC/$DATA; trap 'rm -rf '"$LOC" EXIT
    t0=$(date +%s); for f in $FILES; do cp $SRC/$f $LOC/$DATA/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$DATA/$f) ]; done
    DP=$LOC; echo "staged $NEED bytes in $(( $(date +%s)-t0 ))s"
  else DP=$NCR/data; echo "no local space ($AVAIL < $NEED+20G): reading $SRC via mmap"; fi
fi

# ---- attempt bookkeeping / resume
J=${SLURM_JOB_ID:-nojob}; RC=${SLURM_RESTART_COUNT:-0}
RES=""; [ -f $OUT/state_last.pt ] && RES="--resume $OUT/state_last.pt"
N=1; [ -f $OUT/ATTEMPTS.tsv ] && N=$(( $(wc -l < $OUT/ATTEMPTS.tsv) + 1 ))
[ -f $OUT/RUN_$J.json ] && mv $OUT/RUN_$J.json $OUT/RUN_${J}_attempt$((N-1)).json   # requeue keeps the job id
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$N" "$(date -u +%FT%TZ)" "$J" "$RC" "$(hostname)" "$GPU" "${RES:-from_scratch}" >> $OUT/ATTEMPTS.tsv
echo "attempt $N job $J restart_count $RC ${RES:-from_scratch}"
cd $D/code
CMD="$PY -B -u r25_run.py --data_path $DP --data_name $DATA --noise_file $NF --noise_ratio $NR --seed $SEED --output_dir $OUT $A $RES $EXTRA"
echo "CMD $CMD"
if [ "$MODE" = formal ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
  set +e; timeout $LIM $CMD; rc=$?; set -e
else
  set +e; $CMD; rc=$?; set -e
fi
if [ -f $OUT/DONE.json ]; then echo "R31_UNIT_DONE $CELL $ARM"; exit 0; fi
# wall-clock continuation (not preemption: a preempted job is requeued by Slurm and resumes above)
if [ "$MODE" = formal ] && [ $rc -eq 124 ] && [ $N -lt 8 ]; then
  NJ=$(sbatch --parsable --job-name=$SLURM_JOB_NAME $D/sbatch/r31_unit.sbatch $CELL $ARM)
  printf "%s\t%s\t%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$NJ" "$CELL" "$ARM" "$SLURM_JOB_ID" >> $D/SUBMITTED.tsv
  echo "timeout: continuation submitted $NJ"; exit 0
fi
echo "R31_UNIT_FAILED rc=$rc attempt=$N"; exit 1
