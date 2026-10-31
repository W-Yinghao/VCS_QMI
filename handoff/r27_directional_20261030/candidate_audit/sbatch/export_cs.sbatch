#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --nice=0
#SBATCH --time=01:15:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030/candidate_audit/logs/%x_%j.out
# Usage: export_cs.sbatch CELL NB OUTSUB  -- same audit for the R27-CS final states (bn_cross_t2i / cn_mix_t2i state_last) of CELL
# (read-only replay), validate each set, run the package tool (normal + --exclude-same-source). No training; TRAIN split only.
set -euo pipefail
CELL=$1; NB=$2; SUB=$3
D=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030/candidate_audit
Q=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
PKG=/home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030
RUNS=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/runs/$CELL
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_directional_20261030/candidate_audit/$SUB
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
eval "$($PY - "$Q/CELLS.json" "$CELL" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
for k in ('data_name', 'noise_ratio', 'seed', 'noise_file', 'prefix'):
    print(f"{k.upper()}={c[k]}")
PYEOF
)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# stage Flickr TRAIN split only (no dev/test)
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp
LOC=$TMPDIR/r27dca_$SLURM_JOB_ID/f30k_precomp; mkdir -p $LOC; trap 'rm -rf $TMPDIR/r27dca_$SLURM_JOB_ID' EXIT
for f in train_ims.npy train_caps.txt; do cp $SRC/$f $LOC/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$f) ]; done
DP=$TMPDIR/r27dca_$SLURM_JOB_ID
mkdir -p $O
CS=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_candsrc_20261031/runs/$CELL
BN=$CS/bn_cross_t2i/state_last.pt; [ -f $CS/bn_cross_t2i/DONE.json ] && [ -f $BN ] || BN="MISSING:bn_state_last_not_found"
CN=$CS/cn_mix_t2i/state_last.pt; [ -f $CS/cn_mix_t2i/DONE.json ] && [ -f $CN ] || CN="MISSING:cn_state_last_not_found"
cd $D
$PY -B -u export_candidates.py $O $CELL $NB bn_cross_last=$BN cn_mix_last=$CN -- --data_path $DP \
  --data_name $DATA_NAME --noise_ratio $NOISE_RATIO --noise_file $NOISE_FILE --seed $SEED --output_dir $O/_opt_unused
for S in $O/$CELL/*/peer_*; do
  [ -f $S/metadata.json ] || continue
  [ -f $S/CHECK.json ] || $PY -B check_export.py $S --device cuda
  [ -d $S/normal ] || $PY -B $PKG/tools/candidate_source_audit.py --features $S/features.npz --labels $S/labels.npz --device cuda --out $S/normal
  [ -d $S/exclude_same_source ] || $PY -B $PKG/tools/candidate_source_audit.py --features $S/features.npz --labels $S/labels.npz --device cuda --exclude-same-source --out $S/exclude_same_source
  echo TOOL_DONE $S
done
echo EXPORT_DONE $CELL
