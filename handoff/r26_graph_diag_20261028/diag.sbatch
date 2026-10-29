#!/bin/bash
#SBATCH --job-name=r26_graphdiag
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r26_graph_diag_20261028/logs/%x_%j.out
# R26 fixed-graph diagnostic (adjustments 20261028 P1-3): isolated replay of frozen checkpoints, no training/update.
set -euo pipefail
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
D=/home/infres/yinwang/CS_QMI/handoff/r26_graph_diag_20261028
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r26_graph_diag_20261028/$SLURM_JOB_ID
R=/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs
P=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups
NZ=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise
mkdir -p $O; nvidia-smi --query-gpu=name,driver_version --format=csv
cd $D
for CELL in f40_s1001 f60_s1001; do
  NR=0.${CELL:1:1}
  $PY -u r26_graph_export.py $O $CELL prefix=$P/$CELL/prefix/prefix_state.pt mid=MISSING_mid_checkpoint_not_saved \
     d_vcs_final=$R/$CELL/d_vcs/state_last.pt d_bce_final=$R/$CELL/d_bce/state_last.pt m_vcs_final=$R/$CELL/m_vcs/state_last.pt \
     -- --data_name f30k_precomp --noise_ratio $NR --noise_file $NZ/f30k_precomp_$NR.npy --seed ${CELL:5:4} --output_dir $O/_opt_unused
done
$PY -u aggregate.py $O /home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/SCALE_flickr.json
echo DIAG_DONE
