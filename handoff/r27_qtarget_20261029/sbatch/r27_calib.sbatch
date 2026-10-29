#!/bin/bash
#SBATCH --job-name=r27_calib
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/logs/%x_%j.out
# Common lambda for A2-A5 at the Flickr common prefix f40_s1001 (same calibration cell as R26 SCALE_flickr).
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage.sh
cd $D/code
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u r27_calibrate.py $D/LAMBDA_flickr.json --data_path $DP \
  --noise_file /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy --noise_ratio 0.4 --seed 1001 \
  --resume /projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1001/prefix/prefix_state.pt \
  --output_dir /projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/calib/f40_s1001
echo CALIB_DONE
