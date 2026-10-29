#!/bin/bash
#SBATCH --job-name=r27_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/logs/%x_%j.out
# Real-data GPU smoke: all five arms from the REAL f40_s1001 common prefix, 2 co-train epochs x 30 iterations each
# (lam = 1.0 placeholder; the formal lam comes from r27_calib), then r27_gate.py.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage.sh
cd $D/code
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/smoke_${SLURM_JOB_ID}
N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy
P=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1001/prefix/prefix_state.pt
C="--data_path $DP --noise_file $N --noise_ratio 0.4 --seed 1001 --log_step 10 --max_iters 30"
for arm in a1_iface a2_mean a3_vcs a4_bce a5_snce; do
  L=1.0; [ $arm = a1_iface ] && L=0.0
  $PY r27_run.py --arm $arm --lam $L $C --output_dir $S/$arm --resume $P --num_epochs 3
done
$PY r27_gate.py $S $N $DP
echo SMOKE_DONE $S
