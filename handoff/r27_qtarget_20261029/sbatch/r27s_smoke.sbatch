#!/bin/bash
#SBATCH --job-name=r27s_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/logs/%x_%j.out
# R27-S real-data GPU smoke (code_v12): six S arms from the REAL f40_s1001 common prefix, 2 co-train epochs x 30 iters,
# frozen lam (LAMBDA_flickr.json, used by S1 only), then r27s_gate.py -> GATE_S.json.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
PY0=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage.sh
cd $D/code_v12
LAM=$($PY0 -c "import json;print(json.load(open('$D/LAMBDA_flickr.json'))['lambda'])")
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/smoke_s_${SLURM_JOB_ID}
N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy
P=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1001/prefix/prefix_state.pt
C="--data_path $DP --noise_file $N --noise_ratio 0.4 --seed 1001 --log_step 10 --max_iters 30"
for arm in s1_balanced_dual s2_single_free_vcs s3_single_cap_delta s4_single_cap_vcs s5_single_cap_bce s6_single_cap_snce; do
  $PY r27_run.py --arm $arm --lam $LAM $C --output_dir $S/$arm --resume $P --num_epochs 3
done
$PY r27s_gate.py $S $N $DP
echo SMOKE_DONE $S
