#!/bin/bash
#SBATCH --job-name=r27d_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030/logs/%x_%j.out
# R27-D real-data GPU smoke + gate (installed snapshot code_frozen): four D arms from the REAL f40_s1001 common prefix,
# 1 co-train epoch x 30 iterations (package validator: smoke <=2 total epochs, <=30 iterations, separate smoke dir), then
# gate/r27d_gate.py from the R27-S smoke state (both queues ID-bearing) -> SMOKE/GPU_GATE.json. No formal checkpoint written.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030
Q=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
PKG=/home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
PY0=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $Q/sbatch/stage.sh
cd $D/code_frozen
LAM=$($PY0 -c "import json;print(json.load(open('$Q/LAMBDA_flickr.json'))['lambda'])")
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_directional_20261030/smoke_d_${SLURM_JOB_ID}
N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy
P=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1001/prefix/prefix_state.pt
G=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/smoke_s_1075954/s4_single_cap_vcs/state_last.pt
C="--data_path $DP --noise_file $N --noise_ratio 0.4 --seed 1001 --k 5 --log_step 10 --max_iters 30"
for arm in d1_vcs_i2t d2_vcs_t2i d3_snce_i2t d4_snce_t2i; do
  $PY r27_run.py --arm $arm --lam $LAM $C --output_dir $S/$arm --resume $P --num_epochs 2
done
$PY0 -u -B $D/gate/r27d_gate.py $S $N $DP $G $D/code_frozen $PKG
cp $S/GPU_GATE.json $D/GPU_GATE_${SLURM_JOB_ID}.json
echo SMOKE_DONE $S
