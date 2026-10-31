#!/bin/bash
#SBATCH --job-name=r27o3_smoke
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_oracle3_20261031/logs/%x_%j.out
# R27-O3 (ORACLE direct true-target diagnostic) real-data GPU smoke + gate (snapshot code_o3): the O3 arm from the REAL
# f40_s1001 common prefix, 1 co-train epoch x 30 iterations (separate smoke dir), then gate/r27o3_gate.py from the R27-S
# smoke state (both queues ID-bearing) -> SMOKE/GPU_GATE.json. No formal checkpoint written.
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r27_oracle3_20261031/server.env
O=$R27O3_ROOT; Q=$R27_ROOT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
cd $R27O3_CODE
LAM=$($PYTHON_BIN -B -c "import json;print(json.load(open('$Q/LAMBDA_flickr.json'))['lambda'])")
S=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_oracle3_20261031/smoke_o3_${SLURM_JOB_ID}
N=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy
P=/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups/f40_s1001/prefix/prefix_state.pt
G=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/smoke_s_1075954/s4_single_cap_vcs/state_last.pt
C="--data_path $DP --noise_file $N --noise_ratio 0.4 --seed 1001 --k 5 --log_step 10 --max_iters 30"
$PYTHON_BIN -u -B r27_run.py --arm o3_true_target_t2i --lam $LAM $C --output_dir $S/o3_true_target_t2i --resume $P --num_epochs 2
set +e
$PYTHON_BIN -u -B $O/gate/r27o3_gate.py $S $N $DP $G $R27O3_CODE
rc=$?
set -e
cp $S/GPU_GATE.json $O/GPU_GATE_${SLURM_JOB_ID}.json
cp $S/GPU_GATE.json $O/GPU_GATE.json
echo SMOKE_DONE $S rc=$rc
