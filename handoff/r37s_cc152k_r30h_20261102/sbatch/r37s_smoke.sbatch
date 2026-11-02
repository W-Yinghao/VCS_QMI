#!/bin/bash
#SBATCH --job-name=r37s_smoke
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/logs/%x_%j.out
# R37S real-data CC152K GPU smoke (code_r37s, frozen PROTOCOL.json): smoke parent (warmup 2 epochs x 30 it, R30 boundary save);
# H0 continuation fork 2 co-train epochs x 30 it, milestone 1 (E1 rank pass, per_captions 1), auto image gather ('cont');
# the same fork stopped after epoch 0 then resumed from its own state_last.pt for epoch 1 with host gather forced ('split');
# runner dry runs; tools/smoke_check.py (fork identity, A1 branches i2t/t2i used, milestone, resume + gather bitwise vs cont).
set -euo pipefail
O=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102
OB=/projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $O/sbatch/stage_cc.sh
cd $O/code_r37s
S=$OB/smoke/smoke_${SLURM_JOB_ID}; P=wp_cc_s1001_H_e5; U=r37s_cc_s1001_H0
C="--data_path $DP --data_name cc152k_precomp --noise_file none --noise_ratio 0.0 --seed 1001 --k 5 --log_step 10 --max_iters 30 --warmup_epoch 2 --num_epochs 2"
$PY -u -B r30_run.py --r30_stage warmup --r30_parent_id $P --r30_source_warmup H $C --prefix_epoch -1 --output_dir $S/parents/$P
jget() { $PY -B -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
J=$S/parents/$P/PARENT.json
F="--r30_stage unit --r30_unit_id $U --r30_source_warmup H --r30_objective H --r30_scale native --r30_parent $(jget $J state_path) --r30_parent_sha $(jget $J sha256) --r30_parent_id $P --r30_parent_fingerprint $(jget $J training_state_fingerprint) --r30_milestone 1"
$PY -u -B r30_run.py $F --r30_mode fork $C --prefix_epoch -1 --output_dir $S/runs/cont
R27X_IMG_GATHER=cpu $PY -u -B r30_run.py $F --r30_mode fork $C --prefix_epoch 0 --stop_after_prefix --output_dir $S/runs/split
mv $S/runs/split/DONE.json $S/runs/split/DONE_part1.json
R27X_IMG_GATHER=cpu $PY -u -B r30_run.py $F --r30_mode resume --resume $S/runs/split/state_last.pt $C --prefix_epoch -1 --output_dir $S/runs/split
set +e
$PY -u -B $O/tools/run_r37s.py --stage parent --root $O --code-dir $O/code_r37s --data-path $DP --out-base $S --python $PY > $S/DRY_parent.txt 2>&1; echo "dry parent rc=$?" >> $S/DRY_parent.txt
$PY -u -B $O/tools/run_r37s.py --stage unit --root $O --code-dir $O/code_r37s --data-path $DP --out-base $S --python $PY > $S/DRY_unit.txt 2>&1; echo "dry unit rc=$?" >> $S/DRY_unit.txt
$PY -u -B $O/tools/smoke_check.py $S
rc=$?
set -e
echo SMOKE_DONE $S check_rc=$rc
