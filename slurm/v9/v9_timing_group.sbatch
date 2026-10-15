#!/usr/bin/env bash
#SBATCH --job-name=v9_timing_group
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-01 grouped budget measurement on Flickr30K precomp (noise 0.4, DECL/RCL index), one GPU, sequential (owner: per-method measured costs, no multiplier):
#   (a) native SGR 1 epoch incl. per-epoch dev validation + final test evalrank          (envs/v9_sgr, SGRAF_py3 a19d1a9, unmodified train.py)
#   (b) NCR  --warmup_epoch 1 --num_epochs 2 (1 warm-up + 1 co-training epoch, 2 nets)   (envs/v9_sgr, NCR 60256ba, unmodified run.py)
#   (c) L2RM --warmup_epoch 1 --num_epochs 2 (1 warm-up + 1 rematching epoch)            (envs/v9_l2rm, L2RM a766c22, unmodified main_L2RM.py)
#   (d) teacher: native SGR on a 1,160-image subset of train for 5 epochs                (envs/v9_sgr; subset = first 1,160 train images in file order, timing only)
# Outputs under outputs_v9/timing only; nothing is selected from these runs; the test evaluations here are host-protocol timing checks, not reported numbers.
ROOT=/projects/EEG-foundation-model/yinghao; C=$ROOT/code_v9; DATA=${DATA:-$ROOT/data/ncr_precomp/NCR-data/data}; VOCAB=${VOCAB:-$ROOT/data/ncr_precomp/NCR-data/vocab}
NOISE=$C/L2RM/noise_index/f30k_precomp_0.4.npy; O=$ROOT/outputs_v9/timing/group_${SLURM_JOB_ID}; mkdir -p $O; export PYTHONNOUSERSITE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) sgraf=$(git -C $C/SGRAF_py3 rev-parse --short HEAD) ncr=$(git -C $C/2021-NeurIPS-NCR rev-parse --short HEAD) l2rm=$(git -C $C/L2RM rev-parse --short HEAD) data=$DATA vocab=$VOCAB"
[ -f $VOCAB/f30k_precomp_vocab.json ] || VOCAB=$C/SGRAF_py3/vocab
t() { echo "[$(date -u +%FT%TZ)] $*"; }
# ---- (a) native SGR, 1 epoch + dev validation + final test eval
t "SGR start"; cd $C/SGRAF_py3; s=$(date +%s)
$ROOT/envs/v9_sgr/bin/python train.py --data_path $DATA --data_name f30k_precomp --vocab_path $VOCAB/ --model_name $O/sgr/checkpoint --logger_name $O/sgr/log --num_epochs 1 --lr_update 30 --module_name SGR --batch_size 128 --workers 4 > $O/sgr_train.log 2>&1; rc_a=$?
t "SGR 1 epoch (+ dev val) rc=$rc_a seconds=$(( $(date +%s) - s ))"; grep -E "Epoch|rsum|Image to text|Text to image|calculate similarity" $O/sgr_train.log | tail -6
s=$(date +%s); $ROOT/envs/v9_sgr/bin/python -c "
import evaluation as EV; EV.evalrank('$O/sgr/checkpoint/model_best.pth.tar', data_path='$DATA', split='test', fold5=False)" > $O/sgr_test_eval.log 2>&1; t "SGR test evalrank rc=$? seconds=$(( $(date +%s) - s ))"; tail -3 $O/sgr_test_eval.log
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
# ---- (b) NCR: 1 warm-up epoch + 1 co-training epoch (two networks), unmodified run.py
t "NCR start"; cd $C/2021-NeurIPS-NCR/NCR; s=$(date +%s)
$ROOT/envs/v9_sgr/bin/python run.py --gpu 0 --workers 2 --data_name f30k_precomp --data_path $DATA --vocab_path $VOCAB --noise_ratio 0.4 --noise_file $NOISE --warmup_epoch 1 --num_epochs 2 --lr_update 30 --seed 917 --output_dir $O/ncr > $O/ncr_train.log 2>&1; rc_b=$?
t "NCR warm-up 1 + co-train 1 rc=$rc_b seconds=$(( $(date +%s) - s ))"; grep -i -E "epoch|rsum|warm|time" $O/ncr_train.log | tail -12
# ---- (c) L2RM: 1 warm-up epoch + 1 rematching epoch, unmodified main_L2RM.py (writes runs/ and output_dir relative to cwd -> run from $O/l2rm_cwd)
t "L2RM start"; mkdir -p $O/l2rm_cwd; cd $O/l2rm_cwd; s=$(date +%s)
PYTHONPATH=$C/L2RM $ROOT/envs/v9_l2rm/bin/python $C/L2RM/main_L2RM.py --comment _timing --module_name SGR --gpu 0 --batch_size 128 --seed 917 --reg 0.01 --num_epochs 2 --lr 2e-4 --lr_cost 2e-6 --workers 0 --lr_update 15 --data_name f30k_precomp --noise_ratio 0.4 --queue_length 128 --warmup_epoch 1 --data_path $DATA --vocab_path $VOCAB --noise_file $NOISE --output_dir $O/l2rm > $O/l2rm_train.log 2>&1; rc_c=$?
t "L2RM warm-up 1 + epoch 1 rc=$rc_c seconds=$(( $(date +%s) - s ))"; grep -i -E "epoch|rsum|warm|time" $O/l2rm_train.log | tail -12
# ---- (d) teacher: SGR on the first 1,160 train images (5,800 captions) for 5 epochs (timing only; the real teacher role is a seeded image-level draw)
t "teacher start"; T=$O/teacher_data/f30k_precomp; mkdir -p $T; $ROOT/envs/v9_sgr/bin/python - "$DATA/f30k_precomp" "$T" <<'PY'
import numpy as np, sys, shutil, os
src, dst = sys.argv[1], sys.argv[2]; n = 1160
ims = np.load(f"{src}/train_ims.npy", mmap_mode="r"); np.save(f"{dst}/train_ims.npy", np.ascontiguousarray(ims[:n]))
caps = open(f"{src}/train_caps.txt", "rb").read().splitlines()[: n * 5]; open(f"{dst}/train_caps.txt", "wb").write(b"\n".join(caps) + b"\n")
for s in ("dev",):
    for ext in ("_ims.npy", "_caps.txt", "_ids.txt"):
        if os.path.exists(f"{src}/{s}{ext}"): shutil.copy(f"{src}/{s}{ext}", f"{dst}/{s}{ext}")
print("teacher subset", n, "images")
PY
cd $C/SGRAF_py3; s=$(date +%s)
$ROOT/envs/v9_sgr/bin/python train.py --data_path $O/teacher_data --data_name f30k_precomp --vocab_path $VOCAB/ --model_name $O/teacher/checkpoint --logger_name $O/teacher/log --num_epochs 5 --lr_update 30 --module_name SGR --batch_size 128 --workers 4 > $O/teacher_train.log 2>&1; rc_d=$?
t "teacher 5 epochs on 1,160 images rc=$rc_d seconds=$(( $(date +%s) - s ))"
rm -rf $O/teacher_data
echo "exit=$(( rc_a | rc_b | rc_c | rc_d )) end=$(date -u +%FT%TZ)"
