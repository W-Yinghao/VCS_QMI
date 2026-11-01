#!/bin/bash
#SBATCH --job-name=r30v11_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/logs/%x_%j.out
# R30 v1.1 real-data GPU smoke + gate (snapshot code_r30v11, frozen v1.1+A1 protocol), reduced to the gate minimum: smoke
# parents wp_f60_s1001_H_e5 and wp_f60_s1001_L_e5 (warmup 2 epochs x 30 it), one smoke calibration (H parent, 4 batches/peer),
# three real branches with matrix unit ids (30 it per co-train epoch, milestone 1): H0 1 epoch, LN (L parent) 1 epoch, PV 2 epochs
# (milestone then continues); every objective is covered in-process by the gate (fork identity, isolation steps, math);
# then gate/r30v11_gate.py.
# Submit with --qos/--partition on the command line.
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $R27_STAGE_SH
cd $R30V11_CODE
PY=$PYTHON_BIN
S=$R30_OUT_BASE/smoke/smoke_v11_${SLURM_JOB_ID}
C="--data_path $DP --data_name f30k_precomp --seed 1001 --k 5 --log_step 10 --max_iters 30 --warmup_epoch 2 --prefix_epoch -1"
declare -A NZ=([f60]="--noise_file $R30_NOISE60 --noise_ratio 0.6" [f00]="--noise_file $R30_NOISE0 --noise_ratio 0.0")
jget() { $PY -B -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
coef() { $PY -B -c "import json,sys;print(repr(json.load(open(sys.argv[1]))['coefficients'][sys.argv[2]]['c']))" "$S/calib/$1/COEFFICIENTS_smoke.json" "$2"; }
parent() {  # parent_id cell source
  $PY -u -B r30_run.py --r30_stage warmup --r30_parent_id $1 --r30_source_warmup $3 $C ${NZ[$2]} --num_epochs 2 --output_dir $S/parents/$1
}
parent wp_f60_s1001_H_e5 f60 H
parent wp_f60_s1001_L_e5 f60 L
for pc in wp_f60_s1001_H_e5:f60; do
  P=${pc%%:*}; c=${pc##*:}; J=$S/parents/$P/PARENT.json
  $PY -u -B r30_calibrate.py $C ${NZ[$c]} --num_epochs 2 --output_dir $S/calib/$P --r30_parent $(jget $J state_path) --r30_parent_sha $(jget $J sha256) \
      --r30_parent_id $P --r30_parent_fingerprint $(jget $J training_state_fingerprint) --n_batches 4 --out $S/calib/$P/COEFFICIENTS_smoke.json
done
unit() {  # unit_id parent_id cell source objective scale epochs [coef_key]
  J=$S/parents/$2/PARENT.json; X=""
  [ "$6" = fit_static ] && X="--r30_coef $(coef $2 $8)"
  $PY -u -B r30_run.py --r30_stage unit --r30_unit_id $1 --r30_source_warmup $4 --r30_objective $5 --r30_scale $6 $X --r30_mode fork \
      --r30_parent $(jget $J state_path) --r30_parent_sha $(jget $J sha256) --r30_parent_id $2 --r30_parent_fingerprint $(jget $J training_state_fingerprint) \
      --r30_milestone 1 $C ${NZ[$3]} --num_epochs $7 --output_dir $S/runs/$1
}
H=wp_f60_s1001_H_e5; L=wp_f60_s1001_L_e5
unit r30_f60_s1001_H0 $H f60 H H native 1
unit r30v11_f60_s1001_LN $L f60 L N 1 1
unit r30_f60_s1001_PV $H f60 H PV fit_static 2 PV
set +e
$PY -u -B $R30_ROOT/gate/r30v11_gate.py $S $DP $R30V11_CODE $R30_ROOT/reference $R30V11_PKG
rc=$?
set -e
cp $S/GPU_GATE.json $R30_ROOT/GPU_GATE_V11_${SLURM_JOB_ID}.json
cp $S/GPU_GATE.json $R30_ROOT/GPU_GATE.json
echo SMOKE_DONE $S gate_rc=$rc
