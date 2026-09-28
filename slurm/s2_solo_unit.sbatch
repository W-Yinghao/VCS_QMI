#!/usr/bin/env bash
#SBATCH --partition=H100,RTX6000PRO,A100,L40S
#SBATCH --exclude=node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# S2 (P75): one solo-learn CIFAR-10 ResNet-18 1000-epoch LARS pretraining unit.  METHOD in {vcs, simclr}, SEED required.
# Resumable: re-submitting the same (METHOD, SEED) continues from the last per-epoch checkpoint (solo-learn auto_resume).
# The official test split is evaluated once, at the end of epoch 1000 (final_metrics.json); a unit whose final_metrics.json
# exists exits without touching the data.
set -u
: "${METHOD:?}"; : "${SEED:?}"
PY=/home/infres/yinwang/CS_QMI/env_solo/bin/python
SOLO=/home/infres/yinwang/CS_QMI/solo_learn
OUT=/home/infres/yinwang/CS_QMI/outputs/P75_S2_solo/${METHOD}_seed${SEED}
mkdir -p "$OUT"
echo "host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION start=$(date -u +%FT%TZ) method=$METHOD seed=$SEED"
echo "solo_commit=$(git -C $SOLO rev-parse --short HEAD) solo_patch_sha256=$(git -C $SOLO diff HEAD | sha256sum | cut -c1-16) ssl_pilot_commit=$(git -C /home/infres/yinwang/CS_QMI/ssl_pilot rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -f "$OUT/final_metrics.json" ]; then echo "final_metrics.json exists: unit complete, nothing to do"; cat "$OUT/final_metrics.json"; exit 0; fi
cd "$SOLO"
"$PY" main_pretrain.py --config-path scripts/pretrain/cifar --config-name "s2_${METHOD}.yaml" \
  seed="$SEED" name="s2-${METHOD}-cifar10-seed${SEED}" checkpoint.dir="$OUT" +default_root_dir="$OUT" data.num_workers=8
rc=$?
[ -f "$OUT/final_metrics.json" ] && cat "$OUT/final_metrics.json"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
