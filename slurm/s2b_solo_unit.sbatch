#!/usr/bin/env bash
#SBATCH --partition=H100,RTX6000PRO,A100,L40S
#SBATCH --exclude=node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# S2b (P77): the P75 VCS unit with the critic scalars on the recipe's optimiser (method vcs_s2b, config s2b_vcs.yaml).
# SEED required.  Resumable (solo-learn auto_resume).  Test-set rule identical to P75: official split read once at epoch 1000.
set -u
: "${SEED:?}"
PY=/home/infres/yinwang/CS_QMI/env_solo/bin/python
SOLO=/home/infres/yinwang/CS_QMI/solo_learn
OUT=/home/infres/yinwang/CS_QMI/outputs/P77_S2b_solo/vcs_s2b_seed${SEED}
mkdir -p "$OUT"
echo "host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION start=$(date -u +%FT%TZ) method=vcs_s2b seed=$SEED"
echo "solo_commit=$(git -C $SOLO rev-parse --short HEAD) solo_patch_sha256=$(git -C $SOLO diff HEAD | sha256sum | cut -c1-16) ssl_pilot_commit=$(git -C /home/infres/yinwang/CS_QMI/ssl_pilot rev-parse --short HEAD)"
echo "s2b_files_sha256: $(cd $SOLO && sha256sum solo/methods/vcs_s2b.py main_pretrain_s2b.py scripts/pretrain/cifar/s2b_vcs.yaml | awk '{print substr($1,1,16)" "$2}' | tr '\n' ';')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ -f "$OUT/final_metrics.json" ]; then echo "final_metrics.json exists: unit complete, nothing to do"; cat "$OUT/final_metrics.json"; exit 0; fi
cd "$SOLO"
"$PY" main_pretrain_s2b.py --config-path scripts/pretrain/cifar --config-name s2b_vcs.yaml \
  seed="$SEED" name="s2b-vcs-cifar10-seed${SEED}" checkpoint.dir="$OUT" +default_root_dir="$OUT" data.num_workers=8
rc=$?
[ -f "$OUT/final_metrics.json" ] && cat "$OUT/final_metrics.json"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
