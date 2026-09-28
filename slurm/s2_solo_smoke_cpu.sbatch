#!/usr/bin/env bash
#SBATCH --job-name=s2_solo_smoke_cpu
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/s2_solo_smoke_cpu_%j.out
# S2 (P75) CPU smoke: (1) port-vs-reference loss/grad equivalence; (2) 3 training steps of VCS and SimCLR through
# solo-learn's main_pretrain on CPU with the official test split disabled (S2_NO_TEST=1, no val loader).
set -u
PY=/home/infres/yinwang/CS_QMI/env_solo/bin/python
SOLO=/home/infres/yinwang/CS_QMI/solo_learn
OUT=/home/infres/yinwang/CS_QMI/outputs/P75_S2_smoke_${SLURM_JOB_ID}
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -u +%FT%TZ) solo_commit=$(git -C $SOLO rev-parse --short HEAD) solo_dirty=$(git -C $SOLO status --porcelain | wc -l)"
$PY /home/infres/yinwang/CS_QMI/ssl_pilot/s2_solo_learn/smoke_equivalence.py; rc0=$?
cd $SOLO
rc=0
for m in vcs simclr; do
  t0=$(date +%s)
  S2_NO_TEST=1 $PY main_pretrain.py --config-path scripts/pretrain/cifar --config-name s2_$m.yaml \
    accelerator=cpu devices=1 strategy=auto precision=32 sync_batchnorm=False max_epochs=1 +limit_train_batches=3 \
    data.num_workers=4 checkpoint.dir=$OUT/$m auto_resume.enabled=False +performance.disable_channel_last=True +default_root_dir=$OUT/$m +log_every_n_steps=1 seed=0 \
    || rc=1
  echo "[$m] rc=$rc wall=$(( $(date +%s) - t0 ))s"; ls -R $OUT/$m | head -8
done
echo "exit_equiv=$rc0 exit_train=$rc end=$(date -u +%FT%TZ)"; exit $(( rc0 | rc ))
