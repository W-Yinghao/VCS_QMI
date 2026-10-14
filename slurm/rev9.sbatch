#!/usr/bin/env bash
#SBATCH --job-name=rev9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV9 (reports/REV9/REV9_PROTOCOL_FROZEN_20261014.yaml).  Positional arguments only.
#   phaseb                    all 21 phase-B runs (9 units x I1 / I2 + I0 on seed 0 of each objective), main env, any GPU
#   eval <bench> <i/n>        frozen D1-30 runner on the REV9 units, SparseVLM env, H100 / L40S only
#   score                     CPU readout (rev9_score.py)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; export PYTHONNOUSERSITE=1
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; RV=/home/infres/yinwang/CS_QMI/outputs/REV9; mkdir -p $RV/selectors $RV/d1_30/pred
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat scripts/rev9_phaseb.py slurm/rev9.sbatch | sha256sum | cut -c1-16)"
what="$1"; shift; rc=0
case "$what" in
  phaseb) for t in vcs_lr0.0003 logistic_lr0.0001 direct_lr0.0001; do for s in 0 1 2; do iv="I1 I2"; [ $s -eq 0 ] && iv="I0 I1 I2"
            "$PYTHON" scripts/rev9_phaseb.py run --tag ${t}_s$s --interventions $iv --d1 $D1 --out $RV/selectors || rc=$?; done; done;;
  eval) E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf
        $E/bin/python scripts/r7_d1_30_run.py --sets-dir $D1/d1_30/sets --bench "$1" --configs-file $RV/configs/configs_$1.txt --shard "$2" --out $RV/d1_30/pred \
          --selectors-dir $RV/selectors --pca $D1/cache/pca.pt; rc=$?;;
  score) mkdir -p /home/infres/yinwang/CS_QMI/outputs/REV9/results; "$PYTHON" scripts/rev9_score.py --out /home/infres/yinwang/CS_QMI/outputs/REV9/results/rev9_results.json; rc=$?;;
  phaseb-full) DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL; RF=/home/infres/yinwang/CS_QMI/outputs/REV9_FULL; mkdir -p $RF/selectors
          for t in vcs_lr0.0001 logistic_lr0.0001 direct_lr0.0001; do for s in 0 1 2; do iv="I1 I2"; [ $s -eq 0 ] && iv="I0 I1 I2"
            "$PYTHON" scripts/rev9_phaseb.py run --tag ${t}_s$s --interventions $iv --d1 $DF --out $RF/selectors || rc=$?; done; done;;
  eval-full) DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL; RF=/home/infres/yinwang/CS_QMI/outputs/REV9_FULL; mkdir -p $RF/d1_30/pred
        E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf
        $E/bin/python scripts/r7_d1_30_run.py --sets-dir $D1/d1_30/sets --bench "$1" --configs-file $RF/configs/configs_$1.txt --shard "$2" --out $RF/d1_30/pred \
          --selectors-dir $RF/selectors --pca $DF/cache/pca.pt; rc=$?;;
  score-full) mkdir -p /home/infres/yinwang/CS_QMI/outputs/REV9_FULL/results; "$PYTHON" scripts/rev9_score.py --d1 /home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL \
          --rv /home/infres/yinwang/CS_QMI/outputs/REV9_FULL --tags vcs_lr0.0001 logistic_lr0.0001 direct_lr0.0001 --out /home/infres/yinwang/CS_QMI/outputs/REV9_FULL/results/rev9_full_results.json; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
