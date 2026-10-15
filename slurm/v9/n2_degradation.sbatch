#!/usr/bin/env bash
#SBATCH --job-name=v9_n2_degradation
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2 degradation check on the REAL Flickr30K precomp data (owner 2026-10-16): the patched NCR / L2RM code with the anchor intervention off and the
# official DECL/RCL noise index must reproduce the native path (same index, labels, items, fixed-batch eval loss, selected indices).  NCR part under
# fork A's v9_sgr env, L2RM part under v9_l2rm (official tokeniser + the baselines' own imports).  Writes reports/V9/V9_N2_degradation/degradation_f30k_<env>.json.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
NCR=${NCR:-/projects/EEG-foundation-model/yinghao/code_v9/2021-NeurIPS-NCR}; L2RM=${L2RM:-/projects/EEG-foundation-model/yinghao/code_v9/L2RM}
DATA=${DATA:-/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/f30k_precomp}; NOISE=${NOISE:-$L2RM/noise_index/f30k_precomp_0.4.npy}
VOCAB=${VOCAB:-/projects/EEG-foundation-model/yinghao/code_v9/SGRAF_py3/vocab/f30k_precomp_vocab.json}; CACHE=${CACHE:-/projects/EEG-foundation-model/yinghao/outputs_v9/n2/token_cache}
OUT=reports/V9/V9_N2_degradation; mkdir -p $OUT
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) patches=$(cat v9/patches/*.diff | sha256sum | cut -c1-16)"
rc=0
for pair in "ncr:/projects/EEG-foundation-model/yinghao/envs/v9_sgr" "l2rm:/projects/EEG-foundation-model/yinghao/envs/v9_l2rm"; do
  only=${pair%%:*}; env=${pair#*:}; echo "[check] $only under $env $(date -u +%FT%TZ)"
  $env/bin/python scripts/v9/n2_degradation.py --ncr $NCR --l2rm $L2RM --only $only --data-root $DATA --noise-file $NOISE --vocab $VOCAB --cache-dir $CACHE --out $OUT/degradation_f30k_$only.json 2>&1 | tail -40; r=${PIPESTATUS[0]}
  echo "[check-exit] $only rc=$r"; [ $r -ne 0 ] && rc=$r
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
