#!/bin/bash
# Submit one layout-free TEST re-read job: submit_layoutfree.sh TAG GPUKEY DATASET [extra args for test_eval_layoutfree.py]
# GPUKEY: rtx | l40s | v100_16 | v100s -> recorded evaluation GPU name + partition (normal QOS, nice 50 in the sbatch file).
set -euo pipefail
T=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101; cd $T
TAG=$1; K=$2; DS=$3; shift 3
case $K in
  rtx)     G="NVIDIA RTX PRO 6000 Blackwell Server Edition"; P="--partition=RTX6000PRO --exclude=node60";;
  l40s)    G="NVIDIA L40S"; P="--partition=L40S --exclude=node51";;
  v100_16) G="Tesla V100-PCIE-16GB"; P="--partition=V100-16GB --exclude=node42";;          # node42 mixes 16GB / 32GB cards
  v100s)   G="Tesla V100S-PCIE-32GB"; P="--partition=V100-32GB --exclude=node08,node10,node12";;  # recorded V100S nodes: node15, node43
  *) echo "unknown GPU key $K"; exit 2;;
esac
j=$(sbatch --parsable --job-name=te_lf_$TAG $P sbatch/layoutfree.sbatch "$G" $DS "$@")
printf "%s\t%s\tte_lf_%s\tsbatch %s sbatch/layoutfree.sbatch '%s' %s %s\ttools/submit_layoutfree.sh\n" "$(date -u +%FT%TZ)" "$j" "$TAG" "$P" "$G" "$DS" "$*" >> SUBMITTED.tsv
echo "te_lf_$TAG $j"
