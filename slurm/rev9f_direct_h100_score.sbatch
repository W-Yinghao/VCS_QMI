#!/usr/bin/env bash
#SBATCH --job-name=rev9f_direct_h100_score
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV9 execution note 6: SECONDARY readout of the direct objective from the H100 rerun (seed 0 only), SparseVLM env (note 5), after eval job 1040705.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; cd /home/infres/yinwang/CS_QMI/ssl_pilot; export PYTHONNOUSERSITE=1
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; RV=/home/infres/yinwang/CS_QMI/outputs/REV9_FULL_H100; DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat scripts/rev9_score.py scripts/rev9_score_secondary.py scripts/r7_d1_30_score.py | sha256sum | cut -c1-16)"
for b in gqa_dev textvqa_val chartqa_val; do n=$(ls $RV/d1_30/pred/$b 2>/dev/null | wc -l); echo "pred files $b: $n"; done
mkdir -p $RV/results
$E/bin/python scripts/rev9_score_secondary.py --d1 $DF --rv $RV --tags direct_lr0.0001 --seeds 0 --out $RV/results/rev9_full_h100_direct_secondary.json; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
