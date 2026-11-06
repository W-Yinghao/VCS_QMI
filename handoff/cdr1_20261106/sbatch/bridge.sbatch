#!/bin/bash
#SBATCH --job-name=cdr1_bridge
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/logs/%x_%j.out
set -euo pipefail
PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
B=/projects/EEG-foundation-model/yinghao/outputs_cdr1/CDR1_20261106/curator_only/banks
echo "host $(hostname) start $(date -u +%FT%TZ)"
for DS in Flickr30K COCO; do $PY -I -B /home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/site/export_bank.py --dataset $DS --out $B/$DS; done
chmod 700 /projects/EEG-foundation-model/yinghao/outputs_cdr1/CDR1_20261106/curator_only
echo "BRIDGE_DONE $(date -u +%FT%TZ)"
