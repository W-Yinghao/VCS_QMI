#!/bin/bash
#SBATCH --job-name=cdr1_synth_iface
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/logs/%x_%j.out
set -euo pipefail
export CDR1_TEST_OUT=/projects/EEG-foundation-model/yinghao/outputs_cdr1/_synthetic_iface_test
B=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/sbatch
bash $B/group.sbatch Flickr30K 1001
bash $B/group.sbatch COCO 1001
bash $B/transfer.sbatch 1001
echo SYNTH_ALL_OK
