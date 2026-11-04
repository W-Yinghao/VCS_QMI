#!/usr/bin/env bash
# NR2D-B prep: truth-free FIT Top100 proposal tables on the NR1 CC152K_RECOVERED cache (read-only), then acceptance.
#SBATCH --job-name=nr2d_cc_prep
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
B=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104; OD=/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2D_20261104
mkdir -p $OD
[ -f $OD/proposal_cc/base_top100_fit.npz ] || $NR1_PY -B site/build_tables_nr2c.py --cache $B/cache/fit --out $OD/proposal_cc --margin 400
export NR1_FEATURE_RAM=global
$NR1_PY -B site/nr2d_cc_acceptance.py
echo NR2D_CC_PREP_DONE
