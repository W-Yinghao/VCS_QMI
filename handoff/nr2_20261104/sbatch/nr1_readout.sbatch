#!/usr/bin/env bash
# Read-only NR2 FIT readout (common product-Q pool / proposal pools) + evaluator-only FIT strata on the existing
# NR1 seed-1001 G checkpoints; writes only under outputs_nr2/.../nr1_readout (NR1 files untouched).
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
TAB=$NR2_OUT/proposal/base_top100_fit.npz
for c in Flickr00 Flickr60 Flickr80; do for o in J BCE NCE; do
  U=NR1_${c}_G_${o}_s1001; R=$NR1_OUT/runs/$U; D=$NR2_OUT/nr1_readout/$U; mkdir -p $D
  $NR1_PY -B site/fit_readout.py --run $R --out $D/FIT_READOUT.json --fit-cache $NR1_OUT/cache/fit --observed $NR1_OUT/data/observed_$c.tsv --tables $TAB
  $NR1_PY -B site/fit_strata_evalonly.py --run $R --out $D/EVALONLY_FIT_STRATA.json --fit-cache $NR1_OUT/cache/fit \
     --observed $NR1_OUT/data/observed_$c.tsv --clean-observed $NR1_OUT/data/observed_Flickr00.tsv
done; done
echo NR1_READOUT_DONE
