#!/usr/bin/env bash
# usage: sbatch --job-name=nr3_prep_<cond> prep.sbatch <COND>
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr3_20261104/sbatch/common_env.sh
C=$1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
pth() { $NR1_PY -I -c "import sys;sys.path.insert(0,'$NR3_H/code');from nr3.paths import cond_paths;print(cond_paths('$C')['$1'])"; }
DEVC=$(pth dev); TRUTH=$(pth truth); OD=$NR3_OUT/prep/$C
export NR1_FEATURE_RAM=global
$NR1_PY -B -m nr3.prep --cond $C
unset NR1_FEATURE_RAM
[ -f $OD/eval_prefix/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $DEVC --truth $TRUTH --checkpoint $OD/prefix/PREFIX_step_008000.pt --out $OD/eval_prefix --device cuda --score-dtype float64
$NR1_PY -B -m nr3.prefix_order_check --cond $C
export NR1_FEATURE_RAM=global
$NR1_PY -B -m nr3.d0 --cond $C
$NR1_PY -B -m nr3.d0_truth_evalonly --cond $C
$NR1_PY -B -m nr3.acceptance --cond $C
echo NR3_PREP_DONE $C
