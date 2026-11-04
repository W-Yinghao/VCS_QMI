#!/usr/bin/env bash
# usage: sbatch --job-name=cr3_<..> student.sbatch <UNIT_ID>   (GPU locked: RTX PRO 6000)
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr3_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr3_20261104/sbatch/common_env.sh
U=$1
read C CONS REL < <($CR3_PY -I -c "import json;d=json.load(open('$CR3_TREE/package/configs/units/$U.json'));f=json.load(open('$CR3_TREE/PROTOCOL_FROZEN_CR3.json'));print(d['condition'],d['consumer'],f['graph_paths']['$U'])")
D=$CR3_OUT/students/$C/$U; mkdir -p $D
if [ "$C" = CC152K_RECOVERED_20261104 ]; then
  DEVC=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/cache/dev
  TRUTH=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/evaluator_only/dev_positives.json
else
  DEVC=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/cache/dev
  TRUTH=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/evaluator_only/dev_positives.json
fi
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $D/ATTEMPTS.txt
[ -f $D/DONE.json ] || $CR3_PY -I -B site/cr3_student.py --bank $CR1_OUT/banks/$C/fit_bank.npz --prepared $CR1_OUT/$C/prepared \
   --relation $REL --unit $U --consumer $CONS --out $D --device cuda
$CR3_PY -I -B -c "import sys;sys.path.insert(0,'$CR3_TREE/code/site');import cr3_student as S;S.frozen_check_cr3('$U')"
$CR3_PY -I -B $CR1_TREE/code/site/eval_student.py --no-frozen-check --student $D --dev-cache $DEVC --truth $TRUTH --prepared $CR1_OUT/$C/prepared --device cuda
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $D/UNIT_COMPLETE.json
echo "EXIT_OK $(date -u +%FT%TZ)"
