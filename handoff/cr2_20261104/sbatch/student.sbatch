#!/usr/bin/env bash
# usage: sbatch --job-name=cr2_stu_<..> student.sbatch <UNIT_ID>
# 1) training process (FIT bank + CR1 prepared + frozen CR2 CSR)  2) separate NR1-bridge DEV evaluator process
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr2_20261104/sbatch/common_env.sh
U=$1
read C ORA < <($CR2_PY -I -c "import json;d=json.load(open('$CR2_TREE/package/configs/units/$U.json'));print(d['condition'],d['oracle'])")
SUB=relations; [ "$ORA" = True ] && SUB=oracle
REL=$CR2_OUT/$SUB/$C/$U.npz; D=$CR2_OUT/students/$C/$U; mkdir -p $D
if [ "$C" = CC152K_RECOVERED_20261104 ]; then
  DEVC=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/cache/dev
  TRUTH=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/evaluator_only/dev_positives.json
else
  DEVC=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/cache/dev
  TRUTH=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/evaluator_only/dev_positives.json
fi
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $D/ATTEMPTS.txt
[ -f $D/DONE.json ] || $CR2_PY -I -B site/cr2_student.py --bank $CR1_OUT/banks/$C/fit_bank.npz --prepared $CR1_OUT/$C/prepared \
   --relation $REL --unit $U --out $D --device cuda
$CR2_PY -I -B -c "import sys;sys.path.insert(0,'$CR2_TREE/code/site');import cr2_student as S;S.frozen_check_cr2('$U')"
$CR2_PY -I -B $CR1_TREE/code/site/eval_student.py --no-frozen-check --student $D --dev-cache $DEVC --truth $TRUTH --prepared $CR1_OUT/$C/prepared --device cuda
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $D/UNIT_COMPLETE.json
echo "EXIT_OK $(date -u +%FT%TZ)"
