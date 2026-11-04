#!/usr/bin/env bash
# usage: sbatch --job-name=cr1_stu_<COND>_<ARM> student.sbatch <COND> <ARM>
# 1) training process (FIT bank + prepared + relation only)  2) separate DEV evaluator process (NR1 bridge)
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr1_20261104/sbatch/common_env.sh
C=$1; A=$2; U=CR1_${C}_TASK_${A}_s1001; D=$CR1_OUT/$C/students/$A; mkdir -p $D
REL=$CR1_OUT/$C/relations/$A.npz; [ "$A" = OBS ] && REL=OBS
if [ "$C" = CC152K_RECOVERED_20261104 ]; then
  DEVC=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/cache/dev
  TRUTH=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104/evaluator_only/dev_positives.json
else
  DEVC=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/cache/dev
  TRUTH=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104/evaluator_only/dev_positives.json
fi
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $D/ATTEMPTS.txt
[ -f $D/DONE.json ] || $CR1_PY -I -B site/cr1_prod.py train-student --bank $CR1_OUT/banks/$C/fit_bank.npz \
   --prepared $CR1_OUT/$C/prepared --unit $U --relation $REL --out $D --device cuda
$CR1_PY -I -B site/eval_student.py --student $D --dev-cache $DEVC --truth $TRUTH --prepared $CR1_OUT/$C/prepared --device cuda
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $D/UNIT_COMPLETE.json
echo "EXIT_OK $(date -u +%FT%TZ)"
