#!/bin/bash
#SBATCH --job-name=te_increment
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,V100,L40S,A100,A40,3090,A30
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/logs/%x_%j.out
# Incremental TEST step: (1) freeze TEST_MANIFEST_INC_<utc>.json with newly finished units (exit 3 = none new),
# (2) evaluate every not-yet-evaluated unit of all frozen manifests (DEV gate + TEST), (3) rebuild the tables.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd /home/infres/yinwang/CS_QMI/handoff/test_eval_20261101
$PY -B -u tools/build_manifest.py --increment; rc=$?
if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then echo "manifest increment failed rc=$rc"; exit $rc; fi
$PY -B -u tools/test_eval.py --host in2r --dataset all || exit 1
$PY -B -u tools/test_eval.py --host sps --dataset all || exit 1
$PY -B -u tools/test_eval.py --host nrcs --dataset all || exit 1
$PY -B -u tools/aggregate_test.py || exit 1
echo TE_INCREMENT_DONE
