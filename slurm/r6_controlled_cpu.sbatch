#!/usr/bin/env bash
#SBATCH --job-name=r6_controlled
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6-02 controlled validations; first arg: tests | reg | gain | check (tests + both smokes) (the rest passed through, e.g. --smoke).  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_r6_controlled.py -v; rc=$?;;
  reg) "$PYTHON" scripts/r6_reg_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled --workers 8 "$@"; rc=$?;;
  gain) "$PYTHON" scripts/r6_gain_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled "$@"; rc=$?;;
  check) rc=0; "$PYTHON" -m unittest tests/test_r6_controlled.py -v || rc=1
         "$PYTHON" scripts/r6_reg_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled --workers 8 --smoke || rc=1
         "$PYTHON" scripts/r6_gain_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled/smoke --smoke || rc=1;;
  full) rc=0; "$PYTHON" -m unittest tests/test_r6_controlled.py -v || rc=1
        "$PYTHON" scripts/r6_reg_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled --workers 8 || rc=1
        "$PYTHON" scripts/r6_gain_controlled.py --out /home/infres/yinwang/CS_QMI/outputs/R6_controlled || rc=1;;
  summary) "$PYTHON" scripts/r6_controlled_summary.py; rc=$?;;
  *) echo "usage: tests | reg | gain | check | full | summary"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
