#!/usr/bin/env bash
#SBATCH --job-name=r7_e2_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 E2 CPU steps (main env unless noted; python -I).  First arg:
#   tests | tests_vl | tests_sv     the E2 CPU tests in the main / vl_baselines / SparseVLM env (python -I tests/test_r7_e2.py; '-m unittest' fails under -I)
#   precheck [...]                  pre-freeze dry run: source verification + static reuse check of R7-40a's CAL / FIT records (+ replay result if present)
#   data <lines file>               the driver's data stage (refuses without the teacher-stage key r7_e2a_staged_freeze)
#   complete                        the frozen R7-40a completeness check on E2's teacher dir (every CAL / FIT_FULL shard, manifest sha256)
#   cachecheck <fitfull|frozen10k>  the E2 cache check (the cache job's outputs against E2's manifest)
#   choose                          the frozen learning-rate choice on E2's seed-0 units
#   check                           the frozen R7-40c stage-0 check with E2's units / PCA / lr_choice (writes E2's 50 core configs)
#   runcomplete                     the frozen R7-40c completeness of the rows E2 generated (eval/run configs)
#   reuserows <selectors|core>      R7-40c's selector-independent rows vs E2's run (+ the grid-row copy when selectors and passing)
#   corecomplete                    the frozen R7-40c completeness of the 50 core rows in E2's prediction dir
#   score [out dir]                 the E2 readout (SparseVLM env; refuses without both freeze keys): the coordinator's step, never the driver's
# Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; E2=/home/infres/yinwang/CS_QMI/outputs/R7_E2
FULL_SHA=c4404f7be19640385072c39e27ce026589acbfcb1187aedffa1bd58c638fb21e; P=reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: E2 $(cat scripts/r7_e2_*.py tests/test_r7_e2.py slurm/r7_e2_* 2>/dev/null | sha256sum | cut -c1-16)"
what="$1"; shift
case "$what" in
  tests) "$PYTHON" -I tests/test_r7_e2.py -v; rc=$?;;
  tests_vl) $Q/bin/python -I tests/test_r7_e2.py -v; rc=$?;;
  tests_sv) $E/bin/python -I tests/test_r7_e2.py -v; rc=$?;;
  precheck) "$PYTHON" -I scripts/r7_e2_data.py precheck "$@"; rc=$?;;
  data) [ -n "${1:-}" ] || { echo "data <lines file>"; exit 2; }; "$PYTHON" -I scripts/r7_e2_data.py data --lines "$1"; rc=$?;;
  complete) "$PYTHON" -I scripts/r7_d1_40_analyze.py complete --teacher-dir "$E2/teacher" --manifest-dir "$E2/data" --frozen-sha256 "$FULL_SHA"; rc=$?;;
  cachecheck) case "${1:-}" in fitfull|frozen10k) ;; *) echo "cachecheck <fitfull|frozen10k>"; exit 2;; esac
              "$PYTHON" -I scripts/r7_e2_data.py cachecheck --basis "$1"; rc=$?;;
  choose) "$PYTHON" -I scripts/r7_e2_plan.py choose --selectors-dir "$E2/selectors"; rc=$?;;
  check) "$PYTHON" -I scripts/r7_d1_40c_run.py check --sets-dir "$D1/d1_30/sets" --configs-dir "$E2/eval/configs" --lr-choice "$E2/selectors/lr_choice.json" \
           --selectors-dir "$E2/selectors" --pca "$E2/cache/pca.pt" --out "$E2/eval/check" --rows core; rc=$?;;
  runcomplete) "$PYTHON" -I scripts/r7_d1_40c_run.py complete --sets-dir "$D1/d1_30/sets" --configs-dir "$E2/eval/run" --pred-dir "$E2/eval/pred" \
                 --out "$E2/eval/run" --no-official; rc=$?;;
  reuserows) case "${1:-}" in selectors|core) ;; *) echo "reuserows <selectors|core>"; exit 2;; esac
             "$PYTHON" -I scripts/r7_e2_data.py reuserows --rows "$1"; rc=$?;;
  corecomplete) "$PYTHON" -I scripts/r7_d1_40c_run.py complete --sets-dir "$D1/d1_30/sets" --configs-dir "$E2/eval/configs" --pred-dir "$E2/eval/pred" \
                  --out "$E2/eval" --no-official; rc=$?;;
  score) { grep -q '^r7_e2a_staged_freeze:' "$P" && grep -q '^r7_e2_staged_freeze:' "$P"; } || { echo "E2 is not frozen ($P lacks r7_e2a / r7_e2 staged-freeze keys)"; exit 3; }
         $E/bin/python -I scripts/r7_e2_score.py ${1:+--out "$1"}; rc=$?;;
  *) echo "usage: tests | tests_vl | tests_sv | precheck | data <lines> | complete | cachecheck <basis> | choose | check | runcomplete | reuserows <rows> | corecomplete | score [out]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
