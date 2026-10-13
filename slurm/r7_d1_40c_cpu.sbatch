#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_40c_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-40c CPU steps.  First arg: tests (main env) | tests_vl (vl_baselines: transformers checks) | tests_sv (SparseVLM env: m4c TextVQA
# scorer) | tests_all (the three envs in one job) | check [rows] (stage 0) | gridcheck (image headers of the D1-30 sets: token counts, grids vs 64 × 64; no answer) | complete (stage 2) | score (the coordinator's step after the driver stops; SparseVLM env for the m4c
# evaluator; refuses without the freeze key) | smoke_prep | smoke_report.  Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R40=$D1/r7_40; RC=$R40/eval_40c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: $(cat scripts/r7_d1_40c_*.py tests/test_r7_d1_40c.py slurm/r7_d1_40c_* 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the R7-40c scripts, tests and wrappers)"
what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_r7_d1_40c.py -v; rc=$?;;
  tests_vl) $Q/bin/python -m unittest tests/test_r7_d1_40c.py -v; rc=$?;;
  tests_sv) $E/bin/python -m unittest tests/test_r7_d1_40c.py -v; rc=$?;;
  tests_all)   # the three environments in one job (main, vl_baselines, SparseVLM); rc = number of failing environments
    rc=0; for py in "$PYTHON" "$Q/bin/python" "$E/bin/python"; do echo "=== $py"; "$py" -m unittest tests/test_r7_d1_40c.py 2>&1 | grep -v '^{"bench"' | tail -25
      [ "${PIPESTATUS[0]}" -eq 0 ] || rc=$((rc + 1)); done;;
  check) "$PYTHON" -I scripts/r7_d1_40c_run.py check --sets-dir "$D1/d1_30/sets" --configs-dir "$RC/configs" --lr-choice "$R40/selectors/lr_choice.json" \
           --out "$RC/check" --rows "${1:-core}"; rc=$?;;
  gridcheck) "$PYTHON" -I scripts/r7_d1_40c_run.py gridcheck --sets-dir "$D1/d1_30/sets" --out "$RC/gridcheck"; rc=$?;;   # image headers only (no answer)
  complete) "$PYTHON" -I scripts/r7_d1_40c_run.py complete --sets-dir "$D1/d1_30/sets" --configs-dir "$RC/configs" --pred-dir "$RC/pred" --out "$RC"; rc=$?;;
  score) $E/bin/python -I scripts/r7_d1_40c_score.py --sets-dir "$D1/d1_30/sets" --pred-dir "$RC/pred" --timing-dir "$RC/timing" --configs-dir "$RC/configs" \
           --lr-choice "$R40/selectors/lr_choice.json" --out "${1:-$RC/results}"; rc=$?;;
  smoke_prep) $E/bin/python -I scripts/r7_d1_40c_smoke.py prep --out "$R40/smoke_40c"; rc=$?;;
  smoke_report) $E/bin/python -I scripts/r7_d1_40c_smoke.py report --out "$R40/smoke_40c"; rc=$?;;
  *) echo "usage: tests | tests_vl | tests_sv | tests_all | check [core|static|dynamic|all] | gridcheck | complete | score [out dir] | smoke_prep | smoke_report"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
