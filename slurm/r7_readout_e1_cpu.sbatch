#!/usr/bin/env bash
#SBATCH --job-name=r7_readout_e1
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 E1 readout (CPU): scripts/r7_e1_readout.py — the frozen R7-E1 analysis (D-E1-7: Bench40 / item_scores / Boot unchanged, no contrast with S1 / S2 /
# S3 / grid).  This wrapper's name lies outside slurm/r7_e1_*, so the code hash the frozen E1 wrappers print (519635f47ebbe9c0) is unchanged.
# First arg: readout [out dir] [method ...] (SparseVLM env for the m4c TextVQA evaluator, as R7-40c's score step; refuses without the key
# r7_e1_staged_freeze and a complete eval_e1/completeness_e1.json; never overwrites; R7-40c's own readout, when present, is used only as an equality
# cross-check of the R7-40c rows recomputed here; methods default to both — name one for a disclosed gap, D-E1-4) | tests (main env) | tests_sv
# (SparseVLM env: the m4c TextVQA path) | tests_all (both envs) | complete <method> [method ...] (stage 2 for a subset of methods — only for a
# disclosed gap, D-E1-4; with both methods use the frozen 'slurm/r7_e1_cpu.sbatch complete anchor').  Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R40=$D1/r7_40; E1=$R40/eval_e1; RC=$R40/eval_40c
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: $(cat scripts/r7_e1_readout.py tests/test_r7_e1_readout.py slurm/r7_readout_e1_cpu.sbatch 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the E1 readout, its tests and this wrapper)"
what="$1"; shift
case "$what" in
  readout)
    out="${1:-$E1/results}"; shift; meth=(); [ $# -gt 0 ] && meth=(--methods "$@")
    xc=(); [ -f "$RC/results/r7_40c_results.json" ] && xc=(--r7-40c-results "$RC/results/r7_40c_results.json")
    $E/bin/python -I scripts/r7_e1_readout.py --sets-dir "$D1/d1_30/sets" --pred-dir "$E1/pred" --configs-dir "$E1/configs" --r40c-pred-dir "$RC/pred" \
        --r40c-configs-dir "$RC/configs" --lr-choice "$R40/selectors/lr_choice.json" --completeness "$E1/completeness_e1.json" --out "$out" "${meth[@]}" "${xc[@]}"; rc=$?;;
  tests) "$PYTHON" -I tests/test_r7_e1_readout.py -v; rc=$?;;
  tests_sv) $E/bin/python -I tests/test_r7_e1_readout.py -v; rc=$?;;
  tests_all)   # main env and SparseVLM env in one job; rc = number of failing environments
    rc=0; for py in "$PYTHON" "$E/bin/python"; do echo "=== $py"; "$py" -I tests/test_r7_e1_readout.py -v 2>&1 | tail -40
      [ "${PIPESTATUS[0]}" -eq 0 ] || rc=$((rc + 1)); done;;
  complete)
    [ $# -gt 0 ] || { echo "complete <method> [method ...]"; exit 2; }
    "$PYTHON" -I scripts/r7_e1_external.py complete --pred-dir "$E1/pred" --out "$E1" --anchor --methods "$@"; rc=$?;;
  *) echo "usage: readout [out dir] [method ...] | tests | tests_sv | tests_all | complete <method> [method ...]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
