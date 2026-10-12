#!/usr/bin/env bash
#SBATCH --job-name=r7_50_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-50 CPU units in the SparseVLM env (pyarrow / PIL for the build; the pinned evaluators).  python -I.  Positional arguments only (no --export).
#   prepare <root> <frozen protocol>                       build (the ONLY job that opens the two official test files) + configs_L + lines_L
#   configs <root> <phase> [lr_choice selectors_dir pca]  configs_<P> + lines_<P>
#   integrity-prepare <root> <phase> [lr_choice selectors_dir pca ref_pred]   |   integrity-compare <root> <phase>   (full-scale gate; non-test data)
#   manifest <root> <phase> [extra_timing_dir]  |  score <root> <phase> [extra_timing_dir]   (phase F: D1-FULL's timing run; score = the ONLY job
#   that opens the sealed labels)
#   tests | smoke-prep <S> | smoke-finish <S> | smoke-prep-integrity <S> <name> <chartqa_qid_prefix> | smoke-integrity <S> <name>
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; PY="$E/bin/python -I"; export PYTHONNOUSERSITE=1
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; TM="--timing-dir $D1/d1_30_smoke/timing --timing-bench gqa_testdev=gqa_fixture chartqa_test=chartqa_proxy"
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; what="$1"; shift
echo "host=$(hostname) start=$(date -u +%FT%TZ) git=$(git rev-parse --short HEAD 2>/dev/null)"
case "$what" in
  prepare)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: prepare <root> <frozen protocol>"; exit 2; }
    $PY scripts/r7_50_test.py prepare --root "$1" --protocol "$2"; rc=$?;;
  configs)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: configs <root> <phase> [lr_choice selectors_dir pca]"; exit 2; }
    X=(); [ -n "$3" ] && X=(--lr-choice "$3" --selectors-dir "$4" --pca "$5")
    $PY scripts/r7_50_test.py configs --root "$1" --phase "$2" "${X[@]}"; rc=$?
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py lines --root "$1" --phase "$2"; rc=$?; };;
  integrity-prepare)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: integrity-prepare <root> <phase> [lr_choice selectors_dir pca ref_pred]"; exit 2; }
    X=(); [ -n "$3" ] && X=(--lr-choice "$3" --selectors-dir "$4" --pca "$5" --ref-pred "$6")
    $PY scripts/r7_50_test.py integrity-prepare --root "$1" --phase "$2" "${X[@]}"; rc=$?;;
  integrity-compare)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: integrity-compare <root> <phase>"; exit 2; }
    $PY scripts/r7_50_test.py integrity-compare --root "$1" --phase "$2"; rc=$?;;
  manifest)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: manifest <root> <phase> [extra_timing_dir]"; exit 2; }
    X=(); [ -n "$3" ] && X=(--extra-timing-dir "$3")
    $PY scripts/r7_50_test.py manifest --root "$1" --phase "$2" "${X[@]}"; rc=$?;;
  score)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: score <root> <phase> [extra_timing_dir]"; exit 2; }
    X=(); [ -n "$3" ] && X=(--extra-timing-dir "$3")
    $PY scripts/r7_50_test.py score --root "$1" --phase "$2" "${X[@]}"; rc=$?;;
  tests) $PY tests/test_r7_50.py -v; rc=$?;;
  smoke-prep)
    [ -n "$1" ] || { echo "usage: smoke-prep <smoke_root>"; exit 2; }
    $PY scripts/r7_50_smoke.py prep --smoke "$1"; rc=$?;;
  smoke-prep-integrity)
    [ -n "$1" ] && [ -n "$2" ] && [ -n "$3" ] || { echo "usage: smoke-prep-integrity <smoke_root> <integrity_root_name> <chartqa_qid_prefix>"; exit 2; }
    $PY scripts/r7_50_smoke.py prep-integrity --smoke "$1" --integrity-root "$2" --chartqa-qid-prefix "$3"; rc=$?;;
  smoke-integrity)
    [ -n "$1" ] && [ -n "$2" ] || { echo "usage: smoke-integrity <smoke_root> <integrity_root_name>"; exit 2; }
    $PY scripts/r7_50_smoke.py integrity --smoke "$1" --integrity-root "$2"; rc=$?
    $PY scripts/r7_50_smoke.py report --smoke "$1"; rc2=$?; [ $rc -eq 0 ] && rc=$rc2;;
  smoke-finish)
    S="$1"; [ -n "$S" ] || { echo "usage: smoke-finish <smoke_root>"; exit 2; }
    $PY scripts/r7_50_test.py manifest --root "$S/scored" --phase L $TM; rc=$?
    [ $rc -eq 0 ] && { $PY scripts/r7_50_smoke.py ftiming --smoke "$S"; rc=$?; }
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py manifest --root "$S/scored" --phase F $TM --extra-timing-dir "$S/timing_F"; rc=$?; }
    [ $rc -eq 0 ] && { $PY scripts/r7_50_smoke.py allow --smoke "$S"; rc=$?; }
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py score --root "$S/scored" --phase L $TM --reps 200; rc=$?; }
    [ $rc -eq 0 ] && { $PY scripts/r7_50_test.py score --root "$S/scored" --phase F $TM --extra-timing-dir "$S/timing_F" --reps 200; rc=$?; }
    [ $rc -eq 0 ] && { $PY scripts/r7_50_smoke.py integrity --smoke "$S"; rc=$?; }
    $PY scripts/r7_50_smoke.py report --smoke "$S"; rc2=$?; [ $rc -eq 0 ] && rc=$rc2;;
  *) echo "usage: see the header"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
