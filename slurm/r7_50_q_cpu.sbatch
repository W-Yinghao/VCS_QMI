#!/usr/bin/env bash
#SBATCH --job-name=r7_50q_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-50 phase Q CPU steps (addendum reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml) in the SparseVLM env, as slurm/r7_50_cpu.sbatch
# (pyarrow / PIL for the gate's development-mode build; the pinned evaluators).  python -I.  Positional arguments only (no --export).
#   integrity-prepare <root>  |  integrity-compare <root>     full-scale gate on GQA DEV / ChartQA val (non-test data; Q3 / Q6)
#   configs <root>            configs_Q (41 rows, unit / PCA / lr_choice sha256, image-grid check of the sealed inputs) + lines_Q
#   manifest <root>           manifest_Q (coverage, every prediction file, rows.json, build summary, R7-40c's timing bundle files + B* report)
#   score <root> [out dir]    the ONLY phase-Q job that opens the sealed labels (after ALLOW_SCORING_Q); a re-verification writes to a new dir
#   tests | tests_vl          tests/test_r7_50_q.py in this env | in vl_baselines (the GPU units' env)
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; PY="$E/bin/python -I"; export PYTHONNOUSERSITE=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; what="$1"; shift
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: $(cat scripts/r7_50_q.py tests/test_r7_50_q.py slurm/r7_50_q_* 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the phase-Q script, tests and wrappers)"
[ -n "$1" ] || case "$what" in tests|tests_vl) ;; *) echo "usage: see the header"; echo "exit=2 end=$(date -u +%FT%TZ)"; exit 2;; esac
case "$what" in
  integrity-prepare|integrity-compare|manifest)
    $PY scripts/r7_50_q.py "$what" --root "$1"; rc=$?;;
  configs)
    $PY scripts/r7_50_q.py configs --root "$1"; rc=$?
    [ $rc -eq 0 ] && { $PY scripts/r7_50_q.py lines --root "$1"; rc=$?; };;
  score)
    X=(); [ -n "$2" ] && X=(--out "$2")
    $PY scripts/r7_50_q.py score --root "$1" "${X[@]}"; rc=$?;;
  tests) $PY tests/test_r7_50_q.py -v; rc=$?;;
  tests_vl) /projects/EEG-foundation-model/yinghao/envs/vl_baselines/bin/python -I tests/test_r7_50_q.py -v; rc=$?;;
  *) echo "usage: see the header"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
