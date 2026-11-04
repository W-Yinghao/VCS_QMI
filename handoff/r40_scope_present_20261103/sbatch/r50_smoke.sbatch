#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R50 GPU smoke / replay (gate mode of the production trainer; scratch dirs, never formal, never reused as results):
#   $1 = tree (r50f|r50c), $2 = runner (train_unit.py|train_oracle.py), $3 = units, $4 = gate steps, $5 = smoke|replay.
#   smoke : every (tree, cond, supervision, encoder) cell of R50, readouts at 0 and $4: step-0 native H, R50 layout checks (P/B/C/D/E/F/G), R38 graph check,
#           R39 / R40 one-step checks, PRESENT p_d recheck + masked-row zero gradient, oracle hook; then tools/r50_smoke_check.py.
#   replay: the R47 G / R48 N reference configs on the R50 FIT-fixed cond; then tools/r50_replay_check.py (batch chain + head / null / student state vs the
#           reference checkpoint at the same step; bitwise expected on the reference GPU type).
#   Oracle-runner units write to ORACLE_RUNS/_gate/<job> (refused by trainer processes); others to OUT_R50/r50_gate/<job>.
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"   # R50 amendment A2 tie policy (R50_AMENDMENT_A2.json)
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
R50OUT=/home/infres/yinwang/CS_QMI/outputs_v9_r50
if [ "$2" = train_oracle.py ]; then G=$R50OUT/oracle_runs_r50/_gate/$SLURM_JOB_ID; else G=$R50OUT/r50_gate/$SLURM_JOB_ID; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "start $(date -u +%FT%TZ) $(hostname) $1 $2 steps $4 $5"
$PY -u -B $W/$1/$2 --units "$3" --gate-out $G --gate-steps $4; rc=$?
if [ "$5" = replay ]; then $PY -I -B $W/tools/r50_replay_check.py $G "$3" $rc; else $PY -I -B $W/tools/r50_smoke_check.py $G "$3" $rc; fi
echo "R50_SMOKE_END rc=$rc $(date -u +%FT%TZ)"
