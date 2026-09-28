#!/usr/bin/env bash
#SBATCH --job-name=estim_v2_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_v2_gate_%j.out
# Estimator package v2 CPU gate (spec §12): the v2 test suite + the stage-0 suite + the reference-core suite, the truth table of the P85 generator,
# the oracle-table re-derivation vs CS_QMI/RESULTS_20260928.md, the P85 runner smoke (two cells, all methods, tiny roles), the fresh-stream
# staircase smoke, and both aggregators.  No GPU, no image data.  OUT (default outputs/P85_gate_<jobid>) holds every artefact.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8
OUT=${OUT:-$OUTPUT_ROOT/P85_gate_${SLURM_JOB_ID}}; mkdir -p "$OUT"
banner "estimator package v2 gate -> $OUT"
"$PYTHON" -m pytest -q tests/test_vcs_estim_v2.py -p no:cacheprovider 2>&1 | tail -40; rc1=${PIPESTATUS[0]}
"$PYTHON" -m pytest -q tests/test_vcs_estim_stage0.py -p no:cacheprovider 2>&1 | tail -5; rc2=${PIPESTATUS[0]}
"$PYTHON" -m pytest -q tests/test_ssl_core.py -p no:cacheprovider 2>&1 | tail -3; rc3=${PIPESTATUS[0]}
echo "== truth table (S of the P85 generator; picks I_PAD)"; PYTHONPATH=src "$PYTHON" -m vcs_estim.benchmark --truth-table --out "$OUT/truth_table.json" | "$PYTHON" -c "
import json,sys; rows=json.load(sys.stdin)
for r in rows: print(f\"{r['setting']:>9s} d={r['d']:>2d} I={r['I']:>4g} rho={r['rho']:.4f} S={r['S']:.4f}±{r['S_se']:.4f} JS2={r['JS2']:.4f} CS_leb={r['CS_lebesgue']}\")"; rc4=${PIPESTATUS[0]}
echo "== oracle table vs RESULTS_20260928.md"; "$PYTHON" scripts/estim_oracle_table.py "$OUT/oracle_table" | tail -8; rc5=${PIPESTATUS[0]}
"$PYTHON" scripts/estim_make_units.py --out slurm/p85_units --chunks 8; rc6=$?
echo "== P85 runner smoke (2 cells, CPU, tiny roles)"; t0=$(date +%s)
"$PYTHON" scripts/estim_p85.py --units slurm/p85_units/smoke.txt --out "$OUT/p85_smoke" --smoke --cpu 2>&1 | tail -30; rc7=${PIPESTATUS[0]}
echo "p85 smoke seconds: $(( $(date +%s) - t0 ))"
"$PYTHON" scripts/estim_p85_aggregate.py --inputs "$OUT/p85_smoke" --out "$OUT/p85_smoke_aggregate" | tail -3; rc8=${PIPESTATUS[0]}
echo "== staircase (fresh-stream) smoke: 2 cells"; t1=$(date +%s)
head -2 slurm/p85_units/staircase_probe.txt > "$OUT/staircase_units.txt"
"$PYTHON" scripts/estim_staircase.py --units "$OUT/staircase_units.txt" --out "$OUT/staircase_smoke" --smoke 2>&1 | tail -5; rc9=${PIPESTATUS[0]}
echo "staircase smoke seconds: $(( $(date +%s) - t1 ))"
"$PYTHON" scripts/estim_aggregate.py --inputs "$OUT/staircase_smoke" --out "$OUT/staircase_smoke_aggregate" | tail -2; rc10=${PIPESTATUS[0]}
echo "rc: v2_tests=$rc1 stage0=$rc2 ssl_core=$rc3 truth=$rc4 oracle=$rc5 units=$rc6 p85_smoke=$rc7 p85_agg=$rc8 staircase=$rc9 staircase_agg=$rc10 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ] && [ $rc5 -eq 0 ] && [ $rc6 -eq 0 ] && [ $rc7 -eq 0 ] && [ $rc8 -eq 0 ] && [ $rc9 -eq 0 ] && [ $rc10 -eq 0 ]
