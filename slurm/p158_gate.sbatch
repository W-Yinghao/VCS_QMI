#!/usr/bin/env bash
#SBATCH --job-name=p158_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P158 CPU gate: tests, the --smoke runner (all three conditions), the oracle values of the pre-named conditions, and a full-size timing
# (weak, 1 repeat, n 3000, 300 steps, with bootstrap).  Numbers are code-path evidence, not results.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P158_GATE; mkdir -p $G; rc=0; banner "P158 gate"; export CUDA_VISIBLE_DEVICES=""
"$PYTHON" -m pytest -q tests/test_p158.py tests/test_p118.py -p no:cacheprovider > $G/pytest.txt 2>&1 || rc=1; tail -3 $G/pytest.txt
"$PYTHON" scripts/p158_truth.py --smoke --out $G/truth_smoke > $G/truth_smoke.txt 2>&1 || rc=1; tail -3 $G/truth_smoke.txt | cut -c1-400
"$PYTHON" scripts/p158_aggregate.py --in $G/truth_smoke --out $G/truth_smoke_agg > $G/agg_smoke.txt 2>&1 || rc=1; tail -1 $G/agg_smoke.txt
"$PYTHON" - <<'PY' > $G/oracle_values.txt 2>&1 || rc=1
import json, sys; sys.path.insert(0, "scripts"); import p158_common as C, p158_truth as T
out = {c: C.oracle(r, n=400_000) for c, r in T.CONDS.items()}
json.dump(out, open("reports/P158_GATE/oracle_values.json", "w"), indent=1)
for c, o in out.items():
    s = o["sets"]; st = o["steps"]
    print(f"{c} rho={o['rho']}: S_B {s['B']['S']:.5f} S_M {s['M']['S']:.5f} (se {s['M']['S_se']:.1e}) S_A {s['A']['S']:.5f}; G_B {s['B']['G']:.5f} G_M {s['M']['G']:.5f} G_A {s['A']['G']:.5f}; "
          f"true M->A: S {st['M->A']['S_diff']:.5f} (L2 {st['M->A']['L2_increment']:.5f}) G {st['M->A']['G_diff']:.5f} (Bregman {st['M->A']['Bregman_increment']:.5f}, se {st['M->A']['Bregman_se']:.1e})")
PY
cat $G/oracle_values.txt
/usr/bin/env bash -c "time '$PYTHON' scripts/p158_truth.py --conds weak --repeats 1 --out $G/truth_timing" > $G/truth_timing.txt 2>&1 || rc=1; tail -4 $G/truth_timing.txt | cut -c1-400
echo "gate_rc=$rc exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
