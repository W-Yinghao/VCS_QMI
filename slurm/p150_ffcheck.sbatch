#!/usr/bin/env bash
#SBATCH --job-name=p150_ffcheck
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P150: verify the --only-cell fast-forward: the null-only smoke must reproduce the gate smoke's null draws (outputs/P150_smoke/c0.json).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
"$PYTHON" scripts/p150_c0.py --features $OUTPUT_ROOT/P150_features --run P107_AP3_views4_800ep_seed1 --out $OUTPUT_ROOT/P150_smoke/c0_nullonly --smoke --only-cell null; rc=$?
"$PYTHON" - $OUTPUT_ROOT/P150_smoke <<'PY' || rc=1
import json, sys
a = json.load(open(sys.argv[1] + "/c0.json"))["cells"]["null"]["repeats"]; b = json.load(open(sys.argv[1] + "/c0_nullonly.json"))["cells"]["null"]["repeats"]
dmax = 0.0
for ra, rb in zip(a, b):
    assert ra["n1_frac_eval"] == rb["n1_frac_eval"], "draws differ"
    for site in ra["sites"]:
        for nm, ca in ra["sites"][site]["critics"].items():
            cb = rb["sites"][site]["critics"][nm]; dmax = max(dmax, abs(ca["J_common_exactQ"] - cb["J_common_exactQ"]))
            assert ca["perm_p"] == cb["perm_p"] or nm.startswith("NEST"), (site, nm, ca["perm_p"], cb["perm_p"])
print("FFCHECK draws identical; max |dJ| over critics", dmax); sys.exit(0 if dmax < 1e-4 else 1)
PY
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
