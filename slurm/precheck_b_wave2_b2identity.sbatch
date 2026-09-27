#!/usr/bin/env bash
#SBATCH --job-name=b2identity
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/b2identity_%j.out
# Byte-identity check of the refactored B2 script: the committed version (OLD) and the working-tree version run the same CPU smoke; the
# per-image numbers must agree exactly.  OLD = path of the saved committed script; REPORT = output prefix.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${OLD:?}"; : "${REPORT:?}"
banner "B2 refactor identity check"
"$PYTHON" "$OLD" --out "${REPORT}_old" --smoke || exit $?
"$PYTHON" scripts/precheck_b2_registration.py --out "${REPORT}_new" --smoke || exit $?
"$PYTHON" - "$REPORT" <<'PY'
import json, sys
o = json.load(open(sys.argv[1] + "_old.json")); n = json.load(open(sys.argv[1] + "_new.json"))
same = o["per_image"] == n["per_image"] and o["aggregate"] == n["aggregate"] and o["image_ids"] == n["image_ids"]
print("IDENTICAL" if same else "DIFFERENT")
if not same:
    for a, b in zip(o["per_image"], n["per_image"]):
        for k in ("metrics", "optimisation"):
            if a[k] != b[k]: print("differs:", a["image_id"], k)
PY
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
