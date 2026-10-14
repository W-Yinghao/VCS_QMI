#!/usr/bin/env bash
#SBATCH --job-name=p155_ap3_official_test
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P155: official CIFAR-10 test of the final recipe A-P3 (owner 2026-10-14).  Gate first: the stand-in (selection split) of unit 1 must
# reproduce its selection-split linear accuracy within 0.1 points; only then is the official partition opened (VCS_FINAL_ROUND=1).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
U=slurm/p155_official_test_units.txt
banner "P155 stand-in gate (selection split, unit 1)"
"$PYTHON" scripts/final_official_test_eval.py --units "$U" --out reports/P155_AP3_OFFICIAL_TEST_standin --standin --limit 1 || exit 4
"$PYTHON" - <<'PY' || exit 5
import json
r = json.load(open("reports/P155_AP3_OFFICIAL_TEST_standin.json"))["rows"][0]
d = abs(r["test_linear"] - r["sel_linear"]); print("stand-in gate", r["run_id"], r["test_linear"], r["sel_linear"], "diff", d, flush=True)
raise SystemExit(0 if d <= 0.1 else 1)
PY
banner "P155 OFFICIAL CIFAR-10 TEST partition (owner 2026-10-14: A-P3 is the final recipe)"
export VCS_FINAL_ROUND=1
"$PYTHON" scripts/final_official_test_eval.py --units "$U" --out reports/P155_AP3_OFFICIAL_TEST; rc=$?
[ $rc -eq 0 ] && "$PYTHON" scripts/p155_contrasts.py --test reports/P155_AP3_OFFICIAL_TEST.json --out reports/P155_AP3_OFFICIAL_TEST_contrasts; rc2=$?
echo "exit=$rc contrasts=${rc2:-NA} end=$(date -u +%FT%TZ)"; exit $rc
