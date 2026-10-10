#!/usr/bin/env bash
#SBATCH --job-name=r4_round1
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V6 server package r4, round 1 (read-only / repair): package reference tests, R4-01 regression test + corrected P151 aggregation,
# R4-E0 P134 audit (new output dir), R4-00 intake hashes of VL3 checkpoints and VL1-21 proposals.  No training, no raw file modified.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0; R=/home/infres/yinwang/CS_QMI/ssl_pilot
echo "== package r4 reference tests"; (cd $R/r4/package_r4_20261010 && "$PYTHON" -m unittest discover -s tests -v) 2>&1 | tail -45 || rc=1
echo "== R4-01 regression test"; (cd $R && "$PYTHON" -m unittest tests/test_p151_r4.py -v) 2>&1 | tail -12 || rc=1
echo "== R4-01 corrected P151 aggregation"; (cd $R && "$PYTHON" scripts/p151_e0_aggregate_r4.py --out reports/R4/P151_r4) || rc=1
echo "== R4-E0 P134 audit"; [ -e $R/reports/R4/P134_audit ] && echo "exists - skipped" || (cd $R/r4/package_r4_20261010/reference && "$PYTHON" audit_p134.py --input $R/reports/P134 --output $R/reports/R4/P134_audit) || rc=1
echo "== R4-00 hashes"; "$PYTHON" - <<'PY' || rc=1
import hashlib, json, os, glob
def h(p):
    s = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): s.update(b)
    return s.hexdigest()
out = {}
for pat in ("/projects/EEG-foundation-model/yinghao/models/vl3/*.pt", "/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals/*.pt", "/home/infres/yinwang/CS_QMI/outputs/VL1_21/exclusions.json",
            "/projects/EEG-foundation-model/yinghao/models/vl1_21_detector/final.pt"):
    for p in sorted(glob.glob(pat)):
        out[p] = {"sha256": h(p), "bytes": os.path.getsize(p)}
json.dump(out, open("/home/infres/yinwang/CS_QMI/ssl_pilot/reports/R4/intake_hashes.json", "w"), indent=1)
print(len(out), "files hashed")
PY
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
