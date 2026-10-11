#!/usr/bin/env bash
#SBATCH --job-name=r4_hashes
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-00 intake manifest: sha256 + size + mtime of the VL3 checkpoints, VL1-21 proposals / exclusions / detector (re-runnable; files modified during the
# window or after their own hash are flagged in_flight and must be re-hashed).  Writes reports/R4/intake_hashes.json with a snapshot time.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
"$PYTHON" - <<'PY'; rc=$?
import datetime, glob, hashlib, json, os
from concurrent.futures import ThreadPoolExecutor
def h(p):
    s = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): s.update(b)
    return s.hexdigest()
pats = ["/projects/EEG-foundation-model/yinghao/models/vl3/*.pt", "/home/infres/yinwang/CS_QMI/outputs/VL1_21/proposals/*.pt", "/home/infres/yinwang/CS_QMI/outputs/VL1_21/exclusions.json",
        "/projects/EEG-foundation-model/yinghao/models/vl1_21_detector/final.pt"]
start = datetime.datetime.now(datetime.timezone.utc); files = sorted(p for pat in pats for p in glob.glob(pat))
def one(p):
    st0 = os.stat(p); d = h(p); st1 = os.stat(p)
    return p, {"sha256": d, "bytes": st1.st_size, "mtime_utc": datetime.datetime.fromtimestamp(st1.st_mtime, datetime.timezone.utc).isoformat(),
               "in_flight": (st0.st_mtime != st1.st_mtime or st0.st_size != st1.st_size or st1.st_mtime > start.timestamp())}
with ThreadPoolExecutor(4) as ex: out = dict(ex.map(one, files))
end = datetime.datetime.now(datetime.timezone.utc); inflight = [p for p, v in out.items() if v["in_flight"]]
json.dump({"snapshot_start_utc": start.isoformat(), "snapshot_end_utc": end.isoformat(), "n_files": len(out), "in_flight": inflight, "files": out},
          open("/home/infres/yinwang/CS_QMI/ssl_pilot/reports/R4/intake_hashes.json", "w"), indent=1)
print(len(out), "files hashed;", len(inflight), "in flight (re-hash later):", inflight)
PY
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
