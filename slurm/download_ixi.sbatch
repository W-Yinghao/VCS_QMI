#!/usr/bin/env bash
#SBATCH --job-name=download_ixi
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/download_ixi_%j.out
# Downloads the IXI T2 and PD archives (brain-development.org/ixi-dataset/, CC BY-SA 3.0) into DATA_DIR, records provenance (url, licence,
# sha256, bytes, utc) in PROVENANCE.json, extracts the NIfTI files, and lists the subjects that have both PD and T2.
# URL_BASE may be overridden (the official directory returned HTTP 403 to the login node on 2026-09-27).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
URL_BASE="${URL_BASE:-https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI}"; DATA_DIR="${DATA_DIR:-/home/infres/yinwang/CS_QMI/data/ixi}"
PAGE="https://brain-development.org/ixi-dataset/"; LICENCE="CC BY-SA 3.0 (page: 'This data is made available under the Creative Commons CC BY-SA 3.0 license. If you use the IXI data please acknowledge the source of the IXI data, e.g. this website.'; https://creativecommons.org/licenses/by-sa/3.0/legalcode)"
banner "download IXI T2 + PD -> $DATA_DIR from $URL_BASE"
mkdir -p "$DATA_DIR"; cd "$DATA_DIR" || exit 2; df -h "$DATA_DIR" | tail -1
for M in T2 PD; do
  if [ ! -s "IXI-$M.tar" ]; then
    curl -4 -L --fail --retry 5 --retry-delay 30 -C - -A "Mozilla/5.0 (X11; Linux x86_64) Firefox/128.0" -e "$PAGE" -o "IXI-$M.tar" "$URL_BASE/IXI-$M.tar"; rc=$?
    if [ $rc -ne 0 ]; then echo "DOWNLOAD FAILED for IXI-$M.tar (curl rc=$rc)"; curl -4 -sS -I -A "Mozilla/5.0" "$URL_BASE/IXI-$M.tar" | head -1; exit 3; fi
  fi
  echo "IXI-$M.tar bytes=$(stat -c %s IXI-$M.tar) sha256=$(sha256sum IXI-$M.tar | cut -d' ' -f1)"
  mkdir -p "$M"; [ -n "$(ls $M 2>/dev/null)" ] || tar -xf "IXI-$M.tar" -C "$M" || exit 4
done
"$PYTHON" - "$URL_BASE" "$PAGE" "$LICENCE" "$DATA_DIR" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
url_base, page, licence, root = sys.argv[1:5]; root = Path(root); prov = {"dataset": "IXI (Information eXtraction from Images), Imperial College London / brain-development.org", "page_url": page, "licence": licence,
    "archives": {}, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "subjects": {}}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
ids = {}
for M in ("T2", "PD"):
    tar = root / f"IXI-{M}.tar"; prov["archives"][M] = {"url": f"{url_base}/IXI-{M}.tar", "bytes": tar.stat().st_size, "sha256": sha(tar)}
    files = sorted((root / M).rglob("*.nii*")); ids[M] = {f.name.split("-")[0]: str(f.relative_to(root)) for f in files}; prov["subjects"][f"n_{M}"] = len(files)
both = sorted(set(ids["T2"]) & set(ids["PD"])); prov["subjects"]["n_both"] = len(both); prov["subjects"]["both_ids_sha256"] = hashlib.sha256(",".join(both).encode()).hexdigest()
json.dump({"both": {s: {"T2": ids["T2"][s], "PD": ids["PD"][s]} for s in both}}, open(root / "pairs_index.json", "w"), indent=1)
json.dump(prov, open(root / "PROVENANCE.json", "w"), indent=2); print(json.dumps(prov, indent=1))
PY
echo "exit=$? end=$(date -u +%FT%TZ)"
