#!/usr/bin/env bash
#SBATCH --job-name=download_rgbnir
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/download_rgbnir_%j.out
# Downloads the EPFL IVRL RGB-NIR Scene dataset archive (nirscene1.zip, ~952 MB, last-modified 2012-09-18) into DATA_DIR, records sha256 /
# bytes / url / utc / the page's terms text in PROVENANCE.json, extracts, and counts pairs per category.  Submit ONLY after the provenance
# section of reports/P60_TASK_B_RGBNIR_PREREG_*.md is written and the usage terms are settled (the dataset page states no licence text).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
URL="${URL:-https://ivrlwww.epfl.ch/supplementary_material/cvpr11/nirscene1.zip}"; DATA_DIR="${DATA_DIR:-/home/infres/yinwang/CS_QMI/data/rgb_nir}"
PAGE="https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/"
banner "download RGB-NIR scene dataset -> $DATA_DIR"
mkdir -p "$DATA_DIR"; cd "$DATA_DIR" || exit 2
[ -f nirscene1.zip ] || curl -L --fail --retry 3 -A "Mozilla/5.0" -o nirscene1.zip "$URL" || exit $?
SHA=$(sha256sum nirscene1.zip | cut -d' ' -f1); BYTES=$(stat -c %s nirscene1.zip); echo "sha256=$SHA bytes=$BYTES"
[ -d nirscene1 ] || unzip -q nirscene1.zip || exit $?
"$PYTHON" - "$URL" "$PAGE" "$SHA" "$BYTES" "$DATA_DIR" <<'PY'
import json, sys, time
from pathlib import Path
url, page, sha, nbytes, root = sys.argv[1:6]; root = Path(root)
pairs = {}
for rgb in root.rglob("*_rgb.tif*"):
    nir = rgb.with_name(rgb.name.replace("_rgb", "_nir"))
    if nir.exists():
        pairs[rgb.parent.name] = pairs.get(rgb.parent.name, 0) + 1
json.dump({"dataset": "EPFL IVRL RGB-NIR Scene dataset (Brown & Suesstrunk, CVPR 2011)", "archive_url": url, "page_url": page, "sha256": sha, "bytes": int(nbytes),
           "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "terms_as_found": "No licence or terms-of-use text on the dataset page (checked 2026-09-27); page offers the archive for download and says 'Please contact us if you require higher resolution or RAW images.'",
           "pairs_per_category": pairs, "n_pairs": sum(pairs.values())}, open(root / "PROVENANCE.json", "w"), indent=2)
print(json.dumps(pairs), "total", sum(pairs.values()))
PY
echo "exit=$? end=$(date -u +%FT%TZ)"
