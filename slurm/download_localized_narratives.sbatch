#!/usr/bin/env bash
#SBATCH --job-name=download_ln
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/download_ln_%j.out
# External target for pre-check A (P57 addendum 1): Localized Narratives on the Open Images VALIDATION split.
# Downloads the validation narratives jsonl (Google, annotations CC BY 4.0), samples N_IMG image ids (seed 20260927, identity = image id,
# first narrative per image), downloads those images from the Open Images S3 bucket (images CC BY 2.0, per-image attribution in the
# validation CSV, also downloaded), and writes PROVENANCE.json (urls, licences as quoted, sha256 of jsonl / id list / csv, counts, bytes, utc).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
DATA_DIR="${DATA_DIR:-/home/infres/yinwang/CS_QMI/data/localized_narratives}"; N_IMG="${N_IMG:-5000}"; SEED="${SEED:-20260927}"
JSONL_URL="https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl"
CSV_URL="https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
IMG_BASE="${IMG_BASE:-https://open-images-dataset.s3.amazonaws.com/validation}"
banner "download Localized Narratives (Open Images validation) -> $DATA_DIR (N_IMG=$N_IMG)"
mkdir -p "$DATA_DIR/images"; cd "$DATA_DIR" || exit 2; df -h . | tail -1
[ -s open_images_validation_localized_narratives.jsonl ] || curl -L --fail --retry 5 --retry-delay 30 -C - -o open_images_validation_localized_narratives.jsonl "$JSONL_URL" || exit 3
[ -s validation-images-with-rotation.csv ] || curl -L --fail --retry 5 --retry-delay 30 -o validation-images-with-rotation.csv "$CSV_URL" || exit 4
"$PYTHON" - "$N_IMG" "$SEED" <<'PY'
import json, hashlib, sys, numpy as np
n_img, seed = int(sys.argv[1]), int(sys.argv[2])
first = {}; n_lines = 0
with open("open_images_validation_localized_narratives.jsonl") as f:
    for line in f:
        n_lines += 1; r = json.loads(line); i = r["image_id"]
        if i not in first: first[i] = {"caption": r["caption"], "annotator_id": r["annotator_id"]}
ids = sorted(first); rng = np.random.default_rng(seed); pick = sorted(rng.choice(len(ids), size=n_img, replace=False).tolist()); sel = [ids[k] for k in pick]
json.dump({"seed": seed, "n_narratives": n_lines, "n_unique_images": len(ids), "n_selected": len(sel), "rule": "first narrative per image; sample without replacement from the sorted unique image ids",
           "image_ids": sel, "captions": {i: first[i]["caption"] for i in sel}, "annotator_id": {i: first[i]["annotator_id"] for i in sel},
           "ids_sha256": hashlib.sha256(",".join(sel).encode()).hexdigest()}, open("selection.json", "w"))
open("ids.txt", "w").write("\n".join(sel) + "\n"); print("narratives", n_lines, "unique images", len(ids), "selected", len(sel))
PY
# images: 8 parallel curls, skip existing, retry
cat ids.txt | xargs -P 8 -I{} sh -c 'f=images/{}.jpg; [ -s "$f" ] || curl -sL --fail --retry 4 --retry-delay 10 -o "$f" "'"$IMG_BASE"'/{}.jpg" || echo "FAILED {}"' > download_failures.txt
NF=$(grep -c FAILED download_failures.txt || true); echo "image download failures: $NF"
"$PYTHON" - "$JSONL_URL" "$CSV_URL" "$IMG_BASE" <<'PY'
import csv, hashlib, json, os, sys, time
from pathlib import Path
jsonl_url, csv_url, img_base = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
sel = json.load(open("selection.json")); ids = sel["image_ids"]; have = [i for i in ids if (Path("images") / f"{i}.jpg").stat().st_size > 0 if (Path("images") / f"{i}.jpg").exists()]
rows = {}
with open("validation-images-with-rotation.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["ImageID"] in sel["captions"]: rows[r["ImageID"]] = r
with open("attribution_selected.csv", "w", newline="") as f:
    w = None
    for i in ids:
        if i in rows:
            if w is None: w = csv.DictWriter(f, fieldnames=list(rows[i].keys())); w.writeheader()
            w.writerow(rows[i])
lic = {}
for r in rows.values(): lic[r.get("License", "")] = lic.get(r.get("License", ""), 0) + 1
prov = {"dataset": "Localized Narratives (Google) on the Open Images validation split", "narratives_url": jsonl_url, "attribution_csv_url": csv_url, "image_url_pattern": img_base + "/<ImageID>.jpg",
        "licence_annotations": "Localized Narratives project page (https://google.github.io/localized-narratives/), verbatim: 'All the annotations available through this website are released under a CC BY 4.0 license.'",
        "licence_images": "Open Images (https://storage.googleapis.com/openimages/web/factsfigures_v7.html), verbatim: 'The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license.' and 'Note: while we tried to identify images that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each image and you should verify the license for each image yourself.' Per-image attribution (author, title, original URL, licence) kept in attribution_selected.csv.",
        "sha256": {"narratives_jsonl": sha("open_images_validation_localized_narratives.jsonl"), "attribution_csv": sha("validation-images-with-rotation.csv"), "selected_ids": sel["ids_sha256"]},
        "bytes": {"narratives_jsonl": os.path.getsize("open_images_validation_localized_narratives.jsonl"), "images": sum((Path("images") / f"{i}.jpg").stat().st_size for i in have)},
        "counts": {"narratives": sel["n_narratives"], "unique_images": sel["n_unique_images"], "selected": len(ids), "images_downloaded": len(have), "attribution_rows_found": len(rows), "licence_values_in_csv": lic},
        "seed": sel["seed"], "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(prov, open("PROVENANCE.json", "w"), indent=2); print(json.dumps(prov, indent=1))
PY
echo "exit=$? end=$(date -u +%FT%TZ)"
