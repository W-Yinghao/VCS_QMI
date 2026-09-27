#!/usr/bin/env bash
#SBATCH --job-name=download_aomic
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/download_aomic_%j.out
# B-T2 third real source (P60 fMRI prereg): AOMIC-PIOP1 (OpenNeuro ds002785, CC0) fMRIPrep derivatives for N_SUB subjects chosen by SEED among
# the subjects that have T1w preproc + brain mask + resting-state space-T1w boldref (availability list from HEAD checks, AVAIL).  Writes
# PROVENANCE.json (urls, licence quotes from both dataset_description.json, per-file sha256 / bytes, id-list sha256, utc) and pairs_index.json.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
DATA_DIR="${DATA_DIR:-/home/infres/yinwang/CS_QMI/data/aomic_piop1}"; N_SUB="${N_SUB:-60}"; SEED="${SEED:-20260927}"
BASE="https://s3.amazonaws.com/openneuro.org/ds002785"; AVAIL="${AVAIL:-$DATA_DIR/availability_head_20260927.txt}"
banner "download AOMIC-PIOP1 (ds002785) fMRIPrep T1w + mask + resting-state boldref -> $DATA_DIR (N_SUB=$N_SUB seed=$SEED)"
mkdir -p "$DATA_DIR"; cd "$DATA_DIR" || exit 2; df -h . | tail -1
curl -sL --fail --retry 5 --retry-delay 20 -o dataset_description.json "$BASE/dataset_description.json" || exit 3
curl -sL --fail --retry 5 --retry-delay 20 -o fmriprep_dataset_description.json "$BASE/derivatives/fmriprep/dataset_description.json" || exit 3
"$PYTHON" - "$AVAIL" "$N_SUB" "$SEED" <<'PY'
import hashlib, json, sys, numpy as np
avail, n_sub, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ok = sorted(l.split()[0] for l in open(avail) if l.split()[1:4] == ["200", "200", "200"])
chosen = sorted(np.random.default_rng(seed).permutation(ok)[:n_sub].tolist())
json.dump({"seed": seed, "n_available": len(ok), "available": ok, "selected": chosen, "ids_sha256": hashlib.sha256(",".join(chosen).encode()).hexdigest(),
           "rule": "numpy.random.default_rng(seed).permutation(sorted(available))[:n_sub], sorted; available = subjects whose T1w preproc, brain mask and resting-state acq-mb3 space-T1w boldref all answered HTTP 200"}, open("selection.json", "w"), indent=1)
open("ids.txt", "w").write("\n".join(chosen) + "\n"); print("available", len(ok), "selected", len(chosen))
PY
mkdir -p files
cat ids.txt | xargs -P 8 -I{} sh -c 'S={}; for spec in "anat/${S}_desc-preproc_T1w.nii.gz" "anat/${S}_desc-brain_mask.nii.gz" "func/${S}_task-restingstate_acq-mb3_space-T1w_boldref.nii.gz"; do f=files/$(basename $spec); [ -s "$f" ] || curl -sL --fail --retry 5 --retry-delay 15 -o "$f" "'"$BASE"'/derivatives/fmriprep/${S}/${spec}" || echo "FAILED $S $spec"; done' > download_failures.txt
NF=$(grep -c FAILED download_failures.txt || true); echo "download failures: $NF"; [ "$NF" = "0" ] || { cat download_failures.txt | head; exit 4; }
"$PYTHON" - "$BASE" <<'PY'
import hashlib, json, time
from pathlib import Path
base = __import__("sys").argv[1]; root = Path("."); sel = json.load(open("selection.json"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
dd = json.load(open("dataset_description.json")); fd = json.load(open("fmriprep_dataset_description.json"))
files = {}; pairs = {}; nbytes = 0
for s in sel["selected"]:
    rec = {}
    for key, spec in (("T1w", f"anat/{s}_desc-preproc_T1w.nii.gz"), ("mask", f"anat/{s}_desc-brain_mask.nii.gz"), ("boldref", f"func/{s}_task-restingstate_acq-mb3_space-T1w_boldref.nii.gz")):
        p = root / "files" / Path(spec).name; files[str(p)] = {"url": f"{base}/derivatives/fmriprep/{s}/{spec}", "bytes": p.stat().st_size, "sha256": sha(p)}; nbytes += p.stat().st_size; rec[key] = str(p)
    pairs[s] = rec
prov = {"dataset": "AOMIC-PIOP1 (Amsterdam Open MRI Collection), OpenNeuro ds002785", "doi": dd.get("DatasetDOI"), "page": "https://openneuro.org/datasets/ds002785",
        "bucket": base, "licence_dataset": dd.get("License"), "licence_derivatives": fd.get("License"), "derivatives_pipeline": fd.get("PipelineDescription"),
        "how_to_acknowledge": {"dataset": dd.get("HowToAcknowledge"), "fmriprep": fd.get("HowToAcknowledge")},
        "dataset_description_sha256": {"dataset": sha("dataset_description.json"), "fmriprep": sha("fmriprep_dataset_description.json")},
        "selection": {k: sel[k] for k in ("seed", "n_available", "rule", "ids_sha256")}, "n_selected": len(sel["selected"]), "subjects": sel["selected"],
        "n_files": len(files), "total_bytes": nbytes, "files": files, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(prov, open("PROVENANCE.json", "w"), indent=1); json.dump({"pairs": pairs}, open("pairs_index.json", "w"), indent=1)
print("files", len(files), "bytes", nbytes, "ids_sha256", sel["ids_sha256"], "licences", dd.get("License"), fd.get("License"))
PY
echo "exit=$? end=$(date -u +%FT%TZ)"
