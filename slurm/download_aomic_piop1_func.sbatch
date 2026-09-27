#!/usr/bin/env bash
#SBATCH --job-name=dl_aomic_func
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/download_aomic_func_%j.out
# AOMIC-PIOP1 (ds002785, CC0) resting-state functional derivatives for the 60 subjects in IDS: T1w-space preproc BOLD, its brain mask, the
# confounds TSV and the raw bold JSON (TR). Writes func/PROVENANCE_func.json (licence quotes, urls, per-file sha256/bytes, id-list sha256).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
ROOT="${ROOT:-/home/infres/yinwang/CS_QMI/data/aomic_piop1}"; IDS="${IDS:-$ROOT/ids.txt}"; B="https://s3.amazonaws.com/openneuro.org/ds002785"
banner "download AOMIC-PIOP1 func derivatives -> $ROOT/func ($(wc -l < $IDS) subjects)"
mkdir -p "$ROOT/func"; cd "$ROOT/func" || exit 2; df -h . | tail -1
: > urls.txt
while read s; do [ -z "$s" ] && continue; mkdir -p "$s"
  for f in "$s/func/${s}_task-restingstate_acq-mb3_space-T1w_desc-preproc_bold.nii.gz" "$s/func/${s}_task-restingstate_acq-mb3_space-T1w_desc-brain_mask.nii.gz" "$s/func/${s}_task-restingstate_acq-mb3_desc-confounds_regressors.tsv"; do echo "$B/derivatives/fmriprep/$f $s/$(basename $f)" >> urls.txt; done
  echo "$B/$s/func/${s}_task-restingstate_acq-mb3_bold.json $s/${s}_task-restingstate_acq-mb3_bold.json" >> urls.txt
done < "$IDS"
cat urls.txt | xargs -P 6 -L 1 bash -c 'u=$0; o=$1; [ -s "$o" ] || curl -sS -L --fail --retry 5 --retry-delay 20 -C - -o "$o" "$u" || echo "FAILED $u" >> failures.txt'
echo "failures: $(cat failures.txt 2>/dev/null | wc -l)"
"$PYTHON" - "$ROOT" "$IDS" "$B" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
root, ids_f, bucket = Path(sys.argv[1]), sys.argv[2], sys.argv[3]; ids = [l.strip() for l in open(ids_f) if l.strip()]
anat = json.load(open(root / "PROVENANCE.json"))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
files = {}; missing = []; total = 0
for s in ids:
    for name in (f"{s}_task-restingstate_acq-mb3_space-T1w_desc-preproc_bold.nii.gz", f"{s}_task-restingstate_acq-mb3_space-T1w_desc-brain_mask.nii.gz", f"{s}_task-restingstate_acq-mb3_desc-confounds_regressors.tsv", f"{s}_task-restingstate_acq-mb3_bold.json"):
        p = root / "func" / s / name
        if p.exists() and p.stat().st_size > 0: files[f"{s}/{name}"] = {"bytes": p.stat().st_size, "sha256": sha(p)}; total += p.stat().st_size
        else: missing.append(f"{s}/{name}")
prov = {"dataset": anat["dataset"], "doi": anat["doi"], "bucket": bucket, "licence_dataset": anat["licence_dataset"], "licence_derivatives": anat["licence_derivatives"],
        "how_to_acknowledge": anat["how_to_acknowledge"], "dataset_description_sha256": anat["dataset_description_sha256"], "space": "T1w (native EPI grid, fMRIPrep 1.4.1)",
        "patterns": ["derivatives/fmriprep/<s>/func/<s>_task-restingstate_acq-mb3_space-T1w_desc-preproc_bold.nii.gz", "…_space-T1w_desc-brain_mask.nii.gz", "…_desc-confounds_regressors.tsv", "<s>/func/<s>_task-restingstate_acq-mb3_bold.json"],
        "subjects": ids, "ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(), "n_subjects": len(ids), "n_files": len(files), "missing": missing, "total_bytes": total, "files": files, "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(prov, open(root / "func" / "PROVENANCE_func.json", "w"), indent=1); print("files", len(files), "missing", len(missing), "bytes", total, "ids_sha256", prov["ids_sha256"])
PY
echo "exit=$? end=$(date -u +%FT%TZ)"
