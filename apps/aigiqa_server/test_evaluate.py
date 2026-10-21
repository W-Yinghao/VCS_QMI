"""Unseal the AGIQA-3K TEST labels once, run the package's evaluate (19 predictions) and compare (VCS-exact vs BCE, each
seed, prompt bootstrap), then reseal.

Owner 2026-10-21 「开始 (Recommended)」: 18 selected models and native ImageReward are predicted on TEST inputs first, then the
MOS labels are unsealed and evaluated once; all results are pilot/diagnostic and nothing is selected after TEST.
Everything is checked before the unseal; only sealed/test_labels.csv is opened (all_labels.csv stays mode 000). A second
run refuses (TEST_UNSEALED.json exists), so the labels are read in exactly one job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", required=True); ap.add_argument("--prepared", required=True)
    ap.add_argument("--pkg", required=True); ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    T, prepared = Path(a.test_dir), Path(a.prepared)
    marker = T / "TEST_UNSEALED.json"
    if marker.exists() or (T / "metrics").exists() or (T / "compare").exists():
        sys.exit("TEST was already evaluated or attempted; a second label read needs an explicit owner decision")
    done = json.loads((T / "PREDICTIONS_DONE.json").read_text())
    if done.get("status") != "PASS" or done.get("n_predictions") != 19 or len(done["compare_pairs"]) != 3:
        sys.exit("PREDICTIONS_DONE is not a complete PASS record")
    bad = [k for k, r in done["predictions"].items()
           if sha(r["csv"]) != r["csv_sha256"] or sha(r["csv"] + ".meta.json") != r["meta_sha256"]]
    if bad or sha(done["release"]) != done["release_file_sha256"]:
        sys.exit(f"predictions or release changed since the GPU job: {bad}")
    sealed, labels = prepared / "sealed", prepared / "sealed" / "test_labels.csv"
    expected = json.loads((prepared / "manifest.json").read_text())["input_and_label_hashes"]["sealed/test_labels.csv"]
    record = {"schema": "aigiqa_server_test_unseal_v1", "owner_approval": "2026-10-21 「开始 (Recommended)」",
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "unsealed_utc": utc(),
              "label_file": str(labels), "other_sealed_files_opened": False, "predictions_done_sha256": sha(T / "PREDICTIONS_DONE.json"),
              "claim_status": "PILOT_DIAGNOSTIC_ONLY", "selection_after_test": "none", "status": "STARTED"}
    marker.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    os.chmod(sealed, 0o500)
    os.chmod(labels, 0o400)
    try:
        record["labels_sha256"] = sha(labels)
        if record["labels_sha256"] != expected:
            raise RuntimeError("sealed TEST label file differs from prepare's manifest hash")
        (T / "metrics").mkdir(); (T / "compare").mkdir()
        def run(args, log):
            with open(log, "w") as stream:
                subprocess.run([sys.executable, "-m", "aigiqa.cli", *args], cwd=a.pkg, check=True, stdout=stream, stderr=subprocess.STDOUT)
        for key, r in sorted(done["predictions"].items()):
            run(["evaluate", "--predictions", r["csv"], "--labels", str(labels), "--target", "alignment",
                 "--output", str(T / "metrics" / f"{key}.json")], T / "metrics" / f"{key}.log")
        for ka, kb, seed in done["compare_pairs"]:
            out = T / "compare" / f"vcs_exact_text_minus_bce_text_seed{seed}.json"
            run(["compare", "--a", done["predictions"][ka]["csv"], "--b", done["predictions"][kb]["csv"], "--labels", str(labels),
                 "--target", "alignment", "--output", str(out)], out.with_suffix(".log"))
        record["status"] = "EVALUATED"
    except Exception as exc:
        record["status"] = "FAILED"; record["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        os.chmod(labels, 0)
        os.chmod(sealed, 0)
        record["resealed_utc"] = utc()
        marker.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    if record["status"] != "EVALUATED":
        sys.exit(record["error"])
    subprocess.run([sys.executable, str(Path(__file__).with_name("test_summary.py")), "--test-dir", str(T), "--write"], check=True)
    ev = Path(a.evidence); ev.mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "compare"):
        shutil.copytree(T / sub, ev / sub, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.log"))
    for name in ("TEST_SUMMARY.json", "TEST_UNSEALED.json", "PREDICTIONS_DONE.json"):
        shutil.copy(T / name, ev / name)


if __name__ == "__main__":
    main()
