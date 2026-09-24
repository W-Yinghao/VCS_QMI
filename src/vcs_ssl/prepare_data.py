"""P0 data step: verify (and optionally extract) the official CIFAR-10 python archive; never called by the trainer.

    python -m vcs_ssl.prepare_data --root "$DATA_ROOT" [--extract]

The archive itself was fetched once, explicitly, from the official page (W4) with the md5 published there.  This module
records the verification so that the trainer can keep ``download=false`` and still assert provenance.
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

from .data.cifar import CIFAR10_ARCHIVE_MD5, CIFAR10_DIR, verify_cifar10_train_files
from .utils import atomic_write_json, md5_file, sha256_file, utc_now


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    archive = root / "cifar-10-python.tar.gz"
    rec = {"root": str(root), "utc": utc_now(), "archive_present": archive.is_file()}
    if archive.is_file():
        rec["archive_md5"] = md5_file(archive)
        rec["archive_sha256"] = sha256_file(archive)
        rec["archive_md5_matches_official"] = rec["archive_md5"] == CIFAR10_ARCHIVE_MD5
        if not rec["archive_md5_matches_official"]:
            print(f"ERROR: archive md5 {rec['archive_md5']} != official {CIFAR10_ARCHIVE_MD5}", file=sys.stderr)
            if args.report:
                atomic_write_json(args.report, rec)
            return 2
        if args.extract and not (root / CIFAR10_DIR).is_dir():
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(root, filter="data")
            rec["extracted"] = True
    try:
        rec["train_files"] = verify_cifar10_train_files(root)
        rec["train_files_verified"] = True
    except Exception as e:  # noqa: BLE001
        rec["train_files_verified"] = False
        rec["error"] = repr(e)
    if args.report:
        atomic_write_json(args.report, rec)
    print(f"archive_ok={rec.get('archive_md5_matches_official')} train_files_verified={rec['train_files_verified']}")
    return 0 if rec["train_files_verified"] else 2


if __name__ == "__main__":
    sys.exit(main())
