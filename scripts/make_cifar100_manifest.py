"""P91 — the frozen identity split of the CIFAR-100 *training* partition (dev 45k / selection 5k = 100 classes x 50), same algorithm and split
seed as CIFAR-10 (`vcs_ssl.data.splits.build_manifest`, numpy default_rng(20260924), per-class permutation, sorted UIDs).

    python scripts/make_cifar100_manifest.py --root /home/infres/yinwang/CS_QMI/data/cifar100 \
        --out /home/infres/yinwang/CS_QMI/manifests/cifar100_dev45k_val5k.json --report reports/<gate>/cifar100_manifest.json

Verifies the local copy first (md5 of train/meta against the published values, index-order sentinel, 100 x 500 class counts; the test file is
hashed for provenance only and never read).  If the manifest exists it is verified against the data on disk (raw file hashes + stored hash);
otherwise it is created.  Never downloads anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vcs_ssl.data.cifar import CIFAR100_FIRST10_LABELS, load_train_partition  # noqa: E402
from vcs_ssl.data.splits import build_manifest, load_manifest, write_manifest  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--report", default=None)
    ap.add_argument("--split-seed", type=int, default=20260924); ap.add_argument("--val-per-class", type=int, default=50)
    a = ap.parse_args()
    data = load_train_partition("cifar100", a.root)
    rep = {"utc": utc_now(), "root": a.root, "n_train": len(data), "n_classes": int(data.targets.max()) + 1, "file_hashes": data.file_hashes, "source": data.source,
           "first10_labels": data.targets[:10].tolist(), "first10_expected": CIFAR100_FIRST10_LABELS, "official_test_read": False,
           "split_seed": a.split_seed, "val_per_class": a.val_per_class}
    out = Path(a.out)
    if out.is_file():
        m = load_manifest(out, expected_seed=a.split_seed, expected_val_per_class=a.val_per_class)
        if m["raw_file_hashes"] != data.file_hashes:
            raise SystemExit("existing manifest raw_file_hashes differ from the data on disk")
        rep["manifest"] = {"path": str(out), "status": "EXISTS_VERIFIED"}
    else:
        m = build_manifest(data.targets, split_seed=a.split_seed, val_per_class=a.val_per_class, source=data.source, file_hashes=data.file_hashes)
        write_manifest(m, out)
        m = load_manifest(out, expected_seed=a.split_seed, expected_val_per_class=a.val_per_class)
        rep["manifest"] = {"path": str(out), "status": "CREATED"}
    rep["manifest"].update({"sha256": m["manifest_sha256"], "n_fit": m["n_fit"], "n_selection": m["n_selection"], "n_classes": m["n_classes"],
                            "fit_class_counts_unique": sorted(set(m["fit_class_counts"])), "selection_class_counts_unique": sorted(set(m["selection_class_counts"])),
                            "fit_first5": m["fit_uids"][:5], "selection_first5": m["selection_uids"][:5], "split_algorithm": m["split_algorithm"]})
    if a.report:
        atomic_write_json(a.report, rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("source",)}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
