"""Raw CIFAR-10 *training* partition loading with file verification.

Only the official 50,000-image training partition is ever instantiated in the first round.  The 10,000-image official
test partition is never read (spec §4.1, §15.2 item 9): :func:`load_cifar10_train` refuses ``train=False`` by
construction, and the trainer never imports another loader.

Final round (P67, owner's rule "official test set once, at the end"): the test partition opens only through the explicit
double unlock ``load_cifar10_train(root, train=False, allow_official_test=True)`` **and** the environment variable
``VCS_FINAL_ROUND=1``; either one missing keeps the PermissionError.  The returned object carries the test batch's sha256.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

import numpy as np

from ..utils import md5_file, sha256_file

CIFAR10_DIR = "cifar-10-batches-py"
# Official archive / batch md5 values (https://www.cs.toronto.edu/~kriz/cifar.html; also used by torchvision).
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_TRAIN_BATCHES = [
    ("data_batch_1", "c99cafc152244af753f735de768cd75f"),
    ("data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
    ("data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
    ("data_batch_4", "634d18415352ddfa80567beed471001a"),
    ("data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"),
]
CIFAR10_META = ("batches.meta", "5ff9c542aee3614f3951f8cda6e48888")
# Official test batch (torchvision CIFAR10.test_list); read only by the final-round unlock below.
CIFAR10_TEST_BATCH = ("test_batch", "40351d587109b95175f43aff81a1287e")
FINAL_ROUND_ENV = "VCS_FINAL_ROUND"
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# First ten labels of data_batch_1 in official order; used as an index-order sentinel in preflight.
CIFAR10_FIRST10_LABELS = [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]


@dataclass
class CifarTrain:
    """Uint8 images ``[50000, 32, 32, 3]`` (HWC, RGB) in official index order plus integer labels."""

    data: np.ndarray
    targets: np.ndarray  # int64 [50000]; labels may only be read by split construction and evaluation
    root: str
    file_hashes: dict[str, dict[str, str]] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.data.shape[0])


def cifar10_files(root: str | Path) -> dict[str, Path]:
    base = Path(root) / CIFAR10_DIR
    files = {name: base / name for name, _ in CIFAR10_TRAIN_BATCHES}
    files[CIFAR10_META[0]] = base / CIFAR10_META[0]
    return files


def verify_cifar10_train_files(root: str | Path) -> dict[str, dict[str, str]]:
    """Return per-file md5/sha256 and raise if any official training batch is missing or corrupt."""
    files = cifar10_files(root)
    out: dict[str, dict[str, str]] = {}
    for name, expected_md5 in [*CIFAR10_TRAIN_BATCHES, CIFAR10_META]:
        p = files[name]
        if not p.is_file():
            raise FileNotFoundError(f"CIFAR-10 file missing: {p}. Data is never downloaded implicitly; run prepare_data.")
        md5 = md5_file(p)
        if md5 != expected_md5:
            raise ValueError(f"md5 mismatch for {p}: {md5} != {expected_md5}")
        out[name] = {"md5": md5, "sha256": sha256_file(p), "bytes": str(p.stat().st_size)}
    archive = Path(root) / "cifar-10-python.tar.gz"
    if archive.is_file():
        out["cifar-10-python.tar.gz"] = {"md5": md5_file(archive), "sha256": sha256_file(archive), "bytes": str(archive.stat().st_size)}
    return out


def official_test_unlocked(allow_official_test: bool) -> bool:
    """True only when the caller opted in AND the process runs in the declared final round (env VCS_FINAL_ROUND=1)."""
    return bool(allow_official_test) and os.environ.get(FINAL_ROUND_ENV, "") == "1"


def _load_cifar10_official_test(root: str | Path) -> CifarTrain:
    """Final round only (P67): the 10,000-image official test partition, md5-verified, with its sha256 in ``file_hashes``."""
    base = Path(root) / CIFAR10_DIR
    p = base / CIFAR10_TEST_BATCH[0]
    if not p.is_file():
        raise FileNotFoundError(f"CIFAR-10 test batch missing: {p}")
    md5 = md5_file(p)
    if md5 != CIFAR10_TEST_BATCH[1]:
        raise ValueError(f"md5 mismatch for {p}: {md5} != {CIFAR10_TEST_BATCH[1]}")
    hashes = verify_cifar10_train_files(root)  # the training batches must be the same files the manifest was built from
    hashes[CIFAR10_TEST_BATCH[0]] = {"md5": md5, "sha256": sha256_file(p), "bytes": str(p.stat().st_size)}
    import torchvision  # noqa: PLC0415
    from torchvision.datasets import CIFAR10  # noqa: PLC0415

    ds = CIFAR10(str(root), train=False, download=False)
    data = np.ascontiguousarray(ds.data)
    targets = np.asarray(ds.targets, dtype=np.int64)
    if data.shape != (10000, 32, 32, 3) or data.dtype != np.uint8 or targets.shape != (10000,):
        raise ValueError(f"unexpected CIFAR-10 test tensor {data.shape} {data.dtype} / {targets.shape}")
    counts = np.bincount(targets, minlength=10)
    if not (counts == 1000).all():
        raise ValueError(f"CIFAR-10 test class counts unexpected: {counts.tolist()}")
    source = {
        "dataset": "CIFAR-10 python version, OFFICIAL TEST partition (final round P67 only)",
        "official_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "loader": "torchvision.datasets.CIFAR10(train=False, download=False)",
        "torchvision_version": torchvision.__version__,
        "index_order": "test_batch file order (uid = position 0..9999 in the test partition)",
        "num_images": int(len(targets)),
        "classes": CIFAR10_CLASSES,
        "test_batch_sha256": hashes[CIFAR10_TEST_BATCH[0]]["sha256"],
    }
    return CifarTrain(data=data, targets=targets, root=str(root), file_hashes=hashes, source=source)


def load_cifar10_train(root: str | Path, *, train: bool = True, allow_official_test: bool = False) -> CifarTrain:
    """Load the official training partition via torchvision (which concatenates data_batch_1..5 in order).

    ``train=False`` is refused unless the final-round double unlock holds (``allow_official_test=True`` and
    ``VCS_FINAL_ROUND=1`` in the environment); then the official test partition is returned instead.
    """
    if train is not True:
        if not official_test_unlocked(allow_official_test):
            raise PermissionError("official CIFAR-10 test partition is not accessible in the first round (spec §4.1); "
                                  f"the final round needs allow_official_test=True and {FINAL_ROUND_ENV}=1")
        return _load_cifar10_official_test(root)
    hashes = verify_cifar10_train_files(root)
    import torchvision  # noqa: PLC0415  (import here so preflight can report a missing torchvision cleanly)
    from torchvision.datasets import CIFAR10  # noqa: PLC0415

    ds = CIFAR10(str(root), train=True, download=False)
    data = np.ascontiguousarray(ds.data)  # [50000,32,32,3] uint8
    targets = np.asarray(ds.targets, dtype=np.int64)
    if data.shape != (50000, 32, 32, 3) or data.dtype != np.uint8:
        raise ValueError(f"unexpected CIFAR-10 train tensor {data.shape} {data.dtype}")
    if targets.shape != (50000,):
        raise ValueError("unexpected CIFAR-10 target shape")
    if targets[:10].tolist() != CIFAR10_FIRST10_LABELS:
        raise ValueError(f"CIFAR-10 index order sentinel failed: {targets[:10].tolist()} != {CIFAR10_FIRST10_LABELS}")
    counts = np.bincount(targets, minlength=10)
    if not (counts == 5000).all():
        raise ValueError(f"CIFAR-10 class counts unexpected: {counts.tolist()}")
    source = {
        "dataset": "CIFAR-10 python version, official training partition only",
        "official_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "loader": "torchvision.datasets.CIFAR10(train=True, download=False)",
        "torchvision_version": torchvision.__version__,
        "index_order": "concatenation of data_batch_1..5 in file order (official index = position)",
        "num_images": int(len(targets)),
        "classes": CIFAR10_CLASSES,
    }
    return CifarTrain(data=data, targets=targets, root=str(root), file_hashes=hashes, source=source)
