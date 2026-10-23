"""CIFAR-100 FIT/CAL/TEST roles with explicit acquisition and isolated RNG.

No torchvision dependency, no TEST label reads during prepare/train, no downloads
except `prepare --download`. Native pickle is read only after official MD5 check.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
ARCHIVE_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"
OFFICIAL_MD5 = {"train": "16019d7e3df5f24257cddd939b257f8d", "test": "f0ef6b0ae62326f3e7ffdfab6717acfc"}
MEAN = torch.tensor((0.5071, 0.4867, 0.4408)).view(3, 1, 1)
STD = torch.tensor((0.2675, 0.2565, 0.2761)).view(3, 1, 1)


def digest_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def download_cifar(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-100-python.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(CIFAR_URL, archive)
    if digest_file(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Official CIFAR-100 archive MD5 mismatch; remove the bad file manually and retry.")
    # Explicit allowlist avoids tar path traversal, links, arbitrary files.
    with tarfile.open(archive, "r:gz") as tf:
        for name in ("train", "test", "meta"):
            member = tf.getmember("cifar-100-python/" + name)
            if not member.isfile():
                raise ValueError("CIFAR archive contains a nonregular expected member")
            target = root / "cifar-100-python" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(member)
            target.write_bytes(stream.read())


def raw_cifar(root, partition):
    if partition not in OFFICIAL_MD5:
        raise ValueError(partition)
    path = Path(root) / "cifar-100-python" / partition
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run explicit `python -m kd prepare --download ...` first.")
    if digest_file(path, "md5") != OFFICIAL_MD5[partition]:
        raise ValueError(f"Official CIFAR-100 {partition} MD5 mismatch")
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    images = torch.from_numpy(data["data"].reshape(-1, 3, 32, 32).copy())
    labels = torch.tensor(data["fine_labels"], dtype=torch.long)
    return images, labels


def prepare(root, output, seed=20261019, cal_per_class=50, download=False, synthetic=False):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Will not overwrite an existing role manifest: {output}")
    if synthetic:
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / "synthetic.pt"
        if data_path.exists():
            raise FileExistsError(data_path)
        g = torch.Generator().manual_seed(seed)
        labels = torch.arange(4).repeat_interleave(24)
        images = torch.randint(0, 48, (len(labels), 3, 32, 32), generator=g, dtype=torch.uint8)
        for i, y in enumerate(labels):
            images[i, int(y) % 3, :16, :16] += 128
        test_y = torch.arange(4).repeat_interleave(8)
        test_x = torch.randint(0, 48, (len(test_y), 3, 32, 32), generator=g, dtype=torch.uint8)
        for i, y in enumerate(test_y):
            test_x[i, int(y) % 3, :16, :16] += 128
        torch.save({"train_x": images, "train_y": labels, "test_x": test_x, "test_y": test_y}, data_path)
        dataset_hash = digest_file(data_path)
        cal_per_class = 4
        n_test = 32
    else:
        if download:
            download_cifar(root)
        _, labels = raw_cifar(root, "train")  # TEST file is not parsed here.
        dataset_hash = digest_file(root / "cifar-100-python" / "train")
        n_test = 10000
    g = torch.Generator().manual_seed(seed)
    fit, cal = [], []
    classes = torch.unique(labels).tolist()
    for y in classes:
        ids = torch.where(labels == y)[0]
        if not 0 < cal_per_class < len(ids):
            raise ValueError("CAL must leave at least one FIT example per class")
        ids = ids[torch.randperm(len(ids), generator=g)].tolist()
        cal.extend(ids[:cal_per_class]); fit.extend(ids[cal_per_class:])
    body = {"schema": "kd_roles_v1", "dataset": "synthetic_smoke" if synthetic else "CIFAR100",
            "seed": seed, "num_classes": len(classes), "fit_ids": sorted(fit), "cal_ids": sorted(cal),
            "test_ids": list(range(n_test)), "test_partition": "official_test", "train_data_sha256": dataset_hash,
            "cal_per_class": cal_per_class, "test_labels_read_during_prepare": False}
    body["split_sha256"] = canonical_hash(body)
    write_json(output, body)
    return body


def read_manifest(path):
    body = json.loads(Path(path).read_text())
    stored = body.pop("split_sha256")
    if canonical_hash(body) != stored:
        raise ValueError("Role manifest hash mismatch")
    body["split_sha256"] = stored
    fit, cal = set(body["fit_ids"]), set(body["cal_ids"])
    if fit & cal or len(fit) != len(body["fit_ids"]) or len(cal) != len(body["cal_ids"]):
        raise ValueError("Duplicate/overlapping FIT/CAL IDs")
    if not fit or not cal or not body["test_ids"]:
        raise ValueError("Empty data role")
    return body


class RoleDataset(Dataset):
    def __init__(self, root, manifest, role, augment=False, seed=0, allow_test=False):
        if role not in ("fit", "cal", "test"):
            raise ValueError(role)
        if role == "test" and not allow_test:
            raise PermissionError("TEST is available only through a valid freeze and evaluate command")
        self.manifest = read_manifest(manifest) if not isinstance(manifest, dict) else manifest
        self.role, self.augment, self.seed, self.epoch = role, augment, seed, 0
        root = Path(root)
        if self.manifest["dataset"] == "synthetic_smoke":
            path = root / "synthetic.pt"
            if digest_file(path) != self.manifest["train_data_sha256"]:
                raise ValueError("Synthetic data content changed since prepare")
            raw = torch.load(path, map_location="cpu", weights_only=True)
            prefix = "test" if role == "test" else "train"
            self.images, self.labels = raw[prefix + "_x"], raw[prefix + "_y"]
        else:
            if digest_file(root / "cifar-100-python" / "train") != self.manifest["train_data_sha256"]:
                raise ValueError("CIFAR training data changed since prepare")
            self.images, self.labels = raw_cifar(root, "test" if role == "test" else "train")
        self.ids = self.manifest[role + "_ids"]
        if any(i < 0 or i >= len(self.images) for i in self.ids):
            raise ValueError("Out-of-range role index")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        idx = self.ids[item]
        x = self.images[idx].float() / 255.0
        if self.augment:
            # A sample receives exactly the same crop/flip across all methods,
            # independently of critic sampling, optimizer state and worker count.
            g = torch.Generator().manual_seed((self.seed + self.epoch * 1000003 + idx * 9176) % (2**63 - 1))
            x = F.pad(x, (4, 4, 4, 4))
            top, left = torch.randint(0, 9, (2,), generator=g).tolist()
            x = x[:, top:top + 32, left:left + 32]
            if torch.rand((), generator=g) < 0.5:
                x = x.flip(-1)
        return (x - MEAN) / STD, self.labels[idx], idx
