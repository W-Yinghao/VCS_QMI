"""Datasets and loaders.  The SSL dataset never stores or returns labels (spec §4.1-4.2)."""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SSLTwoViewDataset(Dataset):
    """Returns ``(view1, view2, uid)``.  Labels are not held by this object at all."""

    def __init__(self, images_uint8: np.ndarray, uids: np.ndarray, transform: Callable) -> None:
        if images_uint8.ndim != 4 or images_uint8.shape[1:] != (32, 32, 3) or images_uint8.dtype != np.uint8:
            raise ValueError("expected uint8 images [N,32,32,3]")
        self.images = images_uint8
        self.uids = np.asarray(uids, dtype=np.int64)
        if len(np.unique(self.uids)) != len(self.uids):
            raise ValueError("duplicate UIDs in SSL dataset")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = int(self.uids[i])
        img = Image.fromarray(self.images[uid])
        v1 = self.transform(img)  # two independent calls: independent random parameters
        v2 = self.transform(img)
        return v1, v2, uid


class SSLMultiViewDataset(Dataset):
    """Named variant: returns ``(view_1, ..., view_n, uid)`` with n independent augmentation calls; no labels."""

    def __init__(self, images_uint8: np.ndarray, uids: np.ndarray, transform: Callable, n_views: int) -> None:
        if n_views < 2:
            raise ValueError("n_views must be >= 2")
        self.images = images_uint8
        self.uids = np.asarray(uids, dtype=np.int64)
        self.transform = transform
        self.n_views = int(n_views)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = int(self.uids[i])
        img = Image.fromarray(self.images[uid])
        return (*[self.transform(img) for _ in range(self.n_views)], uid)


class LabeledCleanDataset(Dataset):
    """Evaluation-only dataset: ``(x_clean, label, uid)``.  Used by probes/kNN/spectrum, never by SSL training."""

    def __init__(self, images_uint8: np.ndarray, targets: np.ndarray, uids: np.ndarray, transform: Callable) -> None:
        self.images = images_uint8
        self.targets = np.asarray(targets, dtype=np.int64)
        self.uids = np.asarray(uids, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = int(self.uids[i])
        return self.transform(Image.fromarray(self.images[uid])), int(self.targets[uid]), uid


class TwoViewNoLabelEvalDataset(Dataset):
    """Critic held-out diagnostic: two random views of selection images, no labels (spec §10.4)."""

    def __init__(self, images_uint8: np.ndarray, uids: np.ndarray, transform: Callable) -> None:
        self.images = images_uint8
        self.uids = np.asarray(uids, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = int(self.uids[i])
        img = Image.fromarray(self.images[uid])
        return self.transform(img), self.transform(img), uid


def seed_worker(worker_id: int) -> None:
    """Derive numpy/python seeds from the torch worker seed (itself drawn from the loader generator)."""
    s = torch.initial_seed() % (2**32)
    np.random.seed(s)
    random.seed(s)


def make_ssl_loader(dataset: Dataset, *, batch_size: int, num_workers: int, pin_memory: bool, persistent_workers: bool,
                    generator: torch.Generator, drop_last: bool = True) -> DataLoader:
    if generator.device.type != "cpu":
        raise ValueError("loader generator must be a CPU generator")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=persistent_workers and num_workers > 0,
                      generator=generator, worker_init_fn=seed_worker)


def make_eval_loader(dataset: Dataset, *, batch_size: int, num_workers: int, generator: torch.Generator,
                     pin_memory: bool = True) -> DataLoader:
    """Deterministic order, dedicated generator so evaluation never consumes training RNG."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                      pin_memory=pin_memory, generator=generator, worker_init_fn=seed_worker)
