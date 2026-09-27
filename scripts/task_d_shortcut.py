"""Second-application task-level pre-check D-T — shortcut-reliance detection in trained classifiers.

    python scripts/task_d_shortcut.py --rho 0.95 --seed 0 --out <dir> [--shortcut colour|blur] [--shift S] [--smoke]

Trains a supervised CIFAR ResNet-18 (the SSL trunk of src/vcs_ssl/models/backbone.py + a linear head, no pretrained weights) on 40 000 of the
45 000 *fit* images of the frozen split with a planted colour-temperature shortcut: N_i = parity(y_i) with probability rho, otherwise
1 − parity(y_i); images with N_i = 1 get R×(1+s), B×(1−s), s = 0.2, planted once on uint8 (deterministic per image).  A fixed seeded 5 000 of
the fit images (held-out seed independent of the model seed) never enter training and serve detection.
Evaluation per model: (a) reliance on the 5 000 selection images = acc(colour matched to parity) − acc(colour flipped); acc(clean) reported;
(b) detection features: penultimate 512-d h of the 5 000 held-out images with N ~ Bernoulli(1/2) drawn independently of Y and the shift
planted on N = 1 at s = 0.2 (case cond_colour0.2) and at s = 0 (case cond_null, exact conditional independence), written in the pre-check D
feature-dir format so scripts/precheck_d_tests.py runs the conditional tests (within-class shuffling) unchanged.  No official test set.
Second shortcut family (P55 addendum 1): --shortcut blur plants a Gaussian blur of radius --shift px (default 1.0; PIL GaussianBlur, as in
precheck_d_features.py --nuisance blur) instead of the colour shift; cases are then cond_blur{shift:g} and cond_null.  --shortcut colour
(default) is byte-identical to the frozen P55 behaviour (default shift 0.2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from vcs_ssl.data.cifar import load_cifar10_train
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.models.backbone import build_resnet18_cifar
from vcs_ssl.utils import atomic_write_json, utc_now

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/infres/yinwang/CS_QMI/data/cifar10")
MANIFEST = os.environ.get("CIFAR_MANIFEST", "/home/infres/yinwang/CS_QMI/manifests/cifar10_dev45k_val5k.json")
STEM = {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False, "maxpool": False}
NORM = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
HOLDOUT_SEED = 20260927


def sha_ids(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def plant(img: np.ndarray, s: float) -> np.ndarray:
    f = img.astype(np.float32); f[..., 0] *= (1.0 + s); f[..., 2] *= (1.0 - s)
    return np.clip(f, 0, 255).astype(np.uint8)


class Planted(Dataset):
    """uint8 CIFAR images by uid; the shortcut of strength s (kind "colour": colour-temperature shift; kind "blur": Gaussian blur of radius s px)
    planted on the items with n = 1 before the transform."""

    def __init__(self, images, targets, uids, n, s, tf, kind="colour"):
        self.images, self.targets, self.uids, self.n, self.s, self.tf = images, np.asarray(targets), np.asarray(uids), np.asarray(n), float(s), tf
        self.kind = kind

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = int(self.uids[i]); img = self.images[uid]
        if self.n[i] == 1 and self.s > 0 and self.kind == "colour":
            img = plant(img, self.s)
        pil = Image.fromarray(img)
        if self.n[i] == 1 and self.s > 0 and self.kind == "blur":
            pil = pil.filter(ImageFilter.GaussianBlur(radius=self.s))
        return self.tf(pil), int(self.targets[uid]), int(self.n[i]), uid


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.trunk = build_resnet18_cifar(STEM); self.head = nn.Linear(512, 10)

    def features(self, x):
        return self.trunk(x)

    def forward(self, x):
        return self.head(self.trunk(x))


@torch.no_grad()
def run_eval(model, ds, device, bs, workers, want_features=False):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers); correct = 0; H, Y, N, U = [], [], [], []
    for x, y, n, u in dl:
        x = x.to(device, non_blocking=True); h = model.features(x); logits = model.head(h)
        correct += int((logits.argmax(1).cpu() == y).sum())
        if want_features:
            H.append(h.float().cpu()); Y.append(y); N.append(n); U.append(u)
    acc = correct / len(ds)
    return (acc, torch.cat(H), torch.cat(Y), torch.cat(N), torch.cat(U)) if want_features else acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shortcut", default="colour", choices=["colour", "blur"], help="shortcut family: colour-temperature shift (P55) or Gaussian blur (addendum 1)")
    ap.add_argument("--shift", type=float, default=None, help="colour: shift s (default 0.2); blur: radius in px (default 1.0)")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--momentum", type=float, default=0.9); ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--holdout", type=int, default=5000); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-epochs", type=int, default=2); ap.add_argument("--smoke-train", type=int, default=2000, help="smoke only: epochs / training images")
    a = ap.parse_args()
    if a.shift is None:
        a.shift = 0.2 if a.shortcut == "colour" else 1.0
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = load_cifar10_train(DATA_ROOT); man = load_manifest(MANIFEST)
    fit = np.asarray(man["fit_uids"], dtype=np.int64); sel = np.asarray(man["selection_uids"], dtype=np.int64); y_all = np.asarray(data.targets)
    # held-out detection pool (fixed, independent of the model seed) and the training pool
    perm = np.random.default_rng(HOLDOUT_SEED).permutation(fit); held, train = perm[: a.holdout], perm[a.holdout:]
    n_train_smoke = None
    if a.smoke:
        a.epochs, a.batch, n_train_smoke = a.smoke_epochs, 128, a.smoke_train; train, held, sel = train[:n_train_smoke], held[:1000], sel[:1000]
    train, held = np.sort(train), np.sort(held)
    # planted shortcut on the training pool: N = parity(y) w.p. rho
    rng = np.random.default_rng([a.seed, 20260927]); par = (y_all[train] % 2).astype(np.int64)
    agree = rng.random(len(train)) < a.rho; n_train = np.where(agree, par, 1 - par)
    tf_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(**NORM)])
    tf_clean = T.Compose([T.ToTensor(), T.Normalize(**NORM)])
    torch.manual_seed(a.seed); model = Net().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd, nesterov=True)
    dl = DataLoader(Planted(data.data, data.targets, train, n_train, a.shift, tf_train, kind=a.shortcut), batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, generator=torch.Generator().manual_seed(a.seed), persistent_workers=a.workers > 0)
    total = a.epochs * len(dl); step = 0; ce = nn.CrossEntropyLoss(); t0 = time.time(); log = []
    for ep in range(1, a.epochs + 1):
        model.train(); tot, correct, seen = 0.0, 0, 0
        for x, y, _, _ in dl:
            lr = 0.5 * a.lr * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x); loss = ce(logits, y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            tot += float(loss.detach()) * len(y); correct += int((logits.argmax(1) == y).sum()); seen += len(y)
        log.append({"epoch": ep, "loss": tot / seen, "train_acc": correct / seen, "seconds": time.time() - t0})
        print(f"[rho={a.rho:g} seed={a.seed}] epoch {ep}/{a.epochs} loss {tot / seen:.3f} acc {correct / seen:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval(); torch.save({"model_state": model.state_dict(), "rho": a.rho, "seed": a.seed, "epochs": a.epochs}, out / "model.pt")
    # (a) reliance on the selection images: colour matched to parity / flipped / clean
    par_sel = (y_all[sel] % 2).astype(np.int64); bs = 512
    acc = {"clean": run_eval(model, Planted(data.data, data.targets, sel, np.zeros(len(sel), np.int64), 0.0, tf_clean, kind=a.shortcut), device, bs, a.workers),
           "matched": run_eval(model, Planted(data.data, data.targets, sel, par_sel, a.shift, tf_clean, kind=a.shortcut), device, bs, a.workers),
           "flipped": run_eval(model, Planted(data.data, data.targets, sel, 1 - par_sel, a.shift, tf_clean, kind=a.shortcut), device, bs, a.workers)}
    acc["reliance"] = acc["matched"] - acc["flipped"]
    print(f"[rho={a.rho:g} seed={a.seed}] selection acc clean {acc['clean']:.4f} matched {acc['matched']:.4f} flipped {acc['flipped']:.4f} reliance {acc['reliance']:+.4f}", flush=True)
    # (b) detection features on the held-out pool: N independent of Y
    n_held = np.random.default_rng([HOLDOUT_SEED, a.seed]).integers(0, 2, size=len(held))
    tag = "" if a.shortcut == "colour" else f"_{a.shortcut}"
    planting = ("N=1 images: R*(1+s), B*(1-s) on uint8, clipped; held-out N ~ Bernoulli(1/2) independent of Y" if a.shortcut == "colour"
                else "N=1 images: PIL GaussianBlur(radius=s px) at 32x32; held-out N ~ Bernoulli(1/2) independent of Y")
    manifest = {"run": f"task_d_shortcut{tag}_rho{a.rho:g}_seed{a.seed}", "checkpoint": "model.pt", "split_hash": man["manifest_sha256"], "n_fit": int(len(held)),
                "task": "D-T shortcut-reliance detection", "shortcut": a.shortcut, "rho": a.rho, "seed": a.seed, "shift": a.shift, "smoke": a.smoke,
                "training": {"epochs": a.epochs, "batch": a.batch, "lr": a.lr, "momentum": a.momentum, "nesterov": True, "wd": a.wd, "schedule": "per-step cosine, no warm-up",
                             "aug": "RandomCrop(32, pad 4) + HorizontalFlip", "n_train": int(len(train)), "train_n1_frac": float(n_train.mean()),
                             "train_agree_frac": float(agree.mean()), "log": log},
                "splits": {"train_sha256": sha_ids(train), "held_sha256": sha_ids(held), "selection_sha256": sha_ids(sel), "holdout_seed": HOLDOUT_SEED, "n_held": int(len(held)), "n_selection": int(len(sel))},
                "selection_accuracy": acc, "planting": planting,
                "cases": {}, "utc": utc_now()}
    for name, s in ((f"cond_{a.shortcut}{a.shift:g}", a.shift), ("cond_null", 0.0)):
        _, H, Y, N, U = run_eval(model, Planted(data.data, data.targets, held, n_held, s, tf_clean, kind=a.shortcut), device, bs, a.workers, want_features=True)
        f = out / f"{name}.pt"; torch.save({"h": H, "y": Y, "n": N, "uids": U, "strength": s}, f)
        manifest["cases"][name] = {"file": f.name, "strength": s, "n1_frac": float(N.float().mean()), "h_mean_abs": float(H.abs().mean()),
                                   "corr_N_Y_even": float(np.corrcoef(N.numpy(), (Y.numpy() % 2 == 0).astype(float))[0, 1])}
        print(f"[rho={a.rho:g} seed={a.seed}] {name}: h {tuple(H.shape)} N1 {N.float().mean():.3f}", flush=True)
    atomic_write_json(out / "manifest.json", manifest); atomic_write_json(out / "result.json", {k: v for k, v in manifest.items() if k != "cases"})
    print("done ->", out); return 0


if __name__ == "__main__":
    sys.exit(main())
