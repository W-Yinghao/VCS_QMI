"""P154 — ImageNet-100 pilot: VCS (A-P3) / matched JS / SimCLR pretraining on the CMC 100-class subset, single GPU, then the CIFAR pilot readout
(frozen-h linear probe + kNN) on a held-out development split.  Reuses the tested pieces: P132 (`vcs_ssl.imagenet`: manifests, the all-view-token
VCS / JS loss and the multi-view SimCLR loss, world size 1) and P139 (`vcs_ssl.imagenet_gpu_aug`: PIL RandomResizedCrop per view on the CPU, the
vectorised flip / jitter / grey / blur / normalisation stage on the GPU with per-image parameters).

Recipe (shared by the three methods, as on CIFAR): torchvision ResNet-18 (standard ImageNet stem, h 512), projector 512 -> 512 (BN, ReLU) -> 128,
L2-normalised z; 4 views at 224 px; batch 256 images; AdamW lr 1e-3, betas (0.9, 0.999), weight decay 1e-4 on encoder / projector matrices (0 on
biases and BN), 10 warm-up epochs then cosine to 0.01 x lr per optimizer step; bf16 autocast for encoder + projector, the loss in float32.
VCS / JS: fixed A-P3 scorer T = tanh(2 s - 1) on all view tokens; SimCLR: NT-Xent over all view pairs, tau 0.2.

Data: ImageNet-1k train rows of the 100 CMC classes (solo-learn list); development split = 50 images per class held out (seed 20261008) for the
readout; the remaining images train the encoder (labels unused) and the probe.  The ImageNet validation set is not touched.
    python scripts/p154_in100.py train --method vcs --run-id P154_IN100_vcs_r18_200ep_seed0 [--epochs 200] [--workers 30]
    python scripts/p154_in100.py eval  --run-id ... [--checkpoint epoch_200.pt]
    python scripts/p154_in100.py train ... --fake        (synthetic 64-px data, CPU tests / smoke)
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
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_ssl import imagenet as IN  # noqa: E402
from vcs_ssl import imagenet_gpu_aug as GA  # noqa: E402
from vcs_ssl.diagnostics import knn_eval, linear_probe  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

CLASS_FILE = Path("/home/infres/yinwang/CS_QMI/solo_learn/solo/data/dataset_subset/imagenet100_classes.txt")
OUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))
SPLIT_SEED, VAL_PER_CLASS = 20261008, 50
RECIPE = {"backbone": "torchvision_resnet18", "h_dim": 512, "projector": [512, 512, 128], "views": 4, "size": 224, "min_scale": 0.08,
          "batch": 256, "lr": 1e-3, "betas": (0.9, 0.999), "wd": 1e-4, "warmup_epochs": 10, "min_lr_ratio": 0.01, "vcs_scale": 2.0,
          "vcs_bias": -1.0, "simclr_tau": 0.2, "precision": "bf16_autocast_encoder_projector"}
# the CIFAR pilot readout, unchanged (diagnostics.linear_probe validates these keys)
LINEAR = {"normalize_h": False, "head": "linear_with_bias", "epochs": 100, "batch_size": 256, "optimizer": "sgd", "lr": 0.1, "momentum": 0.9,
          "weight_decay": 0.0, "schedule": "cosine", "min_lr_ratio": 0.001, "seed": 20260925, "checkpoint_rule": "final_probe_epoch"}
KNN = {"k": 200, "temperature": 0.1}
MILESTONES = (50, 100, 200)


# ----------------------------------------------------------------------------------------------------------- data
def in100_split(rows=None, class_file: Path = CLASS_FILE, seed: int = SPLIT_SEED, val_per_class: int = VAL_PER_CLASS):
    """(train rows, dev-val rows, class index) for the 100 CMC classes; per class, `val_per_class` rows (seeded permutation of the class's rows
    sorted by path) are held out."""
    wnids = class_file.read_text().split()
    if len(wnids) != 100 or len(set(wnids)) != 100:
        raise ValueError("expected 100 distinct wnids")
    keep = set(wnids); rows = rows if rows is not None else IN.read_manifest("train")
    by = {w: sorted(r for r in rows if r[1] == w) for w in sorted(keep)}
    rng = np.random.default_rng(seed); train, val = [], []
    for w in sorted(keep):
        rs = by[w]; p = rng.permutation(len(rs))
        val += [rs[i] for i in p[:val_per_class]]; train += [rs[i] for i in p[val_per_class:]]
    cls = {w: i for i, w in enumerate(sorted(keep))}
    return train, val, cls


def split_digest(train, val) -> str:
    h = hashlib.sha256()
    for r in train + [("--", "--")] + val:
        h.update(f"{r[0]}\t{r[1]}\n".encode())
    return h.hexdigest()


class FakeCrops(torch.utils.data.Dataset):
    """Synthetic stand-in for ImageNetCropViews (tests / smoke): uint8 [V, 3, s, s], label, index; deterministic per index."""

    def __init__(self, n: int = 512, views: int = 4, size: int = 64, classes: int = 10) -> None:
        self.n, self.views, self.size, self.classes = n, views, size, classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(i)
        base = torch.randint(0, 256, (3, self.size, self.size), generator=g, dtype=torch.uint8)
        v = torch.stack([torch.roll(base, shifts=(int(torch.randint(0, 4, (1,), generator=g)),), dims=(2,)) for _ in range(self.views)])
        return v, i % self.classes, i


class CenterCrops(torch.utils.data.Dataset):
    """Evaluation view: resize 256 (bilinear, antialias) -> centre crop 224, uint8 [3, 224, 224] (normalised on the GPU)."""

    def __init__(self, rows, classes, root: str = IN.IMAGENET_ROOT) -> None:
        from torchvision import transforms as T
        self.rows, self.cls, self.root = rows, classes, root
        self.tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR, antialias=True), T.CenterCrop(224)])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor
        rel, w = self.rows[i]
        with Image.open(os.path.join(self.root, rel)) as im:
            return pil_to_tensor(self.tf(im.convert("RGB"))), self.cls[w]


class FakeCenter(torch.utils.data.Dataset):
    def __init__(self, n: int = 256, size: int = 64, classes: int = 10) -> None:
        self.n, self.size, self.classes = n, size, classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(10_000 + i)
        return torch.randint(0, 256, (3, self.size, self.size), generator=g, dtype=torch.uint8), i % self.classes


# ----------------------------------------------------------------------------------------------------------- model
class Projector(nn.Sequential):
    def __init__(self, d_in: int = 512, hidden: int = 512, out: int = 128) -> None:
        super().__init__(nn.Linear(d_in, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out, bias=True))


def build(seed: int):
    import torchvision
    torch.manual_seed(seed)
    enc = torchvision.models.resnet18(weights=None); enc.fc = nn.Identity()
    proj = Projector(RECIPE["h_dim"], RECIPE["projector"][1], RECIPE["projector"][2])
    return enc, proj


def param_groups(*modules: nn.Module, wd: float):
    decay, no_decay = [], []
    for m in modules:
        for p in m.parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, steps_per_epoch: int, epochs: int, base: float = RECIPE["lr"], warmup_epochs: int = RECIPE["warmup_epochs"],
          min_ratio: float = RECIPE["min_lr_ratio"]) -> float:
    """Linear warm-up from 0 over `warmup_epochs`, then cosine from base to min_ratio * base at the last step (per optimizer step)."""
    w, total = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch
    if step < w:
        return base * (step + 1) / w
    prog = (step - w) / max(total - w - 1, 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def ssl_loss(method: str, zs_raw: list[torch.Tensor]) -> dict:
    """zs_raw: V tensors [B, 128] (any dtype).  VCS / JS on L2-normalised float32 tokens (A-P3 scorer); SimCLR normalises inside."""
    if method == "simclr":
        return IN.dist_simclr_multiview_loss([z.float() for z in zs_raw], temperature=RECIPE["simclr_tau"])
    zs = [F.normalize(z.float(), dim=-1, eps=1e-8) for z in zs_raw]
    return IN.dist_allview_tokens_loss(zs, scale=RECIPE["vcs_scale"], bias=RECIPE["vcs_bias"], objective=method, chunk_size=4096)


def forward_views(enc, proj, x_u8: torch.Tensor, aug, device, amp: bool) -> list[torch.Tensor]:
    """x_u8 [B, V, 3, s, s] -> view-major batch -> GPU augmentation -> one encoder pass -> V tensors z [B, 128]."""
    B, V = x_u8.shape[:2]
    x = x_u8.to(device, non_blocking=True).transpose(0, 1).reshape(V * B, *x_u8.shape[2:])
    x = aug(x)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        z = proj(enc(x))
    return list(z.float().split(B, dim=0))


# ----------------------------------------------------------------------------------------------------------- training
def run_dir_of(run_id: str) -> Path:
    return OUT_ROOT / run_id


def train(a) -> int:
    dev = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    amp = dev.type == "cuda"
    rd = run_dir_of(a.run_id); (rd / "checkpoints").mkdir(parents=True, exist_ok=True); (rd / "logs").mkdir(exist_ok=True)
    if a.fake:
        ds = FakeCrops(n=a.fake_n, views=RECIPE["views"], size=64); manifest = {"fake": True, "n": a.fake_n}
    else:
        tr, va, cls = in100_split()
        ds = GA.ImageNetCropViews(views=RECIPE["views"], size=RECIPE["size"], min_scale=RECIPE["min_scale"], rows=tr, classes=cls)
        manifest = {"classes_file": str(CLASS_FILE), "n_classes": 100, "n_train": len(tr), "n_dev_val": len(va), "split_seed": SPLIT_SEED,
                    "val_per_class": VAL_PER_CLASS, "split_sha256": split_digest(tr, va), "official_val_used": False}
    atomic_write_json(rd / "manifest.json", manifest)
    cfg = {"run_id": a.run_id, "method": a.method, "seed": a.seed, "epochs": a.epochs, "recipe": RECIPE, "linear": LINEAR, "knn": KNN,
           "workers": a.workers, "fake": a.fake, "max_steps_per_epoch": a.max_steps_per_epoch}
    atomic_write_json(rd / "config.json", cfg)
    enc, proj = build(a.seed); enc.to(dev); proj.to(dev)
    if dev.type == "cuda":
        enc = enc.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(enc, proj, wd=RECIPE["wd"]), lr=RECIPE["lr"], betas=RECIPE["betas"], eps=1e-8)
    bs = RECIPE["batch"] if not a.fake else a.fake_batch
    steps_per_epoch = len(ds) // bs if a.max_steps_per_epoch is None else min(len(ds) // bs, a.max_steps_per_epoch)
    start_epoch, step = 0, 0
    last = rd / "checkpoints" / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck["encoder"]); proj.load_state_dict(ck["projector"]); opt.load_state_dict(ck["optimizer"])
        start_epoch, step = ck["completed_epoch"], ck["step"]
        print(f"[resume] {a.run_id} from epoch {start_epoch} step {step}", flush=True)
    atomic_write_json(rd / "status.json", {"status": "RUNNING", "utc": utc_now(), "epoch": start_epoch})
    for epoch in range(start_epoch, a.epochs):
        g = torch.Generator().manual_seed(a.seed * 1_000_003 + epoch)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, generator=g, num_workers=a.workers, drop_last=True,
                                         pin_memory=dev.type == "cuda", persistent_workers=False, prefetch_factor=4 if a.workers else None)
        aug = GA.GpuViewAug(seed=a.seed * 7_000_003 + epoch)
        enc.train(); proj.train()
        t0, n_img, tl, tj, data_wait = time.time(), 0, 0.0, 0.0, 0.0; tw = time.time()
        for i, (x, _, _) in enumerate(dl):
            if i >= steps_per_epoch:
                break
            data_wait += time.time() - tw
            lr = lr_at(step, steps_per_epoch, a.epochs)
            for gr in opt.param_groups:
                gr["lr"] = lr
            zs = forward_views(enc, proj, x, aug, dev, amp)
            out = ssl_loss(a.method, zs)
            opt.zero_grad(set_to_none=True); out["loss"].backward(); opt.step()
            if not torch.isfinite(out["loss"]):
                atomic_write_json(rd / "status.json", {"status": "FAILED_NONFINITE", "utc": utc_now(), "epoch": epoch, "step": step})
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            tl += float(out["loss"]); tj += float(out.get("J_global", torch.tensor(0.0))); n_img += x.shape[0]; step += 1
            tw = time.time()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sec = time.time() - t0
        rec = {"epoch": epoch + 1, "step": step, "loss": tl / steps_per_epoch, "J_train": tj / steps_per_epoch if a.method != "simclr" else None,
               "lr_end": lr, "epoch_seconds": sec, "images_per_s": n_img / sec, "data_wait_seconds": data_wait, "utc": utc_now()}
        with open(rd / "logs" / "epochs.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{a.run_id}] epoch {epoch + 1}/{a.epochs} loss {rec['loss']:.4f} J {rec['J_train']} {rec['images_per_s']:.0f} img/s "
              f"(data wait {data_wait:.0f}s of {sec:.0f}s)", flush=True)
        state = {"encoder": enc.state_dict(), "projector": proj.state_dict(), "optimizer": opt.state_dict(), "completed_epoch": epoch + 1,
                 "step": step, "config": cfg}
        tmp = last.with_suffix(".tmp"); torch.save(state, tmp); os.replace(tmp, last)
        if epoch + 1 in MILESTONES or epoch + 1 == a.epochs:
            torch.save({k: v for k, v in state.items() if k != "optimizer"}, rd / "checkpoints" / f"epoch_{epoch + 1}.pt")
        if a.stop_after_epochs is not None and epoch + 1 - start_epoch >= a.stop_after_epochs:
            print(f"[stop] after {a.stop_after_epochs} epoch(s) this invocation (test of resume)", flush=True); return 0
    atomic_write_json(rd / "status.json", {"status": "COMPLETED", "utc": utc_now(), "epoch": a.epochs})
    return 0


# ----------------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def extract_h(enc, ds, dev, workers: int, bs: int = 256):
    mean = torch.tensor(IN.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IN.IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=dev.type == "cuda")
    H, Y = [], []
    for x, y in dl:
        x = (x.to(dev, non_blocking=True).float() / 255.0 - mean) / std
        if dev.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            h = enc(x)
        H.append(h.float().cpu()); Y.append(torch.as_tensor(y))
    return torch.cat(H), torch.cat(Y)


def evaluate(a) -> int:
    dev = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    rd = run_dir_of(a.run_id); ck = torch.load(rd / "checkpoints" / a.checkpoint, map_location="cpu", weights_only=False)
    enc, _ = build(0); enc.load_state_dict(ck["encoder"]); enc.to(dev).eval()
    if dev.type == "cuda":
        enc = enc.to(memory_format=torch.channels_last)
    t0 = time.time()
    if a.fake:
        tr_ds, va_ds, ncls = FakeCenter(n=a.fake_n, size=64), FakeCenter(n=128, size=64), 10
    else:
        tr, va, cls = in100_split(); ncls = 100
        tr_ds, va_ds = CenterCrops(tr, cls), CenterCrops(va, cls)
    hf, yf = extract_h(enc, tr_ds, dev, a.workers); hv, yv = extract_h(enc, va_ds, dev, a.workers)
    t_feat = time.time() - t0
    lp = linear_probe(hf, yf, hv, yv, LINEAR, device=dev, n_classes=ncls)
    kn = knn_eval(hf, yf, hv, yv, k=min(KNN["k"], len(yf)), temperature=KNN["temperature"], n_classes=ncls, device=dev)  # labels stay on the CPU; k < 200 only for the synthetic test bank
    res = {"run_id": a.run_id, "checkpoint": a.checkpoint, "completed_epoch": ck["completed_epoch"], "n_fit": int(len(yf)), "n_val": int(len(yv)),
           "linear_val_top1_pct": float(lp["linear_val_top1_pct"]), "knn_val_top1_pct": float(kn["knn_val_top1_pct"]),
           "linear": {k: v for k, v in lp.items() if k != "curve"}, "linear_curve": lp["curve"], "knn": {k: v for k, v in kn.items()},
           "feature_seconds": t_feat, "seconds": time.time() - t0, "official_val_used": False, "utc": utc_now()}
    (rd / "evaluations").mkdir(exist_ok=True)
    atomic_write_json(rd / "evaluations" / f"evaluation_{Path(a.checkpoint).stem}.json", res)
    print(f"[eval {a.run_id} {a.checkpoint}] linear {res['linear_val_top1_pct']:.2f} kNN {res['knn_val_top1_pct']:.2f} ({res['seconds']:.0f}s)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True); p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 2)
        p.add_argument("--fake", action="store_true"); p.add_argument("--fake-n", type=int, default=512); p.add_argument("--cpu", action="store_true")
        if name == "train":
            p.add_argument("--method", required=True, choices=["vcs", "js", "simclr"]); p.add_argument("--seed", type=int, default=0)
            p.add_argument("--epochs", type=int, default=200); p.add_argument("--max-steps-per-epoch", type=int, default=None)
            p.add_argument("--fake-batch", type=int, default=32); p.add_argument("--stop-after-epochs", type=int, default=None)
        else:
            p.add_argument("--checkpoint", default="epoch_200.pt")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    return train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
