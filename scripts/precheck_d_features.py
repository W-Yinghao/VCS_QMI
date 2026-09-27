"""Second-application pre-check D (leakage detection power) — step 1: frozen features with a planted nuisance.

    python scripts/precheck_d_features.py --run <run_id> --checkpoint epoch_800.pt --out <dir> [--strengths 0,0.01,...] [--cond-strength 0.3 --cond-colour 0.05]
                                          [--nuisance colour|blur --cond-blur 0.5] [--limit N]

For the 45k *fit* images of the frozen split (labels never used by the encoder), draw a nuisance bit N_i ~ Bernoulli(1/2) i.i.d. with a fixed
seed (the SAME assignment for every strength, so strength 0 is exact independence), plant a colour-temperature shift of strength s on the
images with N_i = 1 (R channel × (1+s), B channel × (1−s), clipped), and store the frozen encoder's h for every strength.
Conditional cases (nuisance correlated with the class Y): N_c | Y ~ Bernoulli(1/2 + c·(+1 if Y even else −1)); case "label_only" = images unchanged
(Z ⊥ N_c | Y holds exactly), case "label_colour" = the N_c = 1 images additionally colour-shifted by --cond-colour (Z depends on N_c beyond Y).
Second nuisance family (wave 2, D-S3): --nuisance blur plants a Gaussian blur of radius s px (PIL GaussianBlur) on the N = 1 images instead of
the colour shift; the conditional case is then "cond_label_blur{--cond-blur}".  --limit N restricts the fit pool to its first N uids (smoke only).
Nothing is trained.  Output: one .pt per case with h [45000,512] float32, N, Y, uids, and a manifest with all seeds and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import load_resolved
from vcs_ssl.data.cifar import load_cifar10_train
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.data.transforms import build_clean_transform
from vcs_ssl.models import build_models
from vcs_ssl.utils import atomic_write_json, utc_now


class PlantedDataset(Dataset):
    """Clean images; for N=1 a colour-temperature shift of strength s (R×(1+s), B×(1−s)) applied on uint8 before the clean transform
    (kind="colour"), or a Gaussian blur of radius s px (kind="blur")."""

    def __init__(self, images, targets, uids, nuisance, strength, transform, kind="colour"):
        self.images, self.targets, self.uids, self.n, self.s, self.tf = images, np.asarray(targets), np.asarray(uids), np.asarray(nuisance), float(strength), transform
        self.kind = kind

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = int(self.uids[i]); img = self.images[uid]
        if self.n[i] == 1 and self.s > 0 and self.kind == "colour":
            f = img.astype(np.float32)
            f[..., 0] *= (1.0 + self.s); f[..., 2] *= (1.0 - self.s)
            img = np.clip(f, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        if self.n[i] == 1 and self.s > 0 and self.kind == "blur":
            pil = pil.filter(ImageFilter.GaussianBlur(radius=self.s))
        return self.tf(pil), int(self.targets[uid]), int(self.n[i]), uid


@torch.no_grad()
def extract(enc, ds, device, bs=512, workers=4):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    H, Y, N, U = [], [], [], []
    for x, y, n, u in dl:
        H.append(enc(x.to(device)).float().cpu()); Y.append(y); N.append(n); U.append(u)
    return torch.cat(H), torch.cat(Y), torch.cat(N), torch.cat(U)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--strengths", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--cond-strength", type=float, default=0.3); ap.add_argument("--cond-colour", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260927); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nuisance", default="colour", choices=["colour", "blur"]); ap.add_argument("--cond-blur", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None, help="smoke only: use the first N fit uids")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    rd = Path(a.output_root) / a.run
    cfg = load_resolved(rd / "config.resolved.yaml"); man = load_manifest(rd / "manifest.json")
    ck = load_checkpoint(rd / "checkpoints" / a.checkpoint)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device); enc = built["encoder"]; enc.load_state_dict(ck["encoder_state"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    data = load_cifar10_train(cfg["data"]["root"]); fit = np.asarray(man["fit_uids"], dtype=np.int64)
    if a.limit:
        fit = fit[: a.limit]
    y = np.asarray(data.targets)[fit]
    rng = np.random.default_rng(a.seed)
    n_rand = rng.integers(0, 2, size=len(fit))                                   # independent of everything
    p_c = 0.5 + a.cond_strength * np.where(y % 2 == 0, 1.0, -1.0)               # class-correlated nuisance
    n_cond = (rng.random(len(fit)) < p_c).astype(np.int64)
    tf = build_clean_transform(cfg["views"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cases = [(f"s{s:g}", n_rand, float(s)) for s in (float(v) for v in a.strengths.split(","))]
    cond_val = a.cond_colour if a.nuisance == "colour" else a.cond_blur
    cases += [("cond_label_only", n_cond, 0.0), (f"cond_label_{a.nuisance}{cond_val:g}", n_cond, cond_val)]
    planting = ("N=1 images: R*(1+s), B*(1-s) on uint8, clipped; N ~ Bernoulli(1/2) i.i.d., same draw for all s" if a.nuisance == "colour"
                else "N=1 images: PIL GaussianBlur(radius=s px) at 32x32; N ~ Bernoulli(1/2) i.i.d., same draw for all s")
    manifest = {"run": a.run, "checkpoint": a.checkpoint, "encoder_sha256": hashlib.sha256(json.dumps({k: float(v.float().sum()) for k, v in ck["encoder_state"].items()}, sort_keys=True).encode()).hexdigest()[:16],
                "split_hash": man["manifest_sha256"], "n_fit": int(len(fit)), "seed": a.seed, "nuisance": a.nuisance, "planting": planting, "limit": a.limit,
                "cond": {"p(N=1|Y)": f"0.5 + {a.cond_strength} * (+1 if Y even else -1)", a.nuisance: cond_val}, "cases": {}, "utc": utc_now()}
    for name, nn_, s in cases:
        H, Y, N, U = extract(enc, PlantedDataset(data.data, data.targets, fit, nn_, s, tf, kind=a.nuisance), device, workers=a.workers)
        f = out / f"{name}.pt"; torch.save({"h": H, "y": Y, "n": N, "uids": U, "strength": s}, f)
        manifest["cases"][name] = {"file": f.name, "strength": s, "n1_frac": float(N.float().mean()), "h_mean_abs": float(H.abs().mean()),
                                   "corr_N_Y_even": float(np.corrcoef(N.numpy(), (Y.numpy() % 2 == 0).astype(float))[0, 1])}
        print(f"[{a.run}] {name}: h {tuple(H.shape)} N1 {N.float().mean():.3f}", flush=True)
    atomic_write_json(out / "manifest.json", manifest); print("done ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
