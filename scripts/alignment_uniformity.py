"""Read-only geometry diagnostic (Wang & Isola 2020 alignment / uniformity + pair-similarity quantiles) on frozen checkpoints.

    python scripts/alignment_uniformity.py --runs A,B,... --out reports/GEOMETRY_DIAG.md

For each run: load epoch_<final>.pt into a fresh model copy (eval mode, frozen), draw two train-distribution views of the 5,000
selection images (fixed RNG), and compute on z_l2 (projector output, L2) and h_l2 (encoder output, L2):
  alignment   = mean ||z1_i - z2_i||^2 over positive pairs                      (lower = views closer)
  uniformity  = log mean_{i != j} exp(-2 ||z_i - z_j||^2) over one view          (lower = more uniform on the sphere)
  cosine quantiles of positive pairs and of shifted (negative) pairs; for cosine critics the learned a, b and the implied threshold -b/a.
Nothing is trained; SimCLR / VICReg controls are processed with the same code (their critic fields are null).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import load_resolved
from vcs_ssl.data.cifar import load_cifar10_train
from vcs_ssl.data.datasets import TwoViewNoLabelEvalDataset, make_eval_loader
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.data.transforms import build_two_view_transform
from vcs_ssl.models import build_models
from vcs_ssl.utils import atomic_write_json, utc_now


@torch.no_grad()
def uniformity(z: torch.Tensor, t: float = 2.0, max_n: int = 5000) -> float:
    z = z[:max_n]
    d2 = torch.cdist(z, z).pow(2)
    mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    return float(torch.log(torch.exp(-t * d2[mask]).mean()))


def q(x: torch.Tensor):
    qs = torch.quantile(x.float().cpu(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return [round(float(v), 4) for v in qs]


@torch.no_grad()
def diagnose(run_dir: Path, device: torch.device) -> dict:
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    man = load_manifest(run_dir / "manifest.json")
    ck_name = f"epoch_{cfg['train']['epochs']:03d}.pt"
    ck = load_checkpoint(run_dir / "checkpoints" / ck_name)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"])
    if crit is not None:
        crit.load_state_dict(ck["critic_state"])
    for m in (enc, proj, crit):
        if m is not None:
            m.eval()
    data = load_cifar10_train(cfg["data"]["root"])
    sel = np.asarray(man["selection_uids"])
    tf = build_two_view_transform(cfg["views"])
    loader = make_eval_loader(TwoViewNoLabelEvalDataset(data.data, sel, tf), batch_size=500, num_workers=4, generator=torch.Generator().manual_seed(777),
                              pin_memory=device.type == "cuda")
    Z1, Z2, H1, H2 = [], [], [], []
    for x1, x2, _ in loader:
        h = enc(torch.cat((x1, x2)).to(device)); p = proj(h)
        z = F.normalize(p, dim=1); hl = F.normalize(h, dim=1)
        z1, z2 = z.chunk(2); h1, h2 = hl.chunk(2)
        Z1.append(z1); Z2.append(z2); H1.append(h1); H2.append(h2)
    Z1, Z2, H1, H2 = (torch.cat(t) for t in (Z1, Z2, H1, H2))
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(len(Z1), generator=g)
    perm = torch.where(perm == torch.arange(len(Z1)), (perm + 1) % len(Z1), perm).to(device)
    out = {"run": run_dir.name, "method": cfg["run"]["method"], "epochs": cfg["train"]["epochs"], "checkpoint": ck_name,
           "critic_input": cfg["model"]["critic"]["input"] if cfg["model"]["critic"]["enabled"] else None,
           "K": cfg["pairing"]["k"], "negative_detach": cfg["pairing"]["negative_detach"], "n_selection": int(len(sel))}
    for name, A, B in (("z_l2", Z1, Z2), ("h_l2", H1, H2)):
        cos_pos = (A * B).sum(-1); cos_neg = (A * B[perm]).sum(-1)
        out[name] = {"alignment": float((A - B).pow(2).sum(-1).mean()), "uniformity": uniformity(A),
                     "cos_pos_q05_25_50_75_95": q(cos_pos), "cos_neg_q05_25_50_75_95": q(cos_neg),
                     "cos_pos_mean": float(cos_pos.mean()), "cos_neg_mean": float(cos_neg.mean())}
    if crit is not None and hasattr(crit, "scale") and hasattr(crit, "bias"):
        a, b = float(crit.scale), float(crit.bias)
        out["critic_ab"] = {"a": a, "b": b, "threshold_cos": (-b / a) if abs(a) > 1e-8 else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run ids under OUTPUT_ROOT")
    ap.add_argument("--output-root", default="/home/infres/yinwang/CS_QMI/outputs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    rows = []
    for r in a.runs.split(","):
        try:
            rows.append(diagnose(Path(a.output_root) / r, device)); print(f"{r}: done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{r}: FAILED {e!r}", flush=True)
    L = [f"# Geometry diagnostic (alignment / uniformity, Wang & Isola) — {utc_now()}", "",
         "Selection set, two train-distribution views (fixed RNG), frozen final checkpoints. Lower alignment = views closer; lower (more negative) uniformity = points spread more evenly on the sphere.", "",
         "| run | method | ep | critic | K | negdet | z: align | z: unif | z cos+ median | z cos− median | h: align | h: unif | h cos+ med | h cos− med | a / b / thr |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ab = r.get("critic_ab"); abs_ = f"{ab['a']:.2f} / {ab['b']:.2f} / {ab['threshold_cos']:.3f}" if ab and ab["threshold_cos"] is not None else "—"
        L.append(f"| {r['run']} | {r['method']} | {r['epochs']} | {r['critic_input']} | {r['K']} | {r['negative_detach']} | {r['z_l2']['alignment']:.3f} | {r['z_l2']['uniformity']:.3f} | "
                 f"{r['z_l2']['cos_pos_q05_25_50_75_95'][2]:.3f} | {r['z_l2']['cos_neg_q05_25_50_75_95'][2]:.3f} | {r['h_l2']['alignment']:.3f} | {r['h_l2']['uniformity']:.3f} | "
                 f"{r['h_l2']['cos_pos_q05_25_50_75_95'][2]:.3f} | {r['h_l2']['cos_neg_q05_25_50_75_95'][2]:.3f} | {abs_} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    atomic_write_json(Path(a.out).with_suffix(".json"), rows)
    print(f"{len(rows)} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
