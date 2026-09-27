"""Second-application task-level pre-check B-T1 (wave 2): the closed-form VCS J* as an UNLABELLED embedding probe — layer correspondence.

    python scripts/task_b_layer_probe.py --out <prefix> [--runs A:ckpt,B:ckpt,C:ckpt] [--pca 64] [--smoke]

Task: given two networks, identify which stage of network B corresponds to each stage of network A from the representations alone (the
CKA sanity test of Kornblith et al. 2019).  Measure under test: the closed-form VCS J* between two representations in a BILINEAR class —
each side is PCA-whitened to p = 64 components (fitted on the fit part), φ(u, v) = vec(u vᵀ) (+ intercept, 4097-d), positives (u_i, v_i) of the
same image, product samples = K = 8 cyclic shifts of v, w* = ½(A_M + λI)⁻¹d; λ ∈ {1e-4, 1e-2, 1} × mean diag(A_M) and the tanh scalar c are
chosen on a validation quarter of the fit part; the reported J is the tanh-wrapped closed form on the eval part (the raw J* is kept too).
Competitors on the same eval images: linear CKA and RBF-kernel CKA (median-heuristic bandwidth), both on the raw stage features.
Stages per network (clean transform, 5 000 selection images, 2 500 fit / 2 500 eval): stem (conv1-bn-relu, avg-pooled, 64), layer1 (64),
layer2 (128), layer3 (256), h (= avg-pooled layer4, 512), z (projector output, L2, 128).  Network pairs: VCS seed 0 vs VCS seed 1 (same recipe),
VCS seed 0 vs SimCLR seed 0.  Identification = for each stage of A, argmax over B's stages equals the same stage (row-wise; column-wise also
reported).  Invariance checks on the (h, h) pair of seed 0 vs seed 1: random orthogonal rotation, isotropic scaling ×10, per-feature scaling
(log-uniform in [0.1, 10]) applied to B's raw features — change of every measure (for J* the PCA-whitening is refitted: it is part of the measure).
Nothing is trained (linear solves, PCA, c).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import load_resolved
from vcs_ssl.data.cifar import load_cifar10_train
from vcs_ssl.data.datasets import LabeledCleanDataset, make_eval_loader
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.data.transforms import build_clean_transform
from vcs_ssl.models import build_models
from vcs_ssl.utils import atomic_write_json, utc_now

STAGES = ("stem", "layer1", "layer2", "layer3", "h", "z")
DEFAULT_RUNS = "P35_vcs_a5_views4_800ep_seed0:epoch_800.pt,P35_vcs_a5_views4_800ep_seed1:epoch_800.pt,P5_simclr_seed0:epoch_200.pt"


@torch.no_grad()
def extract_stages(run_dir: Path, ckpt: str, uids, device, workers=4):
    cfg = load_resolved(run_dir / "config.resolved.yaml"); ck = load_checkpoint(run_dir / "checkpoints" / ckpt)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device); enc, proj = built["encoder"], built["projector"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"]); enc.eval(); proj.eval()
    feats: dict[str, list] = {}
    def pooled(name):
        def fn(_m, _i, out):
            feats.setdefault(name, []).append(torch.flatten(F.adaptive_avg_pool2d(out, 1), 1).float().cpu())
        return fn
    hooks = [enc.relu.register_forward_hook(pooled("stem"))] + [getattr(enc, f"layer{i}").register_forward_hook(pooled(f"layer{i}")) for i in (1, 2, 3)]
    data = load_cifar10_train(cfg["data"]["root"]); tf = build_clean_transform(cfg["views"])
    loader = make_eval_loader(LabeledCleanDataset(data.data, data.targets, uids, tf), batch_size=500, num_workers=workers, generator=torch.Generator().manual_seed(5), pin_memory=False)
    for x, _, _ in loader:
        h = enc(x.to(device)); feats.setdefault("h", []).append(h.float().cpu()); feats.setdefault("z", []).append(F.normalize(proj(h), dim=1).float().cpu())
    for hk in hooks:
        hk.remove()
    return cfg["run"]["method"], {k: torch.cat(v) for k, v in feats.items()}


def pca_whiten(fit, p):
    """Mean and whitened top-p principal directions fitted on `fit` [n,d]; returns a projector f(X) -> [n,p] with unit-variance components."""
    mu = fit.mean(0, keepdim=True); X = (fit - mu).double(); U, Sv, Vh = torch.linalg.svd(X, full_matrices=False)
    p = min(p, Vh.shape[0]); W = Vh[:p].T / (Sv[:p] / (len(X) - 1) ** 0.5).clamp_min(1e-8)
    return lambda Y: ((Y - mu).double() @ W).float()


def bilinear(u, v):
    return (u[:, :, None] * v[:, None, :]).reshape(len(u), -1)


def with_intercept(phi):
    return torch.cat((phi, torch.ones(len(phi), 1, dtype=phi.dtype)), 1)


def J_of(tp, tn):
    return float((tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean())


def phi_sets(u, v, K, gen):
    n = len(u); ks = torch.randperm(n - 1, generator=gen)[:K] + 1
    idx = torch.cat([(torch.arange(n) + int(k)) % n for k in ks])
    return with_intercept(bilinear(u, v)).double(), with_intercept(bilinear(u.repeat(K, 1), v[idx])).double()


def moments(pp, pq, chunk=4096):
    d = pp.mean(0) - pq.mean(0); D = pp.shape[1]; A = torch.zeros(D, D, dtype=torch.float64)
    for M, w in ((pp, 0.5 / len(pp)), (pq, 0.5 / len(pq))):
        for s in range(0, len(M), chunk):
            c = M[s: s + chunk]; A += w * (c.T @ c)
    return d, A


def closed_form_J(u_fit, v_fit, u_val, v_val, u_ev, v_ev, K, ridges, gen_seed=3):
    pf, qf = phi_sets(u_fit, v_fit, K, torch.Generator().manual_seed(gen_seed)); pv, qv = phi_sets(u_val, v_val, K, torch.Generator().manual_seed(gen_seed + 1)); pe, qe = phi_sets(u_ev, v_ev, K, torch.Generator().manual_seed(gen_seed + 2))
    d, A = moments(pf, qf); scale = float(torch.diag(A).mean()); cs = torch.logspace(-1, 1.5, 40, dtype=torch.float64); best = None
    for lam in ridges:
        w = 0.5 * torch.linalg.solve(A + lam * scale * torch.eye(len(A), dtype=torch.float64), d); tp, tn = pv @ w, qv @ w
        Jc = [J_of(torch.tanh(c * tp), torch.tanh(c * tn)) for c in cs]; i = int(np.argmax(Jc))
        if best is None or Jc[i] > best["J_val_tanh"]:
            best = {"lam": lam, "c": float(cs[i]), "J_val_tanh": Jc[i], "w": w}
    w, c = best["w"], best["c"]; tp, tn = pe @ w, qe @ w
    return {"J_tanh": J_of(torch.tanh(c * tp), torch.tanh(c * tn)), "J_closed": J_of(tp, tn), "lam": best["lam"], "c": c}


def cka_linear(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True); X, Y = X.double(), Y.double()
    return float((Y.T @ X).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def cka_rbf(X, Y):
    def gram(Z):
        Z = Z.double(); d2 = torch.cdist(Z, Z) ** 2; med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0); Kk = torch.exp(-d2 / (2 * med))
        n = len(Kk); H = torch.eye(n, dtype=torch.float64) - 1.0 / n; return H @ Kk @ H
    Kx, Ky = gram(X), gram(Y)
    return float((Kx * Ky).sum() / ((Kx * Kx).sum().sqrt() * (Ky * Ky).sum().sqrt()))


def measures(FA, FB, fit_ix, val_ix, ev_ix, p, K, ridges):
    """All three measures between representation FA [n,dA] and FB [n,dB] on the common split."""
    pa, pb = pca_whiten(FA[fit_ix], p), pca_whiten(FB[fit_ix], p); UA, UB = pa(FA), pb(FB)
    cf = closed_form_J(UA[fit_ix], UB[fit_ix], UA[val_ix], UB[val_ix], UA[ev_ix], UB[ev_ix], K, ridges)
    return {"vcs_tanh": cf["J_tanh"], "vcs_closed": cf["J_closed"], "lam": cf["lam"], "c": cf["c"], "cka_linear": cka_linear(FA[ev_ix], FB[ev_ix]), "cka_rbf": cka_rbf(FA[ev_ix], FB[ev_ix])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="run:ckpt items (',' or ';'); the first is network A of both pairs, the second and third are the two B networks")
    ap.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "/home/infres/yinwang/CS_QMI/outputs")); ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=64); ap.add_argument("--K", type=int, default=8); ap.add_argument("--ridge", default="1e-4,1e-2,1"); ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--n-images", type=int, default=5000); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    if a.smoke:
        a.pca, a.K, a.n_images = 16, 4, 600
    ridges = [float(v) for v in a.ridge.split(",")]
    items = [it.strip().split(":") for it in a.runs.replace(";", ",").split(",") if it.strip()]
    man = load_manifest(Path(a.output_root) / items[0][0] / "manifest.json"); sel = np.asarray(man["selection_uids"], dtype=np.int64)
    rng = np.random.default_rng(a.seed); sel = np.sort(rng.permutation(sel)[: a.n_images]); n = len(sel)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed)); half = n // 2; fit_all = perm[:half]; ev_ix = perm[half:]; nv = half // 4; val_ix, fit_ix = fit_all[:nv], fit_all[nv:]
    t0 = time.time(); nets = {}
    for run, ckpt in items:
        method, Fs = extract_stages(Path(a.output_root) / run, ckpt, sel, device, a.workers); nets[run] = (method, Fs)
        print(f"[{run}] {method}: " + " ".join(f"{k}{tuple(v.shape)}" for k, v in Fs.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    runA = items[0][0]; out = {"settings": vars(a), "device": str(device), "utc": utc_now(), "n_images": int(n), "split": {"fit": int(len(fit_ix)), "val": int(len(val_ix)), "eval": int(len(ev_ix))}, "pairs": {}}
    for runB, _ in items[1:]:
        FA, FB = nets[runA][1], nets[runB][1]; M = {m: np.zeros((len(STAGES), len(STAGES))) for m in ("vcs_tanh", "vcs_closed", "cka_linear", "cka_rbf")}; detail = {}
        for i, sa in enumerate(STAGES):
            for j, sb in enumerate(STAGES):
                r = measures(FA[sa], FB[sb], fit_ix, val_ix, ev_ix, a.pca, a.K, ridges); detail[f"{sa}|{sb}"] = r
                for m in M:
                    M[m][i, j] = r[m]
            print(f"[{runA} vs {runB}] {sa}: " + " ".join(f"{sb}: J {M['vcs_tanh'][i, j]:.3f} cka {M['cka_linear'][i, j]:.3f}/{M['cka_rbf'][i, j]:.3f}" for j, sb in enumerate(STAGES)) + f" ({time.time() - t0:.0f}s)", flush=True)
        ident = {m: {"row_wise": int((M[m].argmax(1) == np.arange(len(STAGES))).sum()), "col_wise": int((M[m].argmax(0) == np.arange(len(STAGES))).sum())} for m in M}
        out["pairs"][f"{runA}|{runB}"] = {"methods": (nets[runA][0], nets[runB][0]), "matrices": {m: M[m].tolist() for m in M}, "identification_of_6": ident, "detail": detail}
        print(f"[{runA} vs {runB}] identification (row/col of {len(STAGES)}): " + " ".join(f"{m} {v['row_wise']}/{v['col_wise']}" for m, v in ident.items()), flush=True)
    # invariance checks on (h, h) of the first pair
    runB = items[1][0]; FA, FB = nets[runA][1]["h"], nets[runB][1]["h"]; g = torch.Generator().manual_seed(a.seed + 1)
    Q, _ = torch.linalg.qr(torch.randn(FB.shape[1], FB.shape[1], generator=g)); scales = torch.exp(torch.empty(FB.shape[1]).uniform_(np.log(0.1), np.log(10), generator=g))
    base = measures(FA, FB, fit_ix, val_ix, ev_ix, a.pca, a.K, ridges); inv = {"base": base}
    for name, FBt in (("orthogonal", FB @ Q), ("isotropic_x10", FB * 10.0), ("per_feature_scale", FB * scales)):
        r = measures(FA, FBt, fit_ix, val_ix, ev_ix, a.pca, a.K, ridges); inv[name] = {**r, "delta": {m: r[m] - base[m] for m in ("vcs_tanh", "vcs_closed", "cka_linear", "cka_rbf")}}
        print(f"[invariance] {name}: " + " ".join(f"{m} {r[m]:.4f} (Δ {inv[name]['delta'][m]:+.1e})" for m in ("vcs_tanh", "cka_linear", "cka_rbf")), flush=True)
    out["invariance_h_pair"] = inv
    atomic_write_json(Path(a.out + ".json"), out)
    L = [f"# Task pre-check B-T1 — closed-form VCS J* (bilinear class, PCA-{a.pca} whitened) as an unlabelled embedding probe: layer correspondence vs CKA — {out['utc']}", "",
         f"{n} selection images: fit {len(fit_ix)} / val {len(val_ix)} / eval {len(ev_ix)}; K = {a.K} cyclic-shift product samples; λ ∈ {ridges} × mean diag and c chosen on val.  Rows = stages of network A, columns = stages of network B; bold = row-wise argmax.", ""]
    for key, pr in out["pairs"].items():
        L += [f"## {key} ({pr['methods'][0]} vs {pr['methods'][1]}) — identification (row-wise / column-wise of {len(STAGES)}): " + ", ".join(f"{m} {v['row_wise']} / {v['col_wise']}" for m, v in pr["identification_of_6"].items()), ""]
        for m in ("vcs_tanh", "cka_linear", "cka_rbf"):
            Mm = np.array(pr["matrices"][m]); L += [f"**{m}**", "", "| A \\ B | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
            for i, sa in enumerate(STAGES):
                j_ = int(Mm[i].argmax()); L.append(f"| {sa} | " + " | ".join((f"**{Mm[i, j]:.3f}**" if j == j_ else f"{Mm[i, j]:.3f}") for j in range(len(STAGES))) + " |")
            L.append("")
    L += ["## Invariance checks on (h, h) of the first pair (B's features transformed)", "", "| transform | vcs_tanh (Δ) | vcs_closed (Δ) | cka_linear (Δ) | cka_rbf (Δ) |", "|---|---|---|---|---|"]
    for name in ("orthogonal", "isotropic_x10", "per_feature_scale"):
        r = inv[name]; L.append(f"| {name} | " + " | ".join(f"{r[m]:.4f} ({r['delta'][m]:+.1e})" for m in ("vcs_tanh", "vcs_closed", "cka_linear", "cka_rbf")) + " |")
    L.append(f"| base | " + " | ".join(f"{base[m]:.4f}" for m in ("vcs_tanh", "vcs_closed", "cka_linear", "cka_rbf")) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
