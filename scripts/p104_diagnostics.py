"""P104 post-hoc diagnostics (package v3 §7.4, §9.1, §9.2) on saved checkpoints; read-only, never touches the official test set.

    python scripts/p104_diagnostics.py --run-dir outputs/P104_G1_views4_800ep_seed0 [--checkpoints initial.pt,epoch_020.pt,...] --out X.json

Per checkpoint (fresh model copy; the run directory is never written):
  §9.1 critic      a, b, zero-score threshold −b/a (a ≠ 0), positive / negative cosine mean + quantiles, T distribution, clean gate
                   1 − ½(E_P T² + E_Q T²) (directly from T², not 1 − J), regression residuals 1 − T on P and −1 − T on Q.
                   Selection images, two train-distribution views, fixed RNG, all modules in eval mode; negatives = a fixed derangement.
  §9.1 geometry    for h, p_raw and z separately: mean-vector norm, centred covariance spectrum (top 16, normalised) and effective rank
                   (exp entropy), alignment and uniformity (Wang & Isola, on the L2-normalised features).
  §9.2 gradients   one fixed batch of B fit images x V training views (fixed augmentation seed) and one fixed shift draw; the loss is
                   the run's own clean training loss −J (cross-view K pairs or all view tokens, with the run's negative routing); a
                   noisy critic is scored clean here.  G_pos / G_neg = gradients (of the minimised loss) of the P and Q parts w.r.t.
                   encoder + projector parameters: norms, inner product, cosine, total norm; input-gradient norms per view (images and
                   z); aggregate / sum-of-per-anchor-norms ratio ||Σ_i g_i|| / Σ_i ||g_i|| over the first 32 anchors (base images).
                   BN runs in training mode on a throw-away copy (as in a training step); buffers are discarded.
  §7.4 noise       noisy critics only: same fixed batch, z detached, same shifts; S noise draws of the R = 1 and R = 4 losses; online
                   Welford variance of dL/dz and of the loss; per-draw time.  Conditional (noise-only) variance, not training variance.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import cyclic_negative_indices  # noqa: E402
from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import load_resolved  # noqa: E402
from vcs_ssl.data.cifar import load_cifar10_train  # noqa: E402
from vcs_ssl.data.datasets import SSLMultiViewDataset, TwoViewNoLabelEvalDataset, make_eval_loader  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.data.transforms import build_two_view_transform  # noqa: E402
from vcs_ssl.models import build_models  # noqa: E402
from vcs_ssl.objectives import noisy_pair_loss_repeats  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def quant(x: torch.Tensor) -> list[float]:
    x = x.detach().float().flatten().cpu()
    if len(x) > 1_000_000:
        x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:1_000_000]]
    return [round(float(v), 5) for v in torch.quantile(x, torch.tensor(QS))]


def eff_rank(ev: torch.Tensor) -> float:
    p = ev.clamp_min(0) / ev.clamp_min(0).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def uniformity(z: torch.Tensor, t: float = 2.0, max_n: int = 5000) -> float:
    z = z[:max_n]
    d2 = torch.cdist(z, z).pow(2)
    mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    return float(torch.log(torch.exp(-t * d2[mask]).mean()))


def load(run_dir: Path, ck_name: str, device: torch.device):
    cfg = load_resolved(run_dir / "config.resolved.yaml")
    ck = load_checkpoint(run_dir / "checkpoints" / ck_name)
    built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"])
    if crit is not None and ck.get("critic_state") is not None:
        crit.load_state_dict(ck["critic_state"])
    return cfg, ck, enc, proj, crit


def clean_scores(crit, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Clean T = tanh(a<l, r> + b) for any cosine-family critic (the noisy critic in eval mode is clean)."""
    was = crit.training
    crit.eval()
    try:
        return crit(left, right)
    finally:
        crit.train(was)


@torch.no_grad()
def critic_and_geometry(cfg, enc, proj, crit, data, sel: np.ndarray, device, n_img: int) -> dict:
    for m in (enc, proj, crit):
        if m is not None:
            m.eval()
    tf = build_two_view_transform(cfg["views"])
    loader = make_eval_loader(TwoViewNoLabelEvalDataset(data.data, sel[:n_img], tf), batch_size=500, num_workers=4,
                              generator=torch.Generator().manual_seed(777), pin_memory=device.type == "cuda")
    eps = cfg["model"]["normalization"]["eps"]
    acc = {k: ([], []) for k in ("h", "p_raw", "z")}
    for x1, x2, _ in loader:
        h = enc(torch.cat((x1, x2)).to(device)); p = proj(h); z = F.normalize(p, dim=1, eps=eps)
        for k, v in (("h", h), ("p_raw", p), ("z", z)):
            a, b = v.chunk(2); acc[k][0].append(a); acc[k][1].append(b)
    feats = {k: (torch.cat(a), torch.cat(b)) for k, (a, b) in acc.items()}
    n = len(feats["z"][0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1))
    perm = torch.where(perm == torch.arange(n), (perm + 1) % n, perm).to(device)
    out: dict = {"n_images": n}
    geo = {}
    for k, (A, B) in feats.items():
        X = A.float()
        mu = X.mean(0)
        C = torch.cov((X - mu).T)
        ev = torch.linalg.eigvalsh(C).flip(0)
        An, Bn = F.normalize(A.float(), dim=1), F.normalize(B.float(), dim=1)
        geo[k] = {"dim": int(X.shape[1]), "mean_vector_norm": float(mu.norm()), "mean_feature_norm": float(X.norm(dim=1).mean()),
                  "spectrum_top16_normalised": [round(float(v), 6) for v in (ev[:16] / ev.clamp_min(0).sum())],
                  "effective_rank": eff_rank(ev), "alignment": float((An - Bn).pow(2).sum(-1).mean()), "uniformity": uniformity(An)}
    out["geometry"] = geo
    z1, z2 = feats["z"]
    cp, cn = (z1 * z2).sum(-1), (z1 * z2[perm]).sum(-1)
    out["cos_pos"] = {"mean": float(cp.mean()), "q05_25_50_75_95": quant(cp)}
    out["cos_neg"] = {"mean": float(cn.mean()), "q05_25_50_75_95": quant(cn)}
    if crit is not None and hasattr(crit, "scale"):
        a, b = float(crit.scale), float(crit.bias)
        tp, tn = clean_scores(crit, z1, z2), clean_scores(crit, z1, z2[perm])
        out["critic"] = {"impl": type(crit).__name__, "a": a, "b": b, "zero_score_threshold": (-b / a) if abs(a) > 1e-8 else None,
                         "trainable_affine": bool(getattr(crit, "trainable_affine", True)),
                         "T_pos": {"mean": float(tp.mean()), "q": quant(tp), "sat_frac": float((tp.abs() > 0.95).float().mean())},
                         "T_neg": {"mean": float(tn.mean()), "q": quant(tn), "sat_frac": float((tn.abs() > 0.95).float().mean())},
                         "gate_clean": float(1.0 - 0.5 * (tp.square().mean() + tn.square().mean())),
                         "J_clean": float(tp.mean() - tn.mean() - 0.5 * tp.square().mean() - 0.5 * tn.square().mean()),
                         "residual_pos": {"mean": float((1 - tp).mean()), "rms": float((1 - tp).square().mean().sqrt())},
                         "residual_neg": {"mean": float((-1 - tn).mean()), "rms": float((-1 - tn).square().mean().sqrt())}}
    return out


def fixed_batch(cfg, data, uids: np.ndarray, B: int, V: int, seed: int = 4242) -> list[torch.Tensor]:
    ds = SSLMultiViewDataset(data.data, uids[:B], build_two_view_transform(cfg["views"]), V)
    torch.manual_seed(seed)  # torchvision transforms draw from the global generator; num_workers = 0 keeps it deterministic
    items = [ds[i] for i in range(B)]
    return [torch.stack([it[v] for it in items]) for v in range(V)]


def term_tensors(cfg, crit, vz: list[torch.Tensor], pair_seed: int = 99):
    """Elementwise clean loss contributions: returns (c_pos, anchor_pos, c_neg, anchor_neg) with L = c_pos.sum() + c_neg.sum()
    equal to the run's clean training loss −J (mean over view pairs for cross_view_k; global counts for all_view_tokens)."""
    V, B = len(vz), vz[0].shape[0]
    nd = cfg["pairing"]["negative_detach"]
    if cfg["pairing"].get("pair_scope", "cross_view_k") == "all_view_tokens":
        flat = torch.cat(vz); n = flat.shape[0]
        ids = torch.arange(n, device=flat.device) % B
        same = ids[:, None] == ids[None, :]; diag = torch.eye(n, dtype=torch.bool, device=flat.device)
        keys = flat.detach() if nd else flat
        tp = crit.score_matrix(flat @ flat.T)[same & ~diag]
        tq = crit.score_matrix(flat @ keys.T)[~same]
        rows = ids[:, None].expand(n, n)
        ap, aq = rows[same & ~diag], rows[~same]
        return (-tp + 0.5 * tp.square()) / len(tp), ap, (tq + 0.5 * tq.square()) / len(tq), aq
    k = int(cfg["pairing"]["k"]); g = torch.Generator().manual_seed(pair_seed)
    pairs = [(a, b) for a in range(V) for b in range(a + 1, V)]
    cps, aps, cqs, aqs = [], [], [], []
    ar = torch.arange(B, device=vz[0].device)
    for a, b in pairs:
        idx, _ = cyclic_negative_indices(B, k, generator=g, device=vz[0].device)
        z1, z2 = vz[a], vz[b]
        tp = crit(z1, z2)
        left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1])
        right = (z2.detach() if nd else z2)[idx].reshape(-1, z2.shape[1])
        tq = crit(left, right)
        cps.append((-tp + 0.5 * tp.square()) / (B * len(pairs))); aps.append(ar)
        cqs.append((tq + 0.5 * tq.square()) / (B * k * len(pairs))); aqs.append(ar.repeat(k))
    return torch.cat(cps), torch.cat(aps), torch.cat(cqs), torch.cat(aqs)


def flat_grad(loss, params, retain=True) -> torch.Tensor:
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])


def gradient_organisation(cfg, enc, proj, crit, views: list[torch.Tensor], device, n_anchors: int) -> dict:
    enc, proj, crit = copy.deepcopy(enc), copy.deepcopy(proj), copy.deepcopy(crit)
    enc.train(); proj.train(); crit.eval()  # BN as in training (copy); critic clean
    eps = cfg["model"]["normalization"]["eps"]
    xs = [v.to(device).requires_grad_(True) for v in views]
    h = enc(torch.cat(xs)); p = proj(h); z = F.normalize(p, dim=1, eps=eps)
    vz = list(z.chunk(len(xs)))
    for t in vz:
        t.retain_grad()
    cp, ap, cq, aq = term_tensors(cfg, crit, vz)
    params = [q for m in (enc, proj) for q in m.parameters() if q.requires_grad]
    Lp, Lq = cp.sum(), cq.sum()
    gp, gq = flat_grad(Lp, params), flat_grad(Lq, params)
    out = {"loss": float(Lp + Lq), "J_clean_batch": float(-(Lp + Lq)), "G_pos_norm": float(gp.norm()), "G_neg_norm": float(gq.norm()),
           "G_pos_dot_G_neg": float(gp @ gq), "cos_G_pos_G_neg": float(gp @ gq / (gp.norm() * gq.norm()).clamp_min(1e-30)),
           "G_total_norm": float((gp + gq).norm()), "bn_mode": "train (throw-away copy)", "critic_scoring": "clean"}
    cparams = [q for q in crit.parameters() if q.requires_grad]
    if cparams:
        out["critic_grad_pos_norm"] = float(flat_grad(Lp, cparams).norm()); out["critic_grad_neg_norm"] = float(flat_grad(Lq, cparams).norm())
    (Lp + Lq).backward(retain_graph=True)
    out["input_grad_norm_per_view"] = [float(x.grad.norm()) for x in xs]
    out["z_grad_norm_per_view"] = [float(t.grad.norm()) for t in vz]
    # aggregate / sum of per-anchor norms (first n_anchors base images; anchor = the left element's base image)
    gi, tot = [], None
    for i in range(n_anchors):
        li = cp[ap == i].sum() + cq[aq == i].sum()
        g = flat_grad(li, params)
        gi.append(float(g.norm())); tot = g if tot is None else tot + g
    out["anchors"] = n_anchors
    out["per_anchor_grad_norm_mean"] = float(np.mean(gi))
    out["aggregate_over_sum_of_norms"] = float(tot.norm()) / max(sum(gi), 1e-30)
    return out


def noise_conditional_variance(cfg, enc, proj, crit, views, device, draws: int) -> dict:
    """§7.4: fixed z (detached leaf) and shifts; S draws of the R-average loss; Welford over dL/dz."""
    if not getattr(crit, "is_noisy", False):
        return {}
    enc, proj, crit = copy.deepcopy(enc).eval(), copy.deepcopy(proj).eval(), copy.deepcopy(crit)
    eps = cfg["model"]["normalization"]["eps"]
    with torch.no_grad():
        z = F.normalize(proj(enc(torch.cat([v.to(device) for v in views]))), dim=1, eps=eps)
    vz0 = [t.contiguous() for t in z.chunk(len(views))]
    k, nd = int(cfg["pairing"]["k"]), cfg["pairing"]["negative_detach"]
    js = cfg["objective"]["loss"] == "js_matched_logistic"
    res = {}
    crit.train()
    for R in (1, 4):
        mean = m2 = None; lm = l2 = 0.0; secs = []
        for s in range(draws):
            vz = [t.clone().requires_grad_(True) for t in vz0]
            g = torch.Generator().manual_seed(123)  # same shifts every draw
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            L = torch.stack([noisy_pair_loss_repeats(vz[a], vz[b], crit, k=k, generator=g, negative_detach=nd, repeats=R, objective="js" if js else "vcs")[0]["loss"]
                             for a in range(len(vz)) for b in range(a + 1, len(vz))]).mean()
            gz = torch.cat(torch.autograd.grad(L, vz)).flatten()
            if device.type == "cuda":
                torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
            n = s + 1
            if mean is None:
                mean, m2 = gz.clone(), torch.zeros_like(gz)
            else:
                d = gz - mean; mean += d / n; m2 += d * (gz - mean)
            dl = float(L) - lm; lm += dl / n; l2 += dl * (float(L) - lm)
        var = m2 / (draws - 1)
        res[f"R{R}"] = {"draws": draws, "grad_z_var_sum": float(var.sum()), "grad_z_mean_sqnorm": float(mean.square().sum()),
                        "loss_mean": lm, "loss_var": l2 / (draws - 1), "seconds_per_draw_median": float(np.median(secs))}
    res["grad_var_ratio_R1_over_R4"] = res["R1"]["grad_z_var_sum"] / max(res["R4"]["grad_z_var_sum"], 1e-30)
    res["note"] = "conditional on images, augmentations and shifts (noise only); not the full training-gradient variance"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoints", default="initial.pt,epoch_020.pt,epoch_100.pt,epoch_400.pt,epoch_800.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-images", type=int, default=5000, help="selection images for §9.1")
    ap.add_argument("--batch", type=int, default=None, help="§9.2 batch (default: the run's batch)")
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--noise-draws", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    device = torch.device("cpu") if a.cpu or not torch.cuda.is_available() else torch.device("cuda", 0)
    run_dir = Path(a.run_dir)
    cfg0 = load_resolved(run_dir / "config.resolved.yaml")
    man = load_manifest(run_dir / "manifest.json")
    data = load_cifar10_train(cfg0["data"]["root"])  # official TRAIN file only (fit / selection)
    sel, fit = np.asarray(man["selection_uids"]), np.asarray(man["fit_uids"])
    V = int(cfg0["views"].get("count", 2)); B = int(a.batch or cfg0["train"]["batch_size_images"])
    views = fixed_batch(cfg0, data, fit, B, V)
    rows = {"run": run_dir.name, "created_utc": utc_now(), "device": str(device), "views": V, "batch": B,
            "pair_scope": cfg0["pairing"].get("pair_scope", "cross_view_k"), "negative_detach": cfg0["pairing"]["negative_detach"], "checkpoints": {}}
    for name in [c for c in a.checkpoints.split(",") if c]:
        if not (run_dir / "checkpoints" / name).is_file():
            rows["checkpoints"][name] = {"missing": True}; continue
        t0 = time.time()
        cfg, ck, enc, proj, crit = load(run_dir, name, device)
        r = {"epoch": ck.get("epoch"), "step": ck.get("step")}
        r.update(critic_and_geometry(cfg, enc, proj, crit, data, sel, device, a.n_images))
        if crit is not None and hasattr(crit, "scale"):
            r["gradients"] = gradient_organisation(cfg, enc, proj, crit, views, device, min(a.anchors, B))
            r["noise_conditional_variance"] = noise_conditional_variance(cfg, enc, proj, crit, views, device, a.noise_draws)
        r["seconds"] = time.time() - t0
        rows["checkpoints"][name] = r
        print(f"{name}: done in {r['seconds']:.0f}s", flush=True)
        del enc, proj, crit
    atomic_write_json(Path(a.out), rows)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
