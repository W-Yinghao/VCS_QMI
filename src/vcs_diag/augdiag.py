"""P115 diagnostics core (v5 §3 "对应诊断"): read-only measurements on a trained checkpoint, on fixed base images with an independent
diagnostic RNG.  Nothing here changes, filters or reweights training pairs; the run's own objective is only re-evaluated.

Recorded per checkpoint:
  * distributions of positive / negative s, T, residuals (1 − T on P, −1 − T on Q) and 1 − T² (VCS); s only for SimCLR;
  * full-loss gradient norms of the P part and the Q part w.r.t. z, h, encoder and projector parameters (the run's training loss:
    VCS −J with its pairing scope and negative routing; SimCLR NT-Xent split as  L = mean(−logit_pos) + mean(logsumexp));
  * positive pairs grouped by the TRUE crop overlap of the two views (IoU and intersection area / image area, from the recorded
    RandomResizedCrop boxes) — each group's share of the total positive-term encoder gradient (dot-product shares, sum to 1);
  * multi-view gradient organisation: pairwise cosines of the per-view-pair positive gradients and ||Σ g|| / Σ ||g||; per-token
    cancellation of the positive pulls (z-space), split by the token's lowest-IoU partner;
  * geometry of h (and z): norm, top-16 normalised spectrum, effective rank — on clean held-out images (diagnostic only).
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

QS = (0.05, 0.25, 0.5, 0.75, 0.95)
IOU_BINS = (0.0, 0.1, 0.25, 0.5, 1.0000001)       # fixed before any run is read
AREA_BINS = (0.0, 0.1, 0.25, 0.5, 1.0000001)       # intersection area as a fraction of the 32 x 32 image


# ------------------------------------------------------------------------------------------------------------- crop geometry
def crop_overlap(b1, b2, img_hw=(32, 32)) -> tuple[float, float]:
    """Boxes (top, left, height, width) in original-image pixels -> (IoU, intersection area / image area).  A horizontal flip
    after the crop does not change which original pixels a view covers, so it does not enter the overlap."""
    t1, l1, h1, w1 = b1; t2, l2, h2, w2 = b2
    ih = max(0, min(t1 + h1, t2 + h2) - max(t1, t2)); iw = max(0, min(l1 + w1, l2 + w2) - max(l1, l2))
    inter = ih * iw; union = h1 * w1 + h2 * w2 - inter
    return (inter / union if union > 0 else 0.0), inter / float(img_hw[0] * img_hw[1])


def quant(x: Tensor) -> list[float]:
    x = x.detach().float().flatten().cpu()
    if x.numel() == 0:
        return []
    if x.numel() > 1_000_000:
        x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]]
    return [round(float(v), 5) for v in torch.quantile(x, torch.tensor(QS))]


def summary(x: Tensor) -> dict[str, Any]:
    x = x.detach().float().flatten()
    return {"n": int(x.numel()), "mean": float(x.mean()) if x.numel() else None, "sd": float(x.std()) if x.numel() > 1 else None, "q05_25_50_75_95": quant(x)}


def eff_rank(ev: Tensor) -> float:
    p = ev.clamp_min(0) / ev.clamp_min(0).sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


# ------------------------------------------------------------------------------------------------------------- views with known boxes
def known_box_views(images_uint8: np.ndarray, uids: np.ndarray, two_view_tf: T.Compose, n_views: int, seed: int):
    """The run's configured augmentation with the RandomResizedCrop box drawn by `get_params` and recorded.  The rest of the pipeline is
    the run's own transform objects (flip, colour jitter, grayscale, optional blur, normalisation).  The global torch RNG is forked and
    seeded so the diagnostic RNG is independent of (and does not disturb) anything else; identical (uids, seed) -> identical views."""
    rrc = two_view_tf.transforms[0]
    assert isinstance(rrc, T.RandomResizedCrop), "first op of the training transform must be RandomResizedCrop"
    rest = T.Compose(two_view_tf.transforms[1:])
    views = [[] for _ in range(n_views)]; boxes = np.zeros((len(uids), n_views, 4), dtype=np.int64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for i, uid in enumerate(uids):
            img = Image.fromarray(images_uint8[int(uid)])
            for v in range(n_views):
                t, l, h, w = rrc.get_params(img, rrc.scale, rrc.ratio)
                x = TF.resized_crop(img, t, l, h, w, rrc.size, rrc.interpolation, antialias=rrc.antialias)
                views[v].append(rest(x)); boxes[i, v] = (t, l, h, w)
    return [torch.stack(v) for v in views], boxes


# ------------------------------------------------------------------------------------------------------------- objective terms
def objective_terms(cfg: dict, crit, vz: list[Tensor], *, pair_seed: int) -> dict[str, Any]:
    """Elementwise decomposition of the run's training loss on views vz (each [B, D], L2-normalised z).
    Returns Lp, Lq (scalars with graph, Lp + Lq = training loss), per-positive-element contributions c_pos with (img, va, vb),
    detached s / T arrays for P and Q, and the coefficient needed for the analytic z-space pulls."""
    method = cfg["run"]["method"]; V, B = len(vz), vz[0].shape[0]; dev = vz[0].device
    if method == "simclr_matched":
        tau = float(cfg["objective"]["simclr_temperature"]); pairs = [(a, b) for a in range(V) for b in range(a + 1, V)]
        Lp = Lq = vz[0].new_zeros(()); cpos, img, va, vb, sp, sq = [], [], [], [], [], []
        for a, b in pairs:
            x = F.normalize(torch.cat((vz[a], vz[b])), dim=-1, eps=1e-8)      # as reference.simclr_nt_xent (second normalise kept)
            logits = x @ x.T / tau; n = 2 * B
            diag = torch.eye(n, dtype=torch.bool, device=dev)
            tgt = (torch.arange(n, device=dev) + B) % n
            pos_logit = logits[torch.arange(n, device=dev), tgt]
            lse = torch.logsumexp(logits.masked_fill(diag, -torch.inf), dim=1)
            Lp = Lp + (-pos_logit).mean() / len(pairs); Lq = Lq + lse.mean() / len(pairs)
            cpos.append(-pos_logit / (n * len(pairs)))
            ar = torch.arange(B, device=dev)
            img.append(torch.cat((ar, ar))); va.append(torch.full((n,), a, device=dev).masked_fill(torch.arange(n, device=dev) >= B, b))
            vb.append(torch.full((n,), b, device=dev).masked_fill(torch.arange(n, device=dev) >= B, a))
            sp.append((pos_logit * tau).detach())
            negmask = ~diag; negmask[torch.arange(n, device=dev), tgt] = False
            sq.append((logits * tau)[negmask].detach())
        return {"kind": "simclr", "Lp": Lp, "Lq": Lq, "c_pos": torch.cat(cpos), "img": torch.cat(img), "va": torch.cat(va), "vb": torch.cat(vb),
                "s_pos": torch.cat(sp), "s_neg": torch.cat(sq), "T_pos": None, "T_neg": None, "tau": tau, "n_pairs": len(pairs)}
    if method != "vcs_qmi":
        raise ValueError(f"unsupported method {method!r}")
    nd = bool(cfg["pairing"]["negative_detach"]); scope = cfg["pairing"].get("pair_scope", "cross_view_k")
    if scope == "all_view_tokens":
        flat = torch.cat(vz); n = flat.shape[0]; ids = torch.arange(n, device=dev) % B; views_of = torch.arange(n, device=dev) // B
        same = ids[:, None] == ids[None, :]; diag = torch.eye(n, dtype=torch.bool, device=dev)
        keys = flat.detach() if nd else flat
        Spp = flat @ flat.T; mp = same & ~diag
        rr, cc = mp.nonzero(as_tuple=True)
        s_pos = Spp[rr, cc]; t_pos = crit.score_matrix(s_pos)
        Sq = flat @ keys.T; s_neg = Sq[~same]; t_neg = crit.score_matrix(s_neg)
        c_pos = (-t_pos + 0.5 * t_pos.square()) / len(t_pos); c_neg = (t_neg + 0.5 * t_neg.square()) / len(t_neg)
        return {"kind": "vcs", "Lp": c_pos.sum(), "Lq": c_neg.sum(), "c_pos": c_pos, "img": ids[rr], "va": views_of[rr], "vb": views_of[cc],
                "s_pos": s_pos.detach(), "s_neg": s_neg.detach(), "T_pos": t_pos.detach(), "T_neg": t_neg.detach(),
                "pos_norm": float(len(t_pos)), "scope": scope}
    from reference.ssl_core import cyclic_negative_indices  # noqa: PLC0415
    k = int(cfg["pairing"]["k"]); g = torch.Generator().manual_seed(pair_seed)
    pairs = [(a, b) for a in range(V) for b in range(a + 1, V)]; ar = torch.arange(B, device=dev)
    cps, cqs, img, va, vb, sp, sq, tp_all, tq_all = [], [], [], [], [], [], [], [], []
    for a, b in pairs:
        idx, _ = cyclic_negative_indices(B, k, generator=g, device=dev)
        z1, z2 = vz[a], vz[b]
        s_p = (z1 * z2).sum(-1); t_p = crit(z1, z2)
        left = z1.unsqueeze(0).expand(k, -1, -1).reshape(-1, z1.shape[1]); right = (z2.detach() if nd else z2)[idx].reshape(-1, z2.shape[1])
        s_q = (left * right).sum(-1); t_q = crit(left, right)
        cps.append((-t_p + 0.5 * t_p.square()) / (B * len(pairs))); cqs.append((t_q + 0.5 * t_q.square()) / (B * k * len(pairs)))
        img.append(ar); va.append(torch.full_like(ar, a)); vb.append(torch.full_like(ar, b))
        sp.append(s_p.detach()); sq.append(s_q.detach()); tp_all.append(t_p.detach()); tq_all.append(t_q.detach())
    c_pos, c_neg = torch.cat(cps), torch.cat(cqs)
    return {"kind": "vcs", "Lp": c_pos.sum(), "Lq": c_neg.sum(), "c_pos": c_pos, "img": torch.cat(img), "va": torch.cat(va), "vb": torch.cat(vb),
            "s_pos": torch.cat(sp), "s_neg": torch.cat(sq), "T_pos": torch.cat(tp_all), "T_neg": torch.cat(tq_all),
            "pos_norm": float(B * len(pairs)), "scope": scope}


def positive_pulls(terms: dict, crit, vz: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic z-space gradient of each positive element's contribution on its two endpoint tokens (exact for the cosine-family
    critics and for the SimCLR numerator incl. its second normalisation).  Returns (pull_on_a [P, D], pull_on_b [P, D], coef [P]).
    Σ over elements per token equals autograd d Lp / d z (checked in tests)."""
    flatz = torch.cat(vz).detach(); B = vz[0].shape[0]
    ta = terms["va"] * B + terms["img"]; tb = terms["vb"] * B + terms["img"]
    za, zb = flatz[ta], flatz[tb]
    if terms["kind"] == "simclr":
        coef = torch.full((len(ta),), -1.0 / (terms["tau"] * len(ta)), device=flatz.device)   # c = −s/τ / (2B·n_pairs); len(ta) = 2B·n_pairs
        s = (za * zb).sum(-1, keepdim=True)
        return coef[:, None] * (zb - s * za), coef[:, None] * (za - s * zb), coef
    a = float(crit.scale); t = terms["T_pos"]
    coef = (-1.0 + t) * (1.0 - t.square()) * a / terms["pos_norm"]      # d c / d s
    return coef[:, None] * zb, coef[:, None] * za, coef


# ------------------------------------------------------------------------------------------------------------- one batch
def flat_grad(loss: Tensor, params: list[Tensor], retain: bool = True) -> Tensor:
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs, params)])


def _cos(u: Tensor, v: Tensor) -> float:
    return float(u @ v / (u.norm() * v.norm()).clamp_min(1e-30))


def batch_diagnostics(cfg: dict, enc, proj, crit, views: list[Tensor], boxes: np.ndarray, device, *, pair_seed: int) -> dict[str, Any]:
    """One fixed batch: BN in training mode on throw-away copies (as a training step; buffers discarded), critic scored clean."""
    enc, proj = copy.deepcopy(enc).train(), copy.deepcopy(proj).train()
    if crit is not None:
        crit = copy.deepcopy(crit).eval()
    eps = cfg["model"]["normalization"]["eps"]; V, B = len(views), views[0].shape[0]
    h = enc(torch.cat([v.to(device) for v in views])); p = proj(h); z = F.normalize(p, dim=1, eps=eps)
    vz = list(z.chunk(V))
    terms = objective_terms(cfg, crit, vz, pair_seed=pair_seed)
    Lp, Lq = terms["Lp"], terms["Lq"]
    pe = [q for q in enc.parameters() if q.requires_grad]; pp = [q for q in proj.parameters() if q.requires_grad]
    gP, gQ = flat_grad(Lp, pe), flat_grad(Lq, pe); gPp, gQp = flat_grad(Lp, pp), flat_grad(Lq, pp)
    zP, zQ = torch.autograd.grad(Lp, z, retain_graph=True)[0], torch.autograd.grad(Lq, z, retain_graph=True)[0]
    hP, hQ = torch.autograd.grad(Lp, h, retain_graph=True)[0], torch.autograd.grad(Lq, h, retain_graph=True)[0]
    out: dict[str, Any] = {"loss": float((Lp + Lq).detach()), "Lp": float(Lp.detach()), "Lq": float(Lq.detach()), "bn_mode": "train (throw-away copy)", "n_pos": int(len(terms["c_pos"])),
                           "n_neg_scored": int(terms["s_neg"].numel())}
    gnorm = lambda a, b: {"P": float(a.norm()), "Q": float(b.norm()), "total": float((a + b).norm()), "cos_P_Q": _cos(a.flatten(), b.flatten())}
    out["grad"] = {"encoder": gnorm(gP, gQ), "projector": gnorm(gPp, gQp), "h": gnorm(hP, hQ), "z": gnorm(zP, zQ)}
    # ---- distributions (the run's own P and Q sets)
    dist = {"s_pos": summary(terms["s_pos"]), "s_neg": summary(terms["s_neg"])}
    if terms["T_pos"] is not None:
        tp, tq = terms["T_pos"], terms["T_neg"]
        a, b = float(crit.scale), float(crit.bias); kappa = -b / a if abs(a) > 1e-8 else None
        dist.update({"T_pos": summary(tp), "T_neg": summary(tq), "residual_pos": summary(1 - tp), "residual_neg": summary(-1 - tq),
                     "one_minus_T2_pos": summary(1 - tp.square()), "one_minus_T2_neg": summary(1 - tq.square()),
                     "critic": {"a": a, "b": b, "kappa": kappa}, "frac_pos_below_kappa": float((terms["s_pos"] < kappa).float().mean()) if kappa is not None else None,
                     "frac_neg_above_kappa": float((terms["s_neg"] > kappa).float().mean()) if kappa is not None else None})
    out["dist"] = dist
    # ---- crop overlap of each positive element
    im, va, vb = terms["img"].cpu().numpy(), terms["va"].cpu().numpy(), terms["vb"].cpu().numpy()
    ov = np.array([crop_overlap(boxes[i, x], boxes[i, y]) for i, x, y in zip(im, va, vb)])
    iou, area = torch.as_tensor(ov[:, 0], device=device), torch.as_tensor(ov[:, 1], device=device)
    gP2 = float(gP @ gP)
    groups = {}
    for name, val, edges in (("iou", iou, IOU_BINS), ("inter_area", area, AREA_BINS)):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (val >= lo) & (val < hi)
            row = {"bin": [lo, min(hi, 1.0)], "n_pos": int(m.sum()), "frac_pos": float(m.float().mean())}
            if int(m.sum()):
                g = flat_grad(terms["c_pos"][m].sum(), pe)
                row.update({"grad_norm": float(g.norm()), "share_of_P_grad": float(g @ gP) / max(gP2, 1e-30), "cos_with_P_grad": _cos(g, gP),
                            "loss_share": float((terms["c_pos"][m].sum() / Lp).detach()) if float(Lp.detach()) != 0 else None,
                            "s_pos_mean": float(terms["s_pos"][m].mean())})
                if terms["T_pos"] is not None:
                    row["T_pos_mean"] = float(terms["T_pos"][m].mean())
                    k_ = dist["critic"]["kappa"]
                    row["frac_below_kappa"] = float((terms["s_pos"][m] < k_).float().mean()) if k_ is not None else None
            rows.append(row)
        groups[name] = rows
    out["crop_groups"] = groups
    out["crop_overlap"] = {"iou": summary(iou), "inter_area": summary(area)}
    # ---- multi-view: per unordered view pair positive gradients
    ups = [(a, b) for a in range(V) for b in range(a + 1, V)]
    G = []
    for a, b in ups:
        m = ((terms["va"] == a) & (terms["vb"] == b)) | ((terms["va"] == b) & (terms["vb"] == a))
        G.append(flat_grad(terms["c_pos"][m].sum(), pe))
    Gm = torch.stack(G); nrm = Gm.norm(dim=1).clamp_min(1e-30); C = (Gm @ Gm.T) / (nrm[:, None] * nrm[None, :])
    out["view_pairs"] = {"pairs": ups, "grad_norm": [float(x) for x in nrm], "cos_matrix": [[round(float(x), 4) for x in r] for r in C],
                         "mean_offdiag_cos": float(C[~torch.eye(len(ups), dtype=torch.bool, device=C.device)].mean()) if len(ups) > 1 else None,
                         "cancellation_ratio": float(Gm.sum(0).norm() / nrm.sum())}
    # ---- token-level pulls (z-space): cancellation of the positive pulls on each token, split by the token's lowest-IoU partner
    pa, pb, _ = positive_pulls(terms, crit, vz)
    ntok = V * B; D = pa.shape[1]
    ta = (terms["va"] * B + terms["img"]); tb = (terms["vb"] * B + terms["img"])
    tot = torch.zeros(ntok, D, device=device).index_add_(0, ta, pa).index_add_(0, tb, pb)
    nsum = torch.zeros(ntok, device=device).index_add_(0, ta, pa.norm(dim=1)).index_add_(0, tb, pb.norm(dim=1))
    minIoU = torch.full((ntok,), 2.0, device=device).scatter_reduce(0, ta, iou.float(), "amin").scatter_reduce(0, tb, iou.float(), "amin")
    ratio = tot.norm(dim=1) / nsum.clamp_min(1e-30)
    strong = minIoU < IOU_BINS[1]
    out["token_pull"] = {"autograd_check_rel_err": float((tot - zP).norm() / zP.norm().clamp_min(1e-30)),
                         "cancellation_ratio_mean": float(ratio.mean()),
                         "cancellation_ratio_tokens_with_partner_iou_lt_0.1": float(ratio[strong].mean()) if int(strong.sum()) else None,
                         "cancellation_ratio_other_tokens": float(ratio[~strong].mean()) if int((~strong).sum()) else None,
                         "n_tokens_with_partner_iou_lt_0.1": int(strong.sum())}
    return out


@torch.no_grad()
def geometry(enc, proj, images_uint8: np.ndarray, uids: np.ndarray, clean_tf, device, eps: float, batch: int = 500) -> dict[str, Any]:
    enc, proj = enc.eval(), proj.eval()
    hs, zs = [], []
    for lo in range(0, len(uids), batch):
        x = torch.stack([clean_tf(Image.fromarray(images_uint8[int(u)])) for u in uids[lo:lo + batch]]).to(device)
        h = enc(x); hs.append(h.float()); zs.append(F.normalize(proj(h), dim=1, eps=eps).float())
    out = {}
    for k, X in (("h", torch.cat(hs)), ("z", torch.cat(zs))):
        mu = X.mean(0); ev = torch.linalg.eigvalsh(torch.cov((X - mu).T)).flip(0)
        out[k] = {"n": int(len(X)), "dim": int(X.shape[1]), "norm": summary(X.norm(dim=1)), "mean_vector_norm": float(mu.norm()),
                  "spectrum_top16_normalised": [round(float(v), 6) for v in ev[:16] / ev.clamp_min(0).sum()], "effective_rank": eff_rank(ev)}
    return out


def resolve_checkpoints(available: list[str], requested=(20, 100, 400, 800)) -> list[dict[str, Any]]:
    """Map requested epochs to saved checkpoints: exact if saved, else the nearest TRAINED checkpoint (epoch >= 1; initial.pt is never a
    substitute); a substitute already used for another request is measured once.  Every mapping is recorded."""
    ep = {}
    for n in available:
        if n.startswith("epoch_") and n.endswith(".pt"):
            ep[int(n[6:-3])] = n
    out, used = [], set()
    for r in requested:
        if not ep:
            out.append({"requested": r, "checkpoint": None, "note": "no trained checkpoint"}); continue
        e = r if r in ep else min(ep, key=lambda x: (abs(x - r), x))
        note = "exact" if e == r else f"epoch {r} not saved; nearest trained = {e}"
        if e in used:
            out.append({"requested": r, "checkpoint": ep[e], "epoch": e, "note": note + " (already measured)", "duplicate": True})
        else:
            used.add(e); out.append({"requested": r, "checkpoint": ep[e], "epoch": e, "note": note, "duplicate": False})
    return out


# =============================================================================================================================
# P123 (v6 V6-GRAD): the same fixed batches, (i) ACTUAL — each checkpoint's own loss / pairing / scale — and (ii) COUNTERFACTUAL-READONLY —
# same checkpoint, images, pairs and gradient routing, only the scorer f replaced; instantaneous derivatives, no parameter update, never a
# training result.  Everything above this line is the frozen P115 code and is not modified (P115 outputs stay reproducible).
#   For A_c(T) = cT − T²/2 with T = tanh f(s):   ∂A_c/∂s = (c − T)(1 − T²) f′(s)   (c = +1 on P, −1 on Q; the loss is −J = −mean_P A_+ − mean_Q A_−).
#   Affine f = a(s − κ): scalar peaks s± = κ ∓ log2 / (2a) (T = ∓1/3), max |∂A/∂s| = 32a/27 when the peak lies in [−1, 1].
#   Curvature (v6 §7): f = a[(s − κ) + λ s(1 − s)], f′ = a[1 + λ(1 − 2s)]; κ is the affine anchor, the actual zero is solved numerically.
# =============================================================================================================================
LOG2 = math.log(2.0)
PEAK_WINDOW = 0.05                                    # |s − s_peak| <= 0.05 counts as "near the scalar peak" (fixed before any run)
COUNTERFACTUAL_SCORERS = tuple([{"name": f"affine_a{a:g}_k{k:g}", "kind": "affine", "a": float(a), "kappa": float(k), "lam": 0.0}
                                for a in (1, 2, 3) for k in (0.25, 0.5, 0.75)]
                               + [{"name": f"curv_a2_k0.5_lam{l:+g}", "kind": "curv", "a": 2.0, "kappa": 0.5, "lam": float(l)} for l in (-0.25, 0.25)])


def scorer_f(spec: dict, s: Tensor) -> Tensor:
    a, k, lam = spec["a"], spec["kappa"], spec.get("lam", 0.0)
    return a * ((s - k) + lam * s * (1.0 - s))


def scorer_fprime(spec: dict, s: Tensor) -> Tensor:
    return spec["a"] * (1.0 + spec.get("lam", 0.0) * (1.0 - 2.0 * s))


def scorer_from_critic(crit) -> dict:
    """The run's own cosine-family critic as an affine spec f = a s + b = a(s − κ), κ = −b/a."""
    a, b = float(crit.scale), float(crit.bias)
    return {"name": "actual", "kind": "affine", "a": a, "kappa": -b / a, "lam": 0.0}


def scorer_zero(spec: dict) -> float | None:
    """Actual zero of f on [−1, 1] (bisection); None if f has no sign change there."""
    lo, hi = -1.0, 1.0
    f = lambda x: float(scorer_f(spec, torch.tensor(x, dtype=torch.float64)))
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def scorer_peaks(spec: dict) -> dict:
    """In-range argmax of |∂A_c/∂s| on [−1, 1] (grid of 8001 points), plus the affine closed form and whether it lies in range."""
    s = torch.linspace(-1.0, 1.0, 8001, dtype=torch.float64); t = torch.tanh(scorer_f(spec, s)); fp = scorer_fprime(spec, s)
    out = {"actual_zero": scorer_zero(spec)}
    for name, c in (("pos", 1.0), ("neg", -1.0)):
        g = ((c - t) * (1 - t.square()) * fp).abs(); i = int(g.argmax())
        out[name] = {"s_peak_in_range": float(s[i]), "max_abs_dA_ds_in_range": float(g[i])}
    if spec["kind"] == "affine":
        sp, sn = spec["kappa"] - LOG2 / (2 * spec["a"]), spec["kappa"] + LOG2 / (2 * spec["a"])
        out["pos"].update({"s_peak_closed_form": sp, "closed_form_in_range": -1.0 <= sp <= 1.0})
        out["neg"].update({"s_peak_closed_form": sn, "closed_form_in_range": -1.0 <= sn <= 1.0})
        out["max_abs_dA_ds_closed_form"] = 32 * spec["a"] / 27
    return out


def pair_sets(cfg: dict, V: int, B: int, *, pair_seed: int, device, simclr_counterfactual_scope: str = "all_view_tokens") -> dict:
    """Token-index pair sets reproducing the run's VCS pairing exactly (objective_terms above): cross-view K cyclic shifts with the same
    pair generator, or all view tokens.  SimCLR checkpoints have no VCS pairing; their counterfactual uses `simclr_counterfactual_scope`
    (all view tokens, full gradient = A-P3's structure), recorded in the output."""
    method = cfg["run"]["method"]
    if method == "vcs_qmi":
        scope = cfg["pairing"].get("pair_scope", "cross_view_k"); nd = bool(cfg["pairing"]["negative_detach"]); k = int(cfg["pairing"].get("k", 8))
    else:
        scope, nd, k = simclr_counterfactual_scope, False, int(cfg["pairing"].get("k", 8))
    ar = torch.arange(B, device=device)
    if scope == "all_view_tokens":
        n = V * B; ids = torch.arange(n, device=device) % B; same = ids[:, None] == ids[None, :]; diag = torch.eye(n, dtype=torch.bool, device=device)
        pl, pr = (same & ~diag).nonzero(as_tuple=True); ql, qr = (~same).nonzero(as_tuple=True)
        return {"scope": scope, "negative_detach": nd, "pl": pl, "pr": pr, "ql": ql, "qr": qr, "wp": 1.0 / len(pl), "wq": 1.0 / len(ql),
                "p_img": ids[pl], "p_va": pl // B, "p_vb": pr // B, "q_img_l": ids[ql], "q_img_r": ids[qr]}
    from reference.ssl_core import cyclic_negative_indices  # noqa: PLC0415
    g = torch.Generator().manual_seed(pair_seed); pairs = [(a, b) for a in range(V) for b in range(a + 1, V)]
    pl, pr, ql, qr, pva, pvb = [], [], [], [], [], []
    for a, b in pairs:
        idx, _ = cyclic_negative_indices(B, k, generator=g, device=device)
        pl.append(a * B + ar); pr.append(b * B + ar); pva.append(torch.full_like(ar, a)); pvb.append(torch.full_like(ar, b))
        ql.append((a * B + ar).unsqueeze(0).expand(k, -1).reshape(-1)); qr.append((b * B + idx).reshape(-1))
    pl, pr, ql, qr = torch.cat(pl), torch.cat(pr), torch.cat(ql), torch.cat(qr)
    return {"scope": scope, "negative_detach": nd, "pl": pl, "pr": pr, "ql": ql, "qr": qr, "wp": 1.0 / len(pl), "wq": 1.0 / len(ql),
            "p_img": pl % B, "p_va": torch.cat(pva), "p_vb": torch.cat(pvb), "q_img_l": ql % B, "q_img_r": qr % B}


def pair_similarities(z: Tensor, sets: dict) -> tuple[Tensor, Tensor]:
    """s on the fixed P and Q sets (with graph); Q's right side detached iff the run detaches negatives.  All-view scope via the token Gram
    matrix (same values as the elementwise products, far less memory)."""
    zr = z.detach() if sets["negative_detach"] else z
    if sets["scope"] == "all_view_tokens":
        return (z @ z.T)[sets["pl"], sets["pr"]], (z @ zr.T)[sets["ql"], sets["qr"]]
    return (z[sets["pl"]] * z[sets["pr"]]).sum(-1), (z[sets["ql"]] * zr[sets["qr"]]).sum(-1)


def cf_terms(z: Tensor, sets: dict, spec: dict, sims: tuple[Tensor, Tensor] | None = None) -> dict:
    """−J with scorer `spec` on the fixed pair sets: Lp = Σ_P wp (−T + T²/2), Lq = Σ_Q wq (T + T²/2).
    Returns the losses (with graph) and detached per-pair arrays s, f, T, c − T, f′, |∂A_c/∂s|."""
    sp, sq = sims if sims is not None else pair_similarities(z, sets)
    fp_, fq_ = scorer_f(spec, sp), scorer_f(spec, sq); tp, tq = torch.tanh(fp_), torch.tanh(fq_)
    c_pos = sets["wp"] * (-tp + 0.5 * tp.square()); c_neg = sets["wq"] * (tq + 0.5 * tq.square())
    out = {"Lp": c_pos.sum(), "Lq": c_neg.sum(), "c_pos": c_pos, "c_neg": c_neg}
    for nm, s, f, t, c in (("pos", sp, fp_, tp, 1.0), ("neg", sq, fq_, tq, -1.0)):
        s, f, t = s.detach(), f.detach(), t.detach(); fpr = scorer_fprime(spec, s)
        out[nm] = {"s": s, "f": f, "T": t, "c_minus_T": c - t, "fprime": fpr, "abs_dA_ds": ((c - t) * (1 - t.square()) * fpr).abs()}
    return out


def _gradblock(L: Tensor, z: Tensor, r: Tensor, h: Tensor, pe: list, pp: list) -> dict:
    gs = torch.autograd.grad(L, [z, r, h] + pe + pp, retain_graph=True, allow_unused=True)
    flat = lambda xs, ps: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(xs, ps)])
    return {"z": gs[0], "r": gs[1], "h": gs[2], "enc": flat(gs[3:3 + len(pe)], pe), "proj": flat(gs[3 + len(pe):], pp)}


def _norms(gp: dict, gq: dict) -> dict:
    return {k: {"P": float(gp[k].norm()), "Q": float(gq[k].norm()), "total": float((gp[k] + gq[k]).norm()), "cos_P_Q": _cos(gp[k].flatten(), gq[k].flatten())}
            for k in ("z", "r", "h", "enc", "proj")}


def _group_shares(c_pos: Tensor, mask_list: list, labels: list, gP: dict, h: Tensor, pe: list) -> list:
    """Projection contributions ⟨g_b, g_P⟩ / ‖g_P‖² of disjoint positive groups (sum to 1; may be negative) at the encoder-parameter and h
    levels, with ‖g_b‖; plus each group's share of the scalar loss."""
    rows, eP2, hP2 = [], float(gP["enc"] @ gP["enc"]), float((gP["h"] * gP["h"]).sum())
    for lab, m in zip(labels, mask_list):
        row = {"bin": lab, "n_pos": int(m.sum())}
        if int(m.sum()):
            Lb = c_pos[m].sum()
            gs = torch.autograd.grad(Lb, [h] + pe, retain_graph=True, allow_unused=True)
            ge = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(gs[1:], pe)])
            row.update({"enc_proj_contribution": float(ge @ gP["enc"]) / max(eP2, 1e-30), "enc_norm": float(ge.norm()),
                        "h_proj_contribution": float((gs[0] * gP["h"]).sum()) / max(hP2, 1e-30), "h_norm": float(gs[0].norm())})
        rows.append(row)
    return rows


def scorer_block(cfg: dict, z: Tensor, r: Tensor, h: Tensor, pe: list, pp: list, sets: dict, spec: dict, iou_pos: Tensor, labels_img: Tensor | None,
                 *, groups_at_encoder: bool = True, sims: tuple[Tensor, Tensor] | None = None) -> dict:
    """All recorded quantities for one scorer on one batch (positives and negatives separately)."""
    t = cf_terms(z, sets, spec, sims)
    gp, gq = _gradblock(t["Lp"], z, r, h, pe, pp), _gradblock(t["Lq"], z, r, h, pe, pp)
    pk = scorer_peaks(spec)
    out: dict[str, Any] = {"scorer": {k: spec[k] for k in ("name", "kind", "a", "kappa", "lam")}, "peaks": pk,
                           "Lp": float(t["Lp"].detach()), "Lq": float(t["Lq"].detach()), "loss": float((t["Lp"] + t["Lq"]).detach()),
                           "grad_vector_norms": _norms(gp, gq)}
    for nm in ("pos", "neg"):
        d = t[nm]; w = sets["wp"] if nm == "pos" else sets["wq"]
        sp_ = pk[nm]["s_peak_in_range"]
        out[nm] = {k: summary(d[k]) for k in ("s", "f", "T", "c_minus_T", "fprime", "abs_dA_ds")}
        out[nm]["scalar_total_abs_action"] = float(w * d["abs_dA_ds"].sum())
        out[nm]["mass_near_peak"] = float(((d["s"] - sp_).abs() <= PEAK_WINDOW).float().mean())
    # positive groups by true crop IoU
    masks = [(iou_pos >= lo) & (iou_pos < hi) for lo, hi in zip(IOU_BINS[:-1], IOU_BINS[1:])]
    labs = [[lo, min(hi, 1.0)] for lo, hi in zip(IOU_BINS[:-1], IOU_BINS[1:])]
    rows = _group_shares(t["c_pos"], masks, labs, gp, h, pe) if groups_at_encoder else [{"bin": l, "n_pos": int(m.sum())} for l, m in zip(labs, masks)]
    act = t["pos"]["abs_dA_ds"]; tot_act = float(act.sum())
    for row, m in zip(rows, masks):
        if int(m.sum()):
            row.update({"scalar_action_share": float(act[m].sum()) / max(tot_act, 1e-30), "s_pos_mean": float(t["pos"]["s"][m].mean()),
                        "T_pos_mean": float(t["pos"]["T"][m].mean())})
    out["iou_groups_pos"] = rows
    # isolated semantic diagnostic view (labels NEVER filter or weight anything; negatives stay the random population)
    if labels_img is not None:
        same = labels_img[sets["q_img_l"]] == labels_img[sets["q_img_r"]]
        sem = {"frac_same_class_neg": float(same.float().mean()), "n_neg": int(same.numel())}
        if int(same.sum()):
            aq = t["neg"]["abs_dA_ds"]
            sem["scalar_action_share_same_class"] = float(aq[same].sum()) / max(float(aq.sum()), 1e-30)
            gh = torch.autograd.grad(t["c_neg"][same].sum(), h, retain_graph=True)[0]
            sem["h_proj_contribution_same_class"] = float((gh * gq["h"]).sum()) / max(float((gq["h"] * gq["h"]).sum()), 1e-30)
            sem["h_norm_same_class"] = float(gh.norm())
        out["semantic_view_isolated"] = sem
    return out


def grad_batch_diagnostics(cfg: dict, enc, proj, crit, views: list[Tensor], boxes: np.ndarray, device, *, pair_seed: int,
                           labels_img: Tensor | None = None, scorers=COUNTERFACTUAL_SCORERS, only_actual: bool = False) -> dict[str, Any]:
    """One fixed batch.  BN in training mode on throw-away copies (as a training step, buffers discarded) — the read-only copy restores the
    train-mode BN condition of the update; eval-mode measurement (geometry) is kept separate.  Critic scored clean (eval)."""
    enc, proj = copy.deepcopy(enc).train(), copy.deepcopy(proj).train()
    if crit is not None:
        crit = copy.deepcopy(crit).eval()
    eps = cfg["model"]["normalization"]["eps"]; V, B = len(views), views[0].shape[0]
    h = enc(torch.cat([v.to(device) for v in views])); r = proj(h); z = F.normalize(r, dim=1, eps=eps)
    pe = [q for q in enc.parameters() if q.requires_grad]; pp = [q for q in proj.parameters() if q.requires_grad]
    sets = pair_sets(cfg, V, B, pair_seed=pair_seed, device=device)
    ov = np.zeros((B, V, V), dtype=np.float64)
    for i in range(B):
        for x in range(V):
            for y in range(V):
                ov[i, x, y] = crop_overlap(boxes[i, x], boxes[i, y])[0]
    ovt = torch.as_tensor(ov, device=device)
    iou_pos = ovt[sets["p_img"], sets["p_va"], sets["p_vb"]]
    lab = labels_img.to(device) if labels_img is not None else None
    sims = pair_similarities(z, sets)
    out: dict[str, Any] = {"bn_mode": "train (throw-away copy)", "pairing": {"scope": sets["scope"], "negative_detach": sets["negative_detach"],
                           "n_pos": int(len(sets["pl"])), "n_neg": int(len(sets["ql"])), "simclr_counterfactual_pairing": cfg["run"]["method"] != "vcs_qmi"}}
    # ---- ACTUAL: the checkpoint's own loss
    if cfg["run"]["method"] == "vcs_qmi":
        spec = scorer_from_critic(crit)
        act = scorer_block(cfg, z, r, h, pe, pp, sets, spec, iou_pos, lab, sims=sims)
        ref = objective_terms(cfg, crit, list(z.chunk(V)), pair_seed=pair_seed)       # P115 decomposition of the training loss
        act["check_vs_training_loss"] = {"Lp_rel_err": abs(act["Lp"] - float(ref["Lp"].detach())) / max(abs(float(ref["Lp"].detach())), 1e-30),
                                         "Lq_rel_err": abs(act["Lq"] - float(ref["Lq"].detach())) / max(abs(float(ref["Lq"].detach())), 1e-30)}
        out["actual"] = act
    else:
        ref = objective_terms(cfg, None, list(z.chunk(V)), pair_seed=pair_seed)        # SimCLR NT-Xent split L = mean(−logit_pos) + mean(lse)
        gp, gq = _gradblock(ref["Lp"], z, r, h, pe, pp), _gradblock(ref["Lq"], z, r, h, pe, pp)
        im, va, vb = ref["img"], ref["va"], ref["vb"]
        iou_s = ovt[im, va, vb]
        masks = [(iou_s >= lo) & (iou_s < hi) for lo, hi in zip(IOU_BINS[:-1], IOU_BINS[1:])]
        out["actual"] = {"loss_kind": "simclr_nt_xent", "Lp": float(ref["Lp"].detach()), "Lq": float(ref["Lq"].detach()),
                         "grad_vector_norms": _norms(gp, gq), "pos": {"s": summary(ref["s_pos"])}, "neg": {"s": summary(ref["s_neg"])},
                         "iou_groups_pos": _group_shares(ref["c_pos"], masks, [[lo, min(hi, 1.0)] for lo, hi in zip(IOU_BINS[:-1], IOU_BINS[1:])], gp, h, pe)}
    if only_actual:
        return out
    # ---- COUNTERFACTUAL-READONLY: same z, pairs, routing; only f replaced
    out["counterfactual"] = {spec["name"]: scorer_block(cfg, z, r, h, pe, pp, sets, spec, iou_pos, lab, sims=sims) for spec in scorers}
    return out
