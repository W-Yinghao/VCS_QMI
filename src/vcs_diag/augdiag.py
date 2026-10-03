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
