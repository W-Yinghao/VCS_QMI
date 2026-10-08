"""VL1-10 (Table A): frozen-feature region–expression scorers on RefCOCOg (UMD), training side only.

Features: the VL1-00 cache (open_clip ViT-B-16-quickgelu OpenAI; L2-normalised here).  Per eligible image (>= 2 referred objects): regions = the
referred objects, phrases = their expressions (aliases = separate columns), laws p_G / q_G from `vcs_vl.pairlaw.pair_laws` (uniform regions; Q =
product of the image's marginals, matching cells kept); images weighted equally.
Routes (same scorer class F2 = MLP(concat(u, v)) 1024-256-256-1, GELU, last layer small init, zero bias, T = tanh f):
  vcs       1 - mean_G J_G                        (padded_batch_loss "vcs")
  js        balanced logistic, same P / Q           ("balanced_logistic")
  softmax   multi-positive candidate cross-entropy  ("conditional_softmax"; task panel only — no common J)
  siglip    SigLIP-style task control: per-pair sigmoid, every (region, phrase) cell of the image weighted equally (actual positive / negative
            counts, prior pi_G = #pos / #cells recorded); common J read with T = tanh(f - logit(pi_G) / 2)  (u = 2 f is the raw logit)
  rff       same-target kernel route: RFF(concat(u, v)) D 1024, bandwidth x median distance, weighted ridge on +-1 targets under M = (P+Q)/2,
            tanh scale calibrated on CAL (two-stage fit, not the exact bounded-J optimum)
  raw       raw CLIP cosine (no training)
Selection (separately saved): estimator_selected = CAL max common J; task_selected = DEV max image-macro Top-1 (earliest tie).  AdamW, weight
decay 1e-4, 32 images per batch, <= 50 epochs or 3 000 updates, evaluation every 50 updates, patience 10 evaluations (on both criteria).
    python scripts/vl1_10_fit.py --route vcs --ns 1000,4000,all --seeds 0,1,2 --lrs 1e-4,5e-4,2e-3 --out <dir> [--smoke]
"""
from __future__ import annotations

import argparse
import copy
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
from vcs_vl import refcocog as RG  # noqa: E402
from vcs_vl.pairlaw import pair_laws, padded_batch_loss, padded_j  # noqa: E402

FEAT = Path("/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_clip_vitb16_openai/train_side_features.pt")
BATCH, MAX_EPOCHS, MAX_UPDATES, EVAL_EVERY, PATIENCE, WD = 32, 50, 3000, 50, 10, 1e-4
RFF_D, RFF_BW, RFF_LAM = 1024, (0.5, 1.0, 2.0), (1e-4, 1e-2, 1.0)


# ----------------------------------------------------------------------------------------------------------- data
def load_scene_tensors(smoke: bool = False, feat: Path = FEAT, law: str = "region_uniform"):
    D = torch.load(feat, weights_only=False)
    img = F.normalize(D["image_feat_fp16"].float(), dim=1); txt = F.normalize(D["text_feat_fp16"].float(), dim=1)
    oi = {tuple(k): i for i, k in enumerate(D["obj_keys"])}; ti = {tuple(k): i for i, k in enumerate(D["text_keys"])}
    scenes, _ = RG.load_scenes(resolve_paths=False); role = RG.dev_roles(scenes); pref = RG.fit_prefixes(scenes, role)
    out = {"FIT": {}, "CAL": [], "DEV": []}
    for s in scenes:
        r = role[s.image_id]
        if r not in ("FIT", "CAL", "DEV") or len(s.referred) < 2 or (s.image_id, s.referred[0].ann_id) not in oi:
            continue
        A, anns, sents = s.compatibility()
        prior = None if law == "region_uniform" else A.sum(1) / A.sum()   # phrase_uniform: p(r) ∝ number of expressions of r (VL1-11)
        p, q = pair_laws(A, prior)
        rec = {"image_id": s.image_id, "U": img[[oi[(s.image_id, a)] for a in anns]], "Ud": img[[oi[(s.image_id, o.ann_id)] for o in s.distractors]],
               "V": txt[[ti[(s.image_id, anns[int(np.argmax(A[:, j]))], sid)] for j, sid in enumerate(sents)]],
               "A": torch.as_tensor(A, dtype=torch.float32), "P": torch.as_tensor(p, dtype=torch.float32), "Q": torch.as_tensor(q, dtype=torch.float32),
               "target": torch.as_tensor(A.argmax(0), dtype=torch.long)}
        if r == "FIT":
            out["FIT"][s.image_id] = rec
        else:
            out[r].append(rec)
    if smoke:
        out["CAL"], out["DEV"] = out["CAL"][:64], out["DEV"][:64]
    return out, pref


def pad(recs, with_distractors: bool = False):
    n = len(recs); R = max(len(r["U"]) + (len(r["Ud"]) if with_distractors else 0) for r in recs); W = max(len(r["V"]) for r in recs); d = recs[0]["U"].shape[1]
    U = torch.zeros(n, R, d); V = torch.zeros(n, W, d); P = torch.zeros(n, R, W); Q = torch.zeros(n, R, W); A = torch.zeros(n, R, W)
    rmask = torch.zeros(n, R, dtype=torch.bool); wmask = torch.zeros(n, W, dtype=torch.bool); tgt = torch.full((n, W), -1, dtype=torch.long)
    for i, r in enumerate(recs):
        m, k = len(r["U"]), len(r["V"]); Uall = torch.cat([r["U"], r["Ud"]]) if with_distractors else r["U"]
        U[i, :len(Uall)] = Uall; V[i, :k] = r["V"]; P[i, :m, :k] = r["P"]; Q[i, :m, :k] = r["Q"]; A[i, :m, :k] = r["A"]
        rmask[i, :len(Uall)] = True; wmask[i, :k] = True; tgt[i, :k] = r["target"]
    return {"U": U, "V": V, "P": P, "Q": Q, "A": A, "rmask": rmask, "wmask": wmask, "tgt": tgt}


# ----------------------------------------------------------------------------------------------------------- scorer
class PairMLP(nn.Module):
    """Scorer class shared by every learned route.
    residual (primary, VL1-10 freeze): f = a * cos(u, v) + b + g([u, v, u*v]) with a = softplus(alpha) > 0, (a, b) initialised at the A-P3 scorer
      (2, -1) and g's last layer zero-initialised, so every route starts exactly at the raw-CLIP ranking and learns corrections (the package's
      identity-initialisation principle; a positive affine map of cos alone cannot change the ranking, the MLP can).
    concat (package F2, sensitivity row): f = MLP(concat(u, v)), last layer small init, zero bias."""

    def __init__(self, d: int = 512, hidden: int = 256, kind: str = "residual") -> None:
        super().__init__()
        self.kind = kind
        din = 3 * d if kind == "residual" else 2 * d
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if kind == "residual":
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
            self.alpha = nn.Parameter(torch.tensor(math.log(math.expm1(2.0)))); self.b = nn.Parameter(torch.tensor(-1.0))
        else:
            nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1); nn.init.zeros_(self.net[-1].bias)

    def forward(self, U, V):                       # U [n, R, d], V [n, W, d] (L2-normalised) -> f [n, R, W]
        n, R, d = U.shape; W = V.shape[1]
        Ue, Ve = U[:, :, None, :].expand(n, R, W, d), V[:, None, :, :].expand(n, R, W, d)
        if self.kind == "residual":
            cos = torch.einsum("nrd,nwd->nrw", U, V)
            return F.softplus(self.alpha) * cos + self.b + self.net(torch.cat([Ue, Ve, Ue * Ve], dim=-1)).squeeze(-1)
        return self.net(torch.cat([Ue, Ve], dim=-1)).squeeze(-1)


def route_loss(route: str, f, b):
    if route == "vcs":
        return padded_batch_loss(f, b["P"], b["Q"], "vcs")
    if route == "js":
        return padded_batch_loss(f, b["P"], b["Q"], "balanced_logistic")
    if route == "softmax":
        return padded_batch_loss(f, b["P"], b["Q"], "conditional_softmax")
    if route == "siglip":
        valid = b["rmask"][:, :, None] & b["wmask"][:, None, :]
        y = (b["A"] > 0).float(); bce = F.binary_cross_entropy_with_logits(2 * f, y, reduction="none")
        return ((bce * valid).sum((1, 2)) / valid.sum((1, 2))).mean()
    raise ValueError(route)


def siglip_prior(b) -> torch.Tensor:
    valid = b["rmask"][:, :, None] & b["wmask"][:, None, :]
    return ((b["A"] > 0) & valid).sum((1, 2)).float() / valid.sum((1, 2)).float()


# ----------------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(score_fn, recs, route: str) -> dict:
    """score_fn(U, V) -> f.  Common J on referred rows (not for softmax / raw); Top-1 over referred rows (primary) and referred + distractors."""
    Js, q1, macro, q1_all, q1c, macroc = [], [], [], [], [], []
    for s in range(0, len(recs), 64):
        chunk = recs[s:s + 64]; b = pad(chunk); f = score_fn(b["U"], b["V"])
        if route in ("vcs", "js", "rff"):
            Js.append(padded_j(f, b["P"], b["Q"]))
        elif route == "siglip":
            pi = siglip_prior(b).clamp(1e-6, 1 - 1e-6); Js.append(padded_j(f - 0.5 * torch.log(pi / (1 - pi))[:, None, None], b["P"], b["Q"]))
        fm = f.masked_fill(~b["rmask"][:, :, None], -torch.inf); pred = fm.argmax(1); hit = (pred == b["tgt"]) & b["wmask"]
        q1.append(hit.sum().item()); macro += (hit.sum(1).float() / b["wmask"].sum(1).float()).tolist()
        # VL1-11 prior-corrected ranking: 2 f + log p_G(r) (O1 §3.4); identical to f under the region-uniform law
        logp = torch.log(b["P"].sum(2).clamp_min(1e-12))[:, :, None]; fc = (2 * f + logp).masked_fill(~b["rmask"][:, :, None], -torch.inf)
        hitc = (fc.argmax(1) == b["tgt"]) & b["wmask"]; q1c.append(hitc.sum().item()); macroc += (hitc.sum(1).float() / b["wmask"].sum(1).float()).tolist()
        ba = pad(chunk, with_distractors=True); fa = score_fn(ba["U"], ba["V"]).masked_fill(~ba["rmask"][:, :, None], -torch.inf)
        q1_all.append(((fa.argmax(1) == ba["tgt"]) & ba["wmask"]).sum().item())
    nq = sum(len(r["V"]) for r in recs)
    out = {"top1_query": sum(q1) / nq, "top1_image_macro": float(np.mean(macro)), "top1_all_objects": sum(q1_all) / nq, "n_images": len(recs), "n_queries": nq,
           "top1_query_prior_corrected": sum(q1c) / nq, "top1_image_macro_prior_corrected": float(np.mean(macroc))}
    if Js:
        out["J_common"] = float(torch.cat(Js).mean())
    return out


# ----------------------------------------------------------------------------------------------------------- neural routes
def fit_neural(route, fit_recs, cal, dev, lr, seed, max_updates=MAX_UPDATES, scorer="residual"):
    torch.manual_seed(seed); m = PairMLP(d=fit_recs[0]["U"].shape[1], kind=scorer); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    g = np.random.default_rng(seed * 1000 + 17); upd, curve = 0, []
    best = {"est": (-math.inf, 0, None), "task": (-math.inf, 0, None)}; since = 0
    t0 = time.time()

    def ev():
        m.eval(); c = evaluate(m, cal, route) if route != "softmax" else None; d = evaluate(m, dev, route); m.train(); return c, d
    c, d = ev(); curve.append({"update": 0, "cal": c, "dev": d})
    if c is not None:
        best["est"] = (c["J_common"], 0, copy.deepcopy(m.state_dict()))
    best["task"] = (d["top1_image_macro"], 0, copy.deepcopy(m.state_dict()))
    for ep in range(MAX_EPOCHS):
        order = g.permutation(len(fit_recs))
        for s in range(0, len(order) - BATCH + 1, BATCH):
            b = pad([fit_recs[i] for i in order[s:s + BATCH]]); loss = route_loss(route, m(b["U"], b["V"]), b)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); upd += 1
            if upd % EVAL_EVERY == 0:
                c, d = ev(); curve.append({"update": upd, "loss": float(loss), "cal": c, "dev": d}); improved = False
                if c is not None and c["J_common"] > best["est"][0]:
                    best["est"] = (c["J_common"], upd, copy.deepcopy(m.state_dict())); improved = True
                if d["top1_image_macro"] > best["task"][0]:
                    best["task"] = (d["top1_image_macro"], upd, copy.deepcopy(m.state_dict())); improved = True
                since = 0 if improved else since + 1
                if since >= PATIENCE:
                    break
            if upd >= max_updates:
                break
        if upd >= max_updates or since >= PATIENCE:
            break
    res = {"lr": lr, "seed": seed, "updates": upd, "epochs": ep + 1, "fit_seconds": time.time() - t0, "curve": curve}
    for key in ("est", "task"):
        if best[key][2] is None:
            continue
        m.load_state_dict(best[key][2]); m.eval()
        res[f"{key}_selected"] = {"update": best[key][1], "cal": evaluate(m, cal, route) if route != "softmax" else None, "dev": evaluate(m, dev, route),
                                  "state": {k: v.cpu() for k, v in best[key][2].items()}}
    return res


# ----------------------------------------------------------------------------------------------------------- RFF route
def fit_rff(fit_recs, cal, dev, seed):
    g = torch.Generator().manual_seed(1000 + seed); d = fit_recs[0]["U"].shape[1]
    def pairs3(U, V):  # [u, v, u*v] — the same input as the residual MLP
        Ue, Ve = U[:, None, :].expand(-1, len(V), -1), V[None].expand(len(U), -1, -1); return torch.cat([Ue, Ve, Ue * Ve], -1).reshape(-1, 3 * d)
    Xs = torch.cat([pairs3(r["U"], r["V"]) for r in fit_recs[:500]])
    idx = torch.randint(0, len(Xs), (2000, 2), generator=g); med = float((Xs[idx[:, 0]] - Xs[idx[:, 1]]).norm(dim=1).median())

    def make_phi(Wf, bf):
        def phi(U, V):  # [cos(u, v), RFF([u, v, u*v]), 1]: the kernel route sees the raw similarity, like the residual MLP
            n, R, _ = U.shape; Wn = V.shape[1]
            Ue, Ve = U[:, :, None, :].expand(n, R, Wn, d), V[:, None, :, :].expand(n, R, Wn, d); x = torch.cat([Ue, Ve, Ue * Ve], -1).double()
            cos = torch.einsum("nrd,nwd->nrw", U, V).double()[..., None]
            return torch.cat([cos, math.sqrt(2.0 / RFF_D) * torch.cos(x @ Wf + bf), torch.ones(n, R, Wn, 1, dtype=torch.float64)], -1)
        return phi
    t0 = time.time(); cands = []
    cal_pads = [pad(cal[s:s + 64]) for s in range(0, len(cal), 64)]
    for bw in RFF_BW:
        Wf = torch.randn(3 * d, RFF_D, generator=g, dtype=torch.float64) / (bw * med); bf = torch.rand(RFF_D, generator=g, dtype=torch.float64) * 2 * math.pi
        phi = make_phi(Wf, bf)
        A = torch.zeros(RFF_D + 2, RFF_D + 2, dtype=torch.float64); dvec = torch.zeros(RFF_D + 2, dtype=torch.float64)
        for s in range(0, len(fit_recs), 64):
            b = pad(fit_recs[s:s + 64]); ph = phi(b["U"], b["V"]); M = 0.5 * (b["P"] + b["Q"]).double(); lab = (b["P"] - b["Q"]).double()
            A += torch.einsum("nrw,nrwi,nrwj->ij", M, ph, ph); dvec += torch.einsum("nrw,nrwi->i", lab, ph)
        A /= len(fit_recs); dvec /= len(fit_recs); sc = float(torch.diag(A).mean())
        cal_phi = [phi(b["U"], b["V"]) for b in cal_pads]
        for lam in RFF_LAM:
            w = 0.5 * torch.linalg.solve(A + lam * sc * torch.eye(RFF_D + 2, dtype=torch.float64), dvec)
            svals = [ph @ w for ph in cal_phi]
            for c in torch.logspace(-2, 2, 25).tolist():
                J = float(torch.cat([padded_j((c * sv).float(), b["P"], b["Q"]) for sv, b in zip(svals, cal_pads)]).mean())
                cands.append({"bw": bw, "lam": lam, "c": c, "cal_J": J, "phi": phi, "w": w})
        del cal_phi
    best = max(cands, key=lambda x: x["cal_J"])
    fn = lambda U, V: (best["c"] * (best["phi"](U, V) @ best["w"])).float()
    return {"seed": seed, "median_distance": med, "fit_seconds": time.time() - t0, "n_candidates": len(cands),
            "selected": {k: best[k] for k in ("bw", "lam", "c", "cal_J")}, "cal": evaluate(fn, cal, "rff"), "dev": evaluate(fn, dev, "rff")}


# ----------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, choices=["vcs", "js", "softmax", "siglip", "rff", "raw"]); ap.add_argument("--ns", default="1000,4000,all")
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--lrs", default="1e-4,5e-4,2e-3"); ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", default="residual", choices=["residual", "concat"]); ap.add_argument("--law", default="region_uniform", choices=["region_uniform", "phrase_uniform"]); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--features", default=str(FEAT), help="feature cache (CLIP default; FG-CLIP 2 / SigLIP 2 caches for VL1-12)")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    data, pref = load_scene_tensors(a.smoke, Path(a.features), a.law); out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    cal, dev = data["CAL"], data["DEV"]
    if a.route == "raw":
        res = {"cal": evaluate(lambda U, V: torch.einsum("nrd,nwd->nrw", U, V), cal, "raw"), "dev": evaluate(lambda U, V: torch.einsum("nrd,nwd->nrw", U, V), dev, "raw")}
        json.dump(res, open(out_dir / "raw.json", "w"), indent=1); print("raw", res); return 0
    for n_tag in a.ns.split(","):
        ids = pref[max(pref)] if n_tag == "all" else pref[int(n_tag)]
        fit = [data["FIT"][i] for i in ids if i in data["FIT"]]
        if a.smoke:
            fit = fit[:96]
        for seed in [int(x) for x in a.seeds.split(",")]:
            tag = f"{a.route}_N{n_tag}_s{seed}" + ("" if a.scorer == "residual" else "_concat") + ("" if a.law == "region_uniform" else "_phraseuniform")
            if (out_dir / f"{tag}.json").exists():
                print("skip", tag); continue
            if a.route == "rff":
                res = fit_rff(fit, cal, dev, seed)
            else:
                runs = [fit_neural(a.route, fit, cal, dev, float(lr), seed, max_updates=60 if a.smoke else MAX_UPDATES, scorer=a.scorer) for lr in a.lrs.split(",")]
                pick = lambda key, crit: max((r for r in runs if f"{key}_selected" in r), key=crit, default=None)
                est = pick("est", lambda r: r["est_selected"]["cal"]["J_common"]) if a.route != "softmax" else None
                task = pick("task", lambda r: r["task_selected"]["dev"]["top1_image_macro"])
                states = {"est": est["est_selected"].pop("state") if est else None, "task": task["task_selected"].pop("state")}
                for r in runs:
                    for key in ("est_selected", "task_selected"):
                        if key in r:
                            r[key].pop("state", None)
                res = {"route": a.route, "n_fit_images": len(fit), "seed": seed, "candidates": runs,
                       "estimator_selected": {"lr": est["lr"], **est["est_selected"]} if est else None,
                       "task_selected": {"lr": task["lr"], **task["task_selected"]}}
                torch.save(states, out_dir / f"{tag}_states.pt")
            res.update({"route": a.route, "n_fit_images": len(fit), "n_tag": n_tag, "law": a.law, "features": a.features, "scorer": a.scorer})
            json.dump(res, open(out_dir / f"{tag}.json", "w"), indent=1)
            ts = res.get("task_selected", res).get("dev", res.get("dev")); es = res.get("estimator_selected") or {}
            print(f"[{tag}] N {len(fit)} task DEV top1 macro {ts['top1_image_macro']:.4f} (all-obj {ts['top1_all_objects']:.4f})"
                  + (f" | est CAL J {es['cal']['J_common']:.4f}" if es and es.get("cal") else (f" | CAL J {res['cal']['J_common']:.4f}" if a.route == "rff" else "")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
