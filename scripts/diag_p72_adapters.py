"""D1 (package v2, Spec §8.1 / Plan §7.1) — drift decomposition of the P72 adapters, split by injected / clean training pairs.

    python scripts/diag_p72_adapters.py --p72 reports/P72_robust_r2r3.json --out <prefix> [--ckpt-dir <dir>] [--m 0,0.4] [--smoke]

DISCLOSED RETRAINING.  `robust_r2r3.py` never saved its adapters, so the P72 checkpoints do not exist.  This script retrains the minimal
set — (shift, m in --m, method in vcs/js/infonce/logistic), seed 0 — through the *original* `robust_r2r3.train_r` with the (lr, epochs)
that P72's clean-CAL selection recorded, the same topic partners (seed 20260927) and the same seeded injection (`inject`, sha256 of the
injected indices checked against the P72 JSON).  Reproduction check (results only): seed-0 SRC/TGT R@1, R3 AUROC and CAL loss vs P72.
The state dicts (early = end of epoch 1, final), the trajectory and the initial / final features are saved this time.

Diagnostics (labels "injected" / "clean" come from the injection record, never from a model):
  (A) training trajectory, from an instrumented copy of the same loop (identical RNG sequence; the parameters after training are compared
      with the original `train_r` model as a QC sentinel): per step the learning rate, |grad| of the injected-anchor part and of the clean
      part of the loss (anchor-row attribution: every term whose anchor image is injected), their cosine, the realised AdamW update |dtheta|
      (the Adam-preconditioned effective update; no gradient accumulation in P72) and the projection of each part onto the realised update
      direction; per-pair positive posterior-logit gradient and |dloss/du| by split.
  (B) early (epoch 1) and final checkpoints, fixed training pairing (the pairs R3 scored: injected partner, else fixed topic partner;
      caption 0): score, cosine, residual, actual gate 1 - T^2 (VCS), posterior q, posterior-logit gradient, |dl/du| of the positive term,
      |dl/du| of the anchor's full loss share (positive + its K negative terms), |dl/dv|, by split; bootstrap (by image) CI of injected - clean.
  (C) drift: parameter displacement (|W - I|_F, |b|, scalars), feature displacement |u_final - u_init| (u_init = the identity adapter) on
      FIT (by split), SRC-EVAL, TGT-EVAL; path length sum_t |dtheta_t| vs net displacement.
  (D) local loss swap on the same frozen scores and pairs: the VCS-form and JS-form posterior-logit gradients evaluated at the SAME
      posterior q of each checkpoint (q = sigma(2 f) for VCS, sigma(f) for JS / logistic; InfoNCE has no pairwise posterior -> absent).
      This is a local identity on one state, not a comparison of two training trajectories (Plan §7.1).
Unified scale: posterior logit l = logit(q).  VCS: T = tanh f, q = (1 + T)/2, l = 2 f, d(-a+(T))/dl = -(1 - T)(1 - T^2)/2.
JS / logistic: q = sigma(f), l = f, d softplus(-f)/dl = -(1 - q).  InfoNCE: row softmax p_ii, gradient -(1 - p_ii) wrt its own logit (not a posterior).
Class ratio / intercept (recorded, never applied to the training record): VCS and JS average P and Q separately (balanced, 1:1 total weight);
logistic averages the B x B in-batch matrix (1 : B-1 pairs; log(B-1) intercept correction for a probability reading); InfoNCE has no pairwise prior.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from precheck_a_adapters import Adapters, infonce_loss, load_split, logistic_loss, pair_sets, retrieval, topic_partners, vcs_loss  # noqa: E402
from precheck_a_wave2 import auroc  # noqa: E402
from robust_r2r3 import METHODS, SPLITS, inject, js_loss, r3_scores, train_r  # noqa: E402

QS = [0.01, 0.1, 0.5, 0.9, 0.99]
POSTERIOR_CONVENTION = {
    "vcs": {"posterior": "q = (1 + tanh f) / 2 = sigma(2 f)", "posterior_logit": "l = 2 f", "class_ratio_training": "1:1 (P and Q means separate; K = 8 pool negatives per anchor)", "intercept_correction_for_probability": 0.0},
    "js": {"posterior": "q = sigma(f)", "posterior_logit": "l = f", "class_ratio_training": "1:1 (P and Q means separate; K = 8 pool negatives per anchor)", "intercept_correction_for_probability": 0.0},
    "logistic": {"posterior": "q = sigma(f) of the in-batch pairwise logistic", "posterior_logit": "l = f", "class_ratio_training": "1 : (B - 1) = 1 : 255 in-batch pairs, mean over the B x B matrix", "intercept_correction_for_probability": math.log(255), "note": "the correction is rank-preserving and is NOT applied to any training or score record here"},
    "infonce": {"posterior": None, "posterior_logit": None, "class_ratio_training": "row / column softmax over B in-batch candidates", "intercept_correction_for_probability": None, "note": "no pairwise posterior; p_ii is a row-softmax probability, reported in its own column"},
}


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def summ(x) -> dict:
    x = np.asarray([float(v) for v in np.asarray(x).reshape(-1)], dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "quantiles": np.quantile(x, QS).tolist()}


def bootstrap_diff(x_inj, x_cln, reps=500, seed=20260930) -> dict:
    """Bootstrap by image (each anchor image is one unit; the two sets are disjoint image sets) of mean(injected) - mean(clean)."""
    xi, xc = np.asarray(x_inj, np.float64), np.asarray(x_cln, np.float64)
    if len(xi) == 0 or len(xc) == 0:
        return {"diff": None}
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        d.append(xi[rng.integers(0, len(xi), len(xi))].mean() - xc[rng.integers(0, len(xc), len(xc))].mean())
    d = np.asarray(d)
    return {"diff": float(xi.mean() - xc.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "reps": reps}


# ----------------------------------------------------------------------------------------------------------- per-pair quantities
def positive_terms(method, m, u, v, extra=None) -> dict:
    """Per positive pair: native logit f, posterior q (None for InfoNCE), score, residual, gate, posterior-logit gradient magnitude."""
    cos = (u * v).sum(-1)
    if method == "vcs":
        f = m.a * cos + m.b; T = torch.tanh(f); q = (1 + T) / 2
        return {"cos": cos, "f": f, "q": q, "score": T, "residual": 1 - T, "gate": 1 - T ** 2, "g_post": (1 - T) * (1 - T ** 2) / 2, "posterior": True}
    if method in ("js", "logistic"):
        f = m.a * cos + m.b; q = torch.sigmoid(f)
        return {"cos": cos, "f": f, "q": q, "score": f, "residual": 1 - q, "gate": q * (1 - q), "g_post": 1 - q, "posterior": True}
    # infonce: row softmax over the in-batch candidates supplied in extra (logits [B, B])
    logits = extra; p = torch.softmax(logits, 1).diagonal()
    return {"cos": cos, "f": cos / m.log_tau.exp(), "q": None, "score": cos / m.log_tau.exp(), "residual": 1 - p, "gate": p * (1 - p), "g_post": 1 - p, "posterior": False}


def split_parts(method, m, u, v, inj, v_pool=None):
    """(original loss, injected-anchor part, clean part, extra): parts sum to the loss (anchor-row attribution)."""
    B = len(u); w_inj = inj.float(); w_cln = 1 - w_inj
    if method in ("vcs", "js"):
        fp = m.a * (u * v).sum(-1) + m.b; fn = m.a * (u[:, None, :] * v_pool).sum(-1) + m.b
        if method == "vcs":
            tp, tn = torch.tanh(fp), torch.tanh(fn); per_p = -(tp - 0.5 * tp ** 2); per_n = -(-tn - 0.5 * tn ** 2)
            loss = vcs_loss(m, u, v, v_pool)[0]
        else:
            per_p = F.softplus(-fp); per_n = F.softplus(fn); loss = js_loss(m, u, v, v_pool)[0]
        part = lambda w: (per_p * w).sum() / B + (per_n * w[:, None]).sum() / per_n.numel()
        return loss, part(w_inj), part(w_cln), None
    if method == "infonce":
        logits = u @ v.T / m.log_tau.exp(); y = torch.arange(B, device=u.device)
        ce_r = F.cross_entropy(logits, y, reduction="none"); ce_c = F.cross_entropy(logits.T, y, reduction="none")
        loss = infonce_loss(m, u, v)[0]; part = lambda w: 0.5 * ((ce_r + ce_c) * w).sum() / B
        return loss, part(w_inj), part(w_cln), logits
    logits = m.a * (u @ v.T) + m.b; y = 2 * torch.eye(B, device=u.device) - 1; per = -F.logsigmoid(y * logits)
    loss = logistic_loss(m, u, v)[0]; part = lambda w: (per * w[:, None]).sum() / per.numel()
    return loss, part(w_inj), part(w_cln), logits


def train_instrumented(method, fit_img, fit_txt, lr, epochs, seed, device, inj_mask, K=8, batch=256, wd=1e-4, fit_partners=None):
    """robust_r2r3.train_r's loop, byte-for-byte in every RNG-consuming line, with the (A) instrumentation added (no RNG consumed).
    Returns the trained model, the step log, meta and the state dict at the end of epoch 1 (early checkpoint)."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Adapters(method).to(device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(fit_img); steps = epochs * (n // batch); step = 0; spe = n // batch
    fit_img, fit_txt = fit_img.to(device), fit_txt.to(device)

    def partner(ix_cpu, cands):
        return torch.tensor([c[int(torch.randint(0, len(c), (1,), generator=g))] if c else int(i) for i, c in zip(ix_cpu.tolist(), (cands[i] for i in ix_cpu.tolist()))])

    params = list(m.parameters()); theta0 = flat([p.detach() for p in params]).clone(); inj_mask = inj_mask.to(device)
    keys = ("lr", "g_inj", "g_cln", "cos_inj_cln", "dtheta", "proj_inj", "proj_cln", "gpost_inj", "gpost_cln", "du_inj", "du_cln", "score_inj", "score_cln", "n_inj")
    log = {k: [] for k in keys}; path_len = 0.0; early = None
    done = False
    while not done:
        perm = torch.randperm(n, generator=g)
        for s in range(0, n - batch + 1, batch):
            if step >= steps:
                done = True; break
            ix_cpu = perm[s: s + batch]; ix = ix_cpu.to(device); ci = torch.randint(0, 4, (batch,), generator=g).to(device)
            src = ix if fit_partners is None else partner(ix_cpu, fit_partners).to(device)
            xi, xt = fit_img[ix], fit_txt[src, ci]
            for pg in opt.param_groups:
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
            opt.zero_grad(); u, v = m.enc(xi, xt); u.retain_grad()
            inj = inj_mask[ix]
            if method in ("vcs", "js"):
                pool = torch.randint(0, n, (batch, K), generator=g).to(device); pc = torch.randint(0, 4, (batch, K), generator=g).to(device)
                v_pool = F.normalize(m.txt(fit_txt[pool, pc]), dim=-1)
                loss, l_inj, l_cln, extra = split_parts(method, m, u, v, inj, v_pool)
            else:
                loss, l_inj, l_cln, extra = split_parts(method, m, u, v, inj)
            # --- instrumentation (autograd.grad does not touch .grad or the RNG) ---
            with torch.no_grad():
                pt = positive_terms(method, m, u, v, extra)
            gi = flat([t if t is not None else torch.zeros_like(p) for t, p in zip(torch.autograd.grad(l_inj, params, retain_graph=True, allow_unused=True), params)]) if bool(inj.any()) else torch.zeros_like(theta0)
            gc = flat([t if t is not None else torch.zeros_like(p) for t, p in zip(torch.autograd.grad(l_cln, params, retain_graph=True, allow_unused=True), params)])
            loss.backward()
            du = u.grad.norm(dim=1).detach()
            before = flat([p.detach() for p in params]).clone()
            opt.step(); step += 1
            dth = flat([p.detach() for p in params]) - before; nd = float(dth.norm()); path_len += nd
            unit = -dth / max(nd, 1e-30)
            log["lr"].append(float(opt.param_groups[0]["lr"])); log["g_inj"].append(float(gi.norm())); log["g_cln"].append(float(gc.norm()))
            log["cos_inj_cln"].append(float((gi @ gc) / max(float(gi.norm() * gc.norm()), 1e-30)))
            log["dtheta"].append(nd); log["proj_inj"].append(float(gi @ unit)); log["proj_cln"].append(float(gc @ unit))
            sel = lambda t, mk: float(t[mk].mean()) if bool(mk.any()) else float("nan")
            log["gpost_inj"].append(sel(pt["g_post"], inj)); log["gpost_cln"].append(sel(pt["g_post"], ~inj))
            log["du_inj"].append(sel(du, inj)); log["du_cln"].append(sel(du, ~inj))
            log["score_inj"].append(sel(pt["score"], inj)); log["score_cln"].append(sel(pt["score"], ~inj)); log["n_inj"].append(int(inj.sum()))
            if step == spe:
                early = copy.deepcopy({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
        if step >= steps:
            done = True
    m.eval()
    theta_T = flat([p.detach() for p in params])
    if early is None:
        early = copy.deepcopy({k: v.detach().cpu().clone() for k, v in m.state_dict().items()})
    return m, {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}, {"steps": steps, "epochs": epochs, "lr_peak": lr, "batch": batch, "weight_decay": wd, "optimizer": "AdamW, cosine lr per step, grad_accumulation 1",
                                                                              "path_length": path_len, "net_displacement": float((theta_T - theta0).norm()), "early_checkpoint_step": min(spe, steps)}, early


@torch.no_grad()
def displacement(m, fit_img, fit_txt, inj_mask, raw, device) -> dict:
    """Feature displacement vs the identity adapter (u_init = L2(x)) and parameter displacement."""
    out = {}
    def feat(img, txt_rows):
        u0, v0 = F.normalize(img, dim=-1), F.normalize(txt_rows, dim=-1); u, v = m.enc(img.to(device), txt_rows.to(device)); u, v = u.cpu(), v.cpu()
        return {"img": summ((u - u0).norm(dim=1)), "img_cos": float((u * u0).sum(-1).mean()), "txt": summ((v - v0).norm(dim=1)), "txt_cos": float((v * v0).sum(-1).mean())}
    inj = inj_mask.bool()
    out["FIT_injected"] = feat(fit_img[inj], fit_txt[inj, 0]) if bool(inj.any()) else None
    out["FIT_clean"] = feat(fit_img[~inj], fit_txt[~inj, 0])
    for split in ("SRC-EVAL", "TGT-EVAL"):
        img, txt, _ = raw[split]; out[split] = feat(img, txt[:, 4])
    I = torch.eye(512)
    out["params"] = {"W_img_minus_I_F": float((m.img.weight.detach().cpu() - I).norm()), "b_img": float(m.img.bias.detach().norm()), "W_txt_minus_I_F": float((m.txt.weight.detach().cpu() - I).norm()),
                     "b_txt": float(m.txt.bias.detach().norm()), "scalars": {k: float(v) for k, v in m.named_parameters() if v.numel() == 1}}
    return out


def pair_metrics(m, method, fit_img, fit_txt, fixed_topic, wrong, inj_mask, device, K=8, seed=20260929) -> dict:
    """(B) and (D) on the fixed training pairing (caption 0).  u, v are leaves (adapter outputs, L2) so the gradients are w.r.t. the
    representations; the adapter parameters are frozen.  K-pool negatives with a fixed seed (same for early / final)."""
    n = len(fit_img); part = torch.tensor([wrong.get(i, int(fixed_topic[i]) if int(fixed_topic[i]) >= 0 else i) for i in range(n)])
    img, txt = fit_img.to(device), fit_txt[part, 0].to(device); inj = inj_mask.bool().to(device)
    m = copy.deepcopy(m).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    res = {"n_pairs": n, "n_injected": int(inj.sum())}; rows = {}
    B = 256; g = torch.Generator().manual_seed(seed); fit_txt_d = fit_txt.to(device)
    acc = {k: [] for k in ("cos", "score", "residual", "gate", "g_post", "q", "du_pos", "du_total", "dv", "swap_vcs", "swap_js", "neg_score", "neg_gate", "neg_gpost", "neg_swap_vcs", "neg_swap_js")}
    for s in range(0, n, B):
        xi, xt = img[s: s + B], txt[s: s + B]
        with torch.no_grad():
            u0, v0 = m.enc(xi, xt)
        u = u0.clone().requires_grad_(True); v = v0.clone().requires_grad_(True)
        extra = None; neg_total = None
        if method in ("vcs", "js"):
            pool = torch.randint(0, n, (len(xi), K), generator=g).to(device); pc = torch.randint(0, 4, (len(xi), K), generator=g).to(device)
            with torch.no_grad():
                v_pool = F.normalize(m.txt(fit_txt_d[pool, pc]), dim=-1)
            fn = m.a * (u[:, None, :] * v_pool).sum(-1) + m.b
            if method == "vcs":
                tn = torch.tanh(fn); qn = (1 + tn) / 2; acc["neg_score"].append(tn.detach().cpu()); acc["neg_gate"].append((1 - tn ** 2).detach().cpu())
                acc["neg_gpost"].append(((1 + tn) * (1 - tn ** 2) / 2).detach().cpu()); neg_total = (tn + 0.5 * tn ** 2).mean(1)
            else:
                qn = torch.sigmoid(fn); acc["neg_score"].append(fn.detach().cpu()); acc["neg_gate"].append((qn * (1 - qn)).detach().cpu())
                acc["neg_gpost"].append(qn.detach().cpu()); neg_total = F.softplus(fn).mean(1)
            Tn = 2 * qn - 1; acc["neg_swap_vcs"].append(((1 + Tn) * (1 - Tn ** 2) / 2).detach().cpu()); acc["neg_swap_js"].append(qn.detach().cpu())
            fp = m.a * (u * v).sum(-1) + m.b
            per = -(torch.tanh(fp) - 0.5 * torch.tanh(fp) ** 2) if method == "vcs" else F.softplus(-fp)
        elif method == "infonce":
            extra = u @ v.T / m.log_tau.exp(); y = torch.arange(len(xi), device=device); per = 0.5 * (F.cross_entropy(extra, y, reduction="none") + F.cross_entropy(extra.T, y, reduction="none"))
        else:
            extra = m.a * (u @ v.T) + m.b; per = -F.logsigmoid(extra.diagonal())
        gu_pos, gv_pos = torch.autograd.grad(per.sum(), (u, v), retain_graph=True)
        if neg_total is not None:
            gu_tot = torch.autograd.grad((per + neg_total).sum(), u)[0]
        else:
            gu_tot = gu_pos
        with torch.no_grad():
            pt = positive_terms(method, m, u, v, extra)
            for k in ("cos", "score", "residual", "gate", "g_post"):
                acc[k].append(pt[k].cpu())
            acc["du_pos"].append(gu_pos.norm(dim=1).cpu()); acc["du_total"].append(gu_tot.norm(dim=1).cpu()); acc["dv"].append(gv_pos.norm(dim=1).cpu())
            if pt["posterior"]:
                q = pt["q"]; T = 2 * q - 1; acc["q"].append(q.cpu()); acc["swap_vcs"].append(((1 - T) * (1 - T ** 2) / 2).cpu()); acc["swap_js"].append((1 - q).cpu())
    inj_c = inj.cpu()
    for k, v in acc.items():
        if not v:
            continue
        t = torch.cat(v, 0)
        if t.ndim == 2:   # negatives [n, K]: split by the anchor image, one value per anchor (mean over its K negatives) for the bootstrap
            ta = t.mean(1)
            rows[k] = {"injected": summ(t[inj_c]), "clean": summ(t[~inj_c]), "bootstrap_injected_minus_clean": bootstrap_diff(ta[inj_c].numpy(), ta[~inj_c].numpy())}
        else:
            rows[k] = {"injected": summ(t[inj_c]), "clean": summ(t[~inj_c]), "bootstrap_injected_minus_clean": bootstrap_diff(t[inj_c].numpy(), t[~inj_c].numpy())}
    if "swap_vcs" in rows:
        t_v, t_j = torch.cat(acc["swap_vcs"]), torch.cat(acc["swap_js"]); r = t_v / t_j.clamp_min(1e-12)
        rows["swap_ratio_vcs_over_js"] = {"injected": summ(r[inj_c]), "clean": summ(r[~inj_c]), "bootstrap_injected_minus_clean": bootstrap_diff(r[inj_c].numpy(), r[~inj_c].numpy()),
                                          "note": "ratio of the VCS-form to the JS-form posterior-logit gradient at the same q: equals the actual gate 1 - T^2 (local identity)"}
    res["by_split"] = rows; res["posterior_defined"] = method != "infonce"
    res["du_note"] = ("du_pos = |d(positive term)/du|; du_total adds the anchor's K negative terms (VCS / JS); for InfoNCE / logistic the per-anchor term already "
                      "includes the anchor's row and column (candidate) roles, so du_total = du_pos")
    return res


def per_epoch(log: dict, steps_per_epoch: int) -> dict:
    out = {}
    for k, v in log.items():
        if k == "n_inj":
            continue
        E = len(v) // max(steps_per_epoch, 1)
        out[k] = [float(np.nanmean(v[e * steps_per_epoch:(e + 1) * steps_per_epoch])) for e in range(E)] if E else []
    return out


def run(a) -> int:
    device = torch.device("cuda", 0) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    P72 = json.load(open(a.p72)); shift_dirs = dict(kv.split("=", 1) for kv in a.shift_dirs.split(","))
    ms = [float(x) for x in a.m.split(",")]; methods = [x for x in a.methods.split(",") if x in METHODS]; K = int(P72["settings"]["K"])
    ck = Path(a.ckpt_dir); ck.mkdir(parents=True, exist_ok=True)
    R = {"settings": {**vars(a), "device": str(device), "K": K}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "p72_json": a.p72,
         "posterior_convention": POSTERIOR_CONVENTION, "retraining_disclosure": "P72 adapters were never saved; every model here is retrained through robust_r2r3.train_r "
         "with P72's selected (lr, epochs), seed 0, the same topic partners and injection (sha256 checked)", "shifts": {}}
    t0 = time.time()
    for shift, fdir in shift_dirs.items():
        fd = Path(fdir); raw = {k: load_split(fd, k) for k in SPLITS}
        if a.smoke:
            raw = {k: (v[0][:a.smoke_n], v[1][:a.smoke_n], v[2][:a.smoke_n]) for k, v in raw.items()}
        fit_img, fit_txt, fit_ids = raw["SRC-FIT"]; cal_img, cal_txt, cal_ids = raw["SRC-CAL"]
        topic_fit, fixed_fit = topic_partners(fit_ids, a.index_dir, 20260927); topic_cal, _ = topic_partners(cal_ids, a.index_dir, 20260927)
        shash = int(hashlib.sha256(shift.encode()).hexdigest()[:8], 16); n = len(fit_img)
        S = {"n_fit": n, "fit_ids_sha256": hashlib.sha256(",".join(str(int(i)) for i in fit_ids).encode()).hexdigest(), "feature_dir": str(fd), "levels": {}}
        S["fit_ids_match_p72"] = (S["fit_ids_sha256"] == P72["shifts"][shift]["fit_ids_sha256"]) if shift in P72["shifts"] else None
        for m_frac in ms:
            cands, inj, wrong = inject(n, m_frac, topic_fit, [20260928, int(round(m_frac * 1000)), shash])
            inj_mask = torch.zeros(n, dtype=torch.bool); inj_mask[torch.as_tensor(inj, dtype=torch.long)] = True
            sha = hashlib.sha256(",".join(map(str, inj.tolist())).encode()).hexdigest()
            p72L = P72["shifts"].get(shift, {}).get("levels", {}).get(f"{m_frac:g}")
            L = {"m": m_frac, "n_injected": int(len(inj)), "injected_sha256": sha, "injected_sha256_matches_p72": (p72L["injected_sha256"] == sha) if p72L else None, "methods": {}}
            lab = torch.zeros(n); lab[torch.as_tensor(inj, dtype=torch.long)] = 1.0 if len(inj) else 0.0
            for method in methods:
                if p72L and method in p72L["methods"]:
                    sel = p72L["methods"][method]["selected"]; lr, ep = float(sel["lr"]), int(sel["epochs"]); p72s = p72L["methods"][method]["seeds"].get("0", {})
                else:
                    lr, ep, p72s = 1e-3, 5, {}
                if a.smoke:
                    ep = a.smoke_epochs
                t1 = time.time()
                # 1. reproduction through the original code path
                m_rep, cal = train_r(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, 0, device, K=K, fit_partners=cands, cal_partners=topic_cal)
                rep = {"lr": lr, "epochs": ep, "seed": 0, "cal_loss": cal, "p72_seed0": {k: p72s.get(k) for k in ("cal_loss", "r3_auroc")}, "train_seconds_original": time.time() - t1}
                for split in ("SRC-EVAL", "TGT-EVAL"):
                    img, txt, _ = raw[split]; _, _, _, u, v = pair_sets(m_rep, img, txt, device, 13, None); r1, r5 = retrieval(u, v)
                    rep[split] = {"R1": r1, "R5": r5, "p72_seed0_R1": (p72s.get(split) or {}).get("R1")}
                if len(inj):
                    sc = r3_scores(m_rep, method, fit_img, fit_txt, fixed_fit, wrong, device); rep["r3_auroc"] = auroc(sc, 1 - lab)
                torch.save({"state_dict": m_rep.state_dict(), "method": method, "shift": shift, "m": m_frac, "lr": lr, "epochs": ep, "seed": 0, "checkpoint": "final",
                            "injected_sha256": sha, "source": "retrained via robust_r2r3.train_r (P72 adapters were never saved)"}, ck / f"{shift}_m{m_frac:g}_{method}_seed0_final.pt")
                # 2. instrumented copy + QC sentinel + early checkpoint
                t2 = time.time()
                m_ins, log, meta, early = train_instrumented(method, fit_img, fit_txt, lr, ep, 0, device, inj_mask, K=K, fit_partners=cands)
                dmax = max(float((p.detach() - q.detach()).abs().max()) for p, q in zip(m_rep.parameters(), m_ins.parameters()))
                np.savez_compressed(ck / f"{shift}_m{m_frac:g}_{method}_seed0_trajectory.npz", **log)
                m_early = Adapters(method).to(device); m_early.load_state_dict(early); m_early.eval()
                torch.save({"state_dict": early, "method": method, "shift": shift, "m": m_frac, "lr": lr, "epochs": ep, "seed": 0, "checkpoint": f"early (step {meta['early_checkpoint_step']})",
                            "injected_sha256": sha, "source": "instrumented copy of robust_r2r3.train_r (identical RNG sequence)"}, ck / f"{shift}_m{m_frac:g}_{method}_seed0_early.pt")
                spe = n // 256
                traj = {**meta, "instrumented_vs_original_max_abs_param_diff": dmax, "steps_per_epoch": spe, "per_epoch_mean": per_epoch(log, spe), "train_seconds_instrumented": time.time() - t2,
                        "totals": {"mean_g_inj": float(np.nanmean(log["g_inj"])) if len(inj) else None, "mean_g_cln": float(np.nanmean(log["g_cln"])),
                                   "sum_proj_inj": float(np.nansum(log["proj_inj"])), "sum_proj_cln": float(np.nansum(log["proj_cln"])), "mean_cos_inj_cln": float(np.nanmean(log["cos_inj_cln"])) if len(inj) else None,
                                   "mean_lr": float(np.mean(log["lr"])), "n_steps": int(len(log["lr"])), "mean_dtheta": float(np.mean(log["dtheta"]))}}
                # 3. pair metrics (early + final), drift, loss swap (on the original-path model)
                fin = pair_metrics(m_rep, method, fit_img, fit_txt, fixed_fit, wrong, inj_mask, device, K=K)
                ear = pair_metrics(m_early, method, fit_img, fit_txt, fixed_fit, wrong, inj_mask, device, K=K)
                drift = displacement(m_rep, fit_img, fit_txt, inj_mask, raw, device)
                drift_early = displacement(m_early, fit_img, fit_txt, inj_mask, raw, device)
                L["methods"][method] = {"reproduction": rep, "trajectory": traj, "final_pairs": fin, "early_pairs": ear, "drift": drift, "drift_early": drift_early, "seconds": time.time() - t1}
                print(f"[{shift}] m={m_frac:g} {method} lr={lr:g} ep={ep}: SRC R@1 {rep['SRC-EVAL']['R1']:.3f} (P72 {rep['SRC-EVAL']['p72_seed0_R1']}) "
                      + (f"R3 {rep.get('r3_auroc', float('nan')):.3f} (P72 {p72s.get('r3_auroc')}) " if len(inj) else "")
                      + f"| param diff instr/orig {dmax:.2e} | net disp {meta['net_displacement']:.3f} path {meta['path_length']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            S["levels"][f"{m_frac:g}"] = L
        R["shifts"][shift] = S
    R["wall_seconds"] = time.time() - t0
    Path(a.out + ".json").parent.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(a.out + ".json", "w"), indent=1, default=float); write_markdown(R, a); print("->", a.out + ".md")
    return 0


def write_markdown(R, a):
    L = [f"# D1 — P72 adapter drift decomposition (retrained through robust_r2r3.train_r; P72 adapters were never saved) — {R['utc']}", "",
         "Per (shift, m, method): reproduction of P72 seed 0; trajectory totals from the instrumented copy (QC: max |Δparam| vs the original path);",
         "final-pair quantities by split (mean; injected / clean); drift.  q = posterior; g_post = |d loss / d posterior-logit| per positive pair;",
         "swap = the VCS-form vs JS-form gradient at the same q (local identity on one state, not a training comparison).", ""]
    for shift, S in R["shifts"].items():
        L += [f"## {shift} (n_fit {S['n_fit']}, fit ids match P72: {S['fit_ids_match_p72']})", "",
              "| m | method | lr / ep | SRC R@1 (P72) | TGT R@1 (P72) | R3 AUROC (P72) | QC |Δparam| | steps | mean |dθ| | Σ|dθ| / net | mean |g_inj| / |g_cln| | Σproj inj / cln | cos(g_inj,g_cln) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for mk, Lv in S["levels"].items():
            for method, D in Lv["methods"].items():
                r, t = D["reproduction"], D["trajectory"]; T = t["totals"]
                f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
                L.append(f"| {mk} | {method} | {r['lr']:g} / {r['epochs']} | {r['SRC-EVAL']['R1']:.3f} ({f(r['SRC-EVAL']['p72_seed0_R1'])}) | {r['TGT-EVAL']['R1']:.3f} ({f(r['TGT-EVAL']['p72_seed0_R1'])}) | "
                         f"{f(r.get('r3_auroc'))} ({f(r['p72_seed0'].get('r3_auroc'))}) | {t['instrumented_vs_original_max_abs_param_diff']:.1e} | {t['steps']} | {T['mean_dtheta']:.2e} | {t['path_length']:.3f} / {t['net_displacement']:.3f} | "
                         f"{f(T['mean_g_inj'])} / {T['mean_g_cln']:.4f} | {T['sum_proj_inj']:.3f} / {T['sum_proj_cln']:.3f} | {f(T['mean_cos_inj_cln'])} |")
        for stage_key, title in (("early_pairs", "early checkpoint (end of epoch 1)"), ("final_pairs", "final checkpoint")):
            L += ["", f"### {title}: pair quantities by split", "",
                  "| m | method | split | score | residual | gate | g_post | |dℓ⁺/du| | |dℓ/du| total | |dℓ⁺/dv| | swap VCS-form | swap JS-form | ratio (=gate) | neg gate | neg g_post | Δ(inj−clean) g_post [CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for mk, Lv in S["levels"].items():
                for method, D in Lv["methods"].items():
                    bs = D[stage_key]["by_split"]
                    for split in ("injected", "clean"):
                        if split == "injected" and float(mk) == 0:
                            continue
                        g = lambda k: (f"{bs[k][split]['mean']:.4f}" if k in bs and bs[k][split].get("n") else "—")
                        bd = bs["g_post"]["bootstrap_injected_minus_clean"] if "g_post" in bs else {"diff": None}
                        ci = (f"{bd['diff']:+.4f} [{bd['ci95'][0]:+.4f}, {bd['ci95'][1]:+.4f}]" if bd.get("diff") is not None else "—") if split == "injected" else ""
                        L.append(f"| {mk} | {method} | {split} | {g('score')} | {g('residual')} | {g('gate')} | {g('g_post')} | {g('du_pos')} | {g('du_total')} | {g('dv')} | {g('swap_vcs')} | {g('swap_js')} | {g('swap_ratio_vcs_over_js')} | {g('neg_gate')} | {g('neg_gpost')} | {ci} |")
        L += ["", "### drift (final; early in brackets)", "", "| m | method | |u−u₀| FIT injected | FIT clean | SRC-EVAL | TGT-EVAL | ‖W_img−I‖_F | ‖W_txt−I‖_F | scalars |", "|---|---|---|---|---|---|---|---|---|"]
        for mk, Lv in S["levels"].items():
            for method, D in Lv["methods"].items():
                dr, de = D["drift"], D["drift_early"]
                fi = lambda d, k: (f"{d[k]['img']['mean']:.3f}" if d.get(k) else "—")
                L.append(f"| {mk} | {method} | {fi(dr, 'FIT_injected')} ({fi(de, 'FIT_injected')}) | {fi(dr, 'FIT_clean')} ({fi(de, 'FIT_clean')}) | {fi(dr, 'SRC-EVAL')} ({fi(de, 'SRC-EVAL')}) | {fi(dr, 'TGT-EVAL')} ({fi(de, 'TGT-EVAL')}) | "
                         f"{dr['params']['W_img_minus_I_F']:.3f} ({de['params']['W_img_minus_I_F']:.3f}) | {dr['params']['W_txt_minus_I_F']:.3f} ({de['params']['W_txt_minus_I_F']:.3f}) | "
                         + ", ".join(f"{k}={v:.3f}" for k, v in dr["params"]["scalars"].items()) + " |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p72", default="reports/P72_robust_r2r3.json"); ap.add_argument("--out", required=True)
    ap.add_argument("--shift-dirs", default="animal=/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A,indoor_outdoor=/home/infres/yinwang/CS_QMI/outputs/P57_precheck_A_wave2/features_indoor_outdoor")
    ap.add_argument("--ckpt-dir", default="/home/infres/yinwang/CS_QMI/outputs/P93_D_diag/p72_adapters")
    ap.add_argument("--m", default="0,0.4"); ap.add_argument("--methods", default="vcs,js,infonce,logistic")
    ap.add_argument("--index-dir", default="/home/infres/yinwang/CS_QMI/data/coco_index"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=2000); ap.add_argument("--smoke-epochs", type=int, default=2)
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
