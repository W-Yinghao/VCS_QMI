"""I1 — paired conditional nuisance audit on cached frozen features (package v4 §I1).

Presence / retention: is H_l ⊥ N | Y violated at layer l?  Per repeat (fresh draws; sampling unit = base image ID):
    FIT  n base images from the FIT split, EVAL n from the EVAL split, POOL n from the VAL split (disjoint by construction);
    N | Y ~ Bernoulli(½ ± 0.3 by class parity) drawn afresh; the observed feature is the planted version iff N = 1 (mode 'planted'),
    clean for all ('null_label_only'), or the planted version for all ('null_all_planted'; N independent of the image given Y).
    Product negatives: N of a same-class POOL item (T1's within_class_pool).  Per layer: features standardised with FIT-sample statistics;
    statistics = the exact linear critics of P105: VCS closed form (`closed_form_critic`) and the exact JS solve (`exact_js_critic`),
    both fitted on FIT only.  B within-class permutations of N on EVAL, shared by every layer and statistic.
    Per-layer p-values, and family-wise adjusted p-values across the predefined layer set by the max-statistic over layers of the
    permutation-standardised statistic (same permutation index for every layer).
Prediction effect (no test): on all EVAL base images, clean vs planted version of the same image: change of the true-class probability,
the margin (true logit − best other) and the accuracy, per class and overall, bootstrap CI by base image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from cond_test_t1_ablation import exact_js_critic  # noqa: E402
from cond_test_t1 import js_value  # noqa: E402
from precheck_d_tests import closed_form_critic, j_stat, within_class_pool  # noqa: E402

AUDIT_LAYERS = ("layer3", "h", "z", "logits")
P_SHIFT = 0.3


def p_n1(y: np.ndarray) -> np.ndarray:
    return 0.5 + P_SHIFT * np.where(np.asarray(y) % 2 == 0, 1.0, -1.0)


class AuditData:
    def __init__(self, path: Path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.run, self.y, self.part = d["run"], d["y"].numpy(), d["part"].numpy()
        self.F = d["features"]
        self.idx = {k: np.where(self.part == i)[0] for i, k in enumerate(("fit", "val", "eval"))}

    def feats(self, version: str, layer: str, rows: np.ndarray) -> torch.Tensor:
        return self.F[version][layer][torch.as_tensor(rows)].float()


def draw(D: AuditData, split: str, n: int, rng, mode: str, version: str, layers=AUDIT_LAYERS):
    rows = rng.choice(D.idx[split], size=n, replace=False); y = D.y[rows]
    nn_ = (rng.random(n) < p_n1(y)).astype(np.int64)
    X = {}
    for l in layers:
        c = D.feats("clean", l, rows)
        if mode == "planted":
            p = D.feats(version, l, rows); X[l] = torch.where(torch.as_tensor(nn_ == 1)[:, None], p, c)
        elif mode == "null_label_only":
            X[l] = c
        elif mode == "null_all_planted":
            X[l] = D.feats(version, l, rows)
        else:
            raise ValueError(mode)
    return X, y, nn_


def run_repeat(D: AuditData, n: int, rng, *, mode: str, version: str, perms: int, seed: int, device, layers=AUDIT_LAYERS, delta: float = 0.05) -> dict:
    Xf, Yf, Nf = draw(D, "fit", n, rng, mode, version, layers); Xe, Ye, Ne = draw(D, "eval", n, rng, mode, version, layers)
    _, Yp, Np = draw(D, "val", n, rng, mode, version, ())
    nf = torch.as_tensor(Nf).to(device); ne = torch.as_tensor(Ne).to(device)
    nf_neg = torch.as_tensor(within_class_pool(Yf, Yp, Np, rng)).to(device); ne_neg = torch.as_tensor(within_class_pool(Ye, Yp, Np, rng)).to(device)
    perm_N = []
    for _ in range(perms):
        pn = Ne.copy()
        for c in np.unique(Ye):
            ii = np.where(Ye == c)[0]; pn[ii] = pn[ii][rng.permutation(len(ii))]
        perm_N.append(torch.as_tensor(pn).to(device))
    out = {"n": n, "n1_frac_eval": float(Ne.mean()), "layers": {}}
    stats = {"vcs_closed": {}, "js_exact": {}}
    for l in layers:
        mu, sd = Xf[l].mean(0), Xf[l].std(0) + 1e-6
        zf, ze = ((Xf[l] - mu) / sd).to(device), ((Xe[l] - mu) / sd).to(device)
        cv = closed_form_critic(zf, nf, nf_neg, seed); cj = exact_js_critic(zf, nf, nf_neg, seed + 3)
        with torch.no_grad():
            tn = torch.tanh(cv(ze, ne_neg)); obs_v = j_stat(torch.tanh(cv(ze, ne)), tn); null_v = np.array([j_stat(torch.tanh(cv(ze, pn)), tn) for pn in perm_N])
            fneg = cj(ze, ne_neg); obs_j = js_value(cj(ze, ne), fneg); null_j = np.array([js_value(cj(ze, pn), fneg) for pn in perm_N])
        stats["vcs_closed"][l] = (obs_v, null_v); stats["js_exact"][l] = (obs_j, null_j)
    for s, per in stats.items():
        # max-statistic across layers on permutation-standardised statistics (same permutation index for every layer)
        zobs, znull = {}, []
        for l in layers:
            obs, null = per[l]; m, sd = null.mean(), null.std() + 1e-12
            zobs[l] = (obs - m) / sd; znull.append((null - m) / sd)
        zmax = np.max(np.stack(znull), axis=0)
        for l in layers:
            obs, null = per[l]
            p = float((1 + (null >= obs).sum()) / (1 + len(null))); p_adj = float((1 + (zmax >= zobs[l]).sum()) / (1 + len(zmax)))
            out["layers"].setdefault(l, {})[s] = {"stat": float(obs), "p": p, "reject": p <= delta, "p_maxT": p_adj, "reject_maxT": p_adj <= delta}
        out[f"{s}_any_layer_reject_maxT"] = any(out["layers"][l][s]["reject_maxT"] for l in layers)
    return out


def prediction_effects(D: AuditData, version: str, boot: int = 1000, seed: int = 0) -> dict:
    """Paired clean vs planted on every EVAL base image: Δ p(true class), Δ margin, accuracy clean / planted; per class and overall."""
    rows = D.idx["eval"]; y = torch.as_tensor(D.y[rows])
    lc, lp = D.F["clean"]["logits"][rows], D.F[version]["logits"][rows]
    pc, pp = lc.softmax(1).gather(1, y[:, None]).squeeze(1), lp.softmax(1).gather(1, y[:, None]).squeeze(1)

    def margin(lg):
        t = lg.gather(1, y[:, None]).squeeze(1); o = lg.clone(); o[torch.arange(len(y)), y] = -float("inf"); return t - o.max(1).values
    dprob = (pp - pc).numpy(); dmar = (margin(lp) - margin(lc)).numpy()
    ac, apl = (lc.argmax(1) == y).float().numpy(), (lp.argmax(1) == y).float().numpy()
    flip = (lc.argmax(1) != lp.argmax(1)).float().numpy()
    rng = np.random.default_rng(seed); B = rng.integers(0, len(rows), size=(boot, len(rows)))

    def ci(v):
        bs = v[B].mean(1); return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res = {"n_eval_base_images": int(len(rows)), "overall": {"d_prob_true": float(dprob.mean()), "d_prob_true_ci": ci(dprob), "d_margin": float(dmar.mean()),
           "d_margin_ci": ci(dmar), "acc_clean": float(ac.mean()), "acc_planted": float(apl.mean()), "d_acc": float((apl - ac).mean()),
           "d_acc_ci": ci(apl - ac), "prediction_flip_rate": float(flip.mean())}, "per_class": {}}
    yy = y.numpy()
    for c in range(10):
        m = yy == c
        res["per_class"][c] = {"n": int(m.sum()), "d_prob_true": float(dprob[m].mean()), "d_margin": float(dmar[m].mean()),
                               "acc_clean": float(ac[m].mean()), "acc_planted": float(apl[m].mean()), "flip": float(flip[m].mean())}
    return res
