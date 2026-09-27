"""Second-application pre-check A — step 2: adapters on frozen CLIP features with three objectives; calibration and threshold transfer.

    python scripts/precheck_a_adapters.py --features <dir from step 1> --out <report prefix> [--smoke]

Methods (identical adapters: Linear(512→256, bias) per tower, L2-normalised outputs; batch 256 images, one of captions 0–3 per image per step;
caption 4 of every image is held out for evaluation):
  vcs       T = tanh(a·cos(u, v) + b), a₀ = 5, b₀ = 0; loss −J with positives (u_i, v_i) and K = 8 negatives per image whose captions come from
            an independent pool (random SRC-FIT images outside the batch); P and Q averaged separately (plan form).
  infonce   symmetric InfoNCE over in-batch pairs with a learnable temperature (init 0.07).
  logistic  SigLIP-style pairwise logistic over in-batch pairs: −log σ(y_ij (a·cos + b)), y = +1 matched / −1 otherwise, learnable a (init 10), b (init −10).
Grid per method: lr ∈ {1e-3, 3e-4} × epochs ∈ {5, 15, 40}; selection on SRC-CAL by the method's own loss (VCS: −J with CAL-pool negatives);
3 seeds of the selected configuration.  Evaluation on SRC-EVAL and TGT-EVAL with balanced pair sets: joint (image, its caption 4) and product
(image, caption 4 of another image by a fixed derangement).  Reported: reliability / ECE (15 equal-mass bins) / Brier for (1+T)/2, for σ(a·cos+b)
of the logistic model, for InfoNCE-cosine + Platt (fitted on SRC-CAL) and for VCS + Platt; threshold transfer (θ set on SRC-CAL at FNR 5 % / 10 %,
applied unchanged to SRC-EVAL / TGT-EVAL; drift of FNR and FPR); held-out J; retrieval R@1/R@5 (secondary); learned a, b; saturation.
"""
from __future__ import annotations

import argparse
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


def load_split(fd, name):
    d = torch.load(fd / "features" / f"{name}.pt", map_location="cpu", weights_only=False)
    return d["img"].float(), d["txt"].float()  # [n,512], [n,5,512]


class Adapters(nn.Module):
    def __init__(self, method, d_in=512, d_out=256):
        super().__init__()
        self.img = nn.Linear(d_in, d_out); self.txt = nn.Linear(d_in, d_out); self.method = method
        if method == "vcs":
            self.a = nn.Parameter(torch.tensor(5.0)); self.b = nn.Parameter(torch.tensor(0.0))
        elif method == "infonce":
            self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
        else:
            self.a = nn.Parameter(torch.tensor(10.0)); self.b = nn.Parameter(torch.tensor(-10.0))

    def enc(self, xi, xt):
        return F.normalize(self.img(xi), dim=-1), F.normalize(self.txt(xt), dim=-1)

    def score(self, u, v):  # pairwise score used for calibration / thresholds; u,v L2 [n,d]
        cos = (u * v).sum(-1)
        if self.method == "vcs":
            return torch.tanh(self.a * cos + self.b), cos
        if self.method == "infonce":
            return cos / self.log_tau.exp(), cos
        return self.a * cos + self.b, cos


def vcs_loss(m, u, v, v_pool):
    """u,v [B,d] positives; v_pool [B,K,d] independent-pool caption features (already adapted, normalised)."""
    tp = torch.tanh(m.a * (u * v).sum(-1) + m.b); tn = torch.tanh(m.a * (u[:, None, :] * v_pool).sum(-1) + m.b)
    J = (tp - 0.5 * tp ** 2).mean() + (-tn - 0.5 * tn ** 2).mean()
    return -J, {"J": float(J), "sat_pos": float((tp.abs() > 0.95).float().mean()), "sat_neg": float((tn.abs() > 0.95).float().mean())}


def infonce_loss(m, u, v):
    logits = u @ v.T / m.log_tau.exp(); y = torch.arange(len(u), device=u.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)), {}


def logistic_loss(m, u, v):
    logits = m.a * (u @ v.T) + m.b; y = 2 * torch.eye(len(u), device=u.device) - 1
    return -F.logsigmoid(y * logits).mean(), {}


def train(method, fit_img, fit_txt, cal_img, cal_txt, lr, epochs, seed, device, K=8, batch=256, wd=1e-4, log=None):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = Adapters(method).to(device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(fit_img); steps = epochs * (n // batch); step = 0
    fit_img, fit_txt = fit_img.to(device), fit_txt.to(device)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n - batch + 1, batch):
            ix = perm[s: s + batch]; ci = torch.randint(0, 4, (batch,), generator=g).to(device)   # captions 0-3 for training
            xi, xt = fit_img[ix], fit_txt[ix, ci]
            for pg in opt.param_groups:
                pg["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / max(steps, 1)))
            opt.zero_grad(); u, v = m.enc(xi, xt)
            if method == "vcs":
                pool = torch.randint(0, n, (batch, K), generator=g).to(device); pc = torch.randint(0, 4, (batch, K), generator=g).to(device)
                v_pool = F.normalize(m.txt(fit_txt[pool, pc]), dim=-1)
                loss, st = vcs_loss(m, u, v, v_pool)
            elif method == "infonce":
                loss, st = infonce_loss(m, u, v)
            else:
                loss, st = logistic_loss(m, u, v)
            loss.backward(); opt.step(); step += 1
    m.eval()
    # selection loss on SRC-CAL (method's own loss; VCS with CAL-pool negatives)
    with torch.no_grad():
        ci = torch.randint(0, 4, (len(cal_img),), generator=g); u, v = m.enc(cal_img.to(device), cal_txt[torch.arange(len(cal_img)), ci].to(device))
        if method == "vcs":
            pool = torch.randint(0, len(cal_img), (len(cal_img), K), generator=g); pc = torch.randint(0, 4, (len(cal_img), K), generator=g)
            v_pool = F.normalize(m.txt(cal_txt[pool, pc].to(device)), dim=-1); val, _ = vcs_loss(m, u, v, v_pool)
        elif method == "infonce":
            vals = []
            for s in range(0, len(u) - 255, 256):
                vals.append(float(infonce_loss(m, u[s: s + 256], v[s: s + 256])[0]))
            val = torch.tensor(np.mean(vals))
        else:
            vals = []
            for s in range(0, len(u) - 255, 256):
                vals.append(float(logistic_loss(m, u[s: s + 256], v[s: s + 256])[0]))
            val = torch.tensor(np.mean(vals))
    return m, float(val)


# --------------------------------------------------------------------------------------------------------------------- evaluation helpers
def ece(p, y, bins=15):
    """Expected calibration error with equal-mass bins; also max bin deviation and the reliability curve."""
    o = torch.argsort(p); p, y = p[o], y[o]; edges = torch.linspace(0, len(p), bins + 1).long(); e, mx, curve = 0.0, 0.0, []
    for i in range(bins):
        sl = slice(int(edges[i]), int(edges[i + 1]))
        if edges[i + 1] > edges[i]:
            conf, acc = float(p[sl].mean()), float(y[sl].mean()); w = (edges[i + 1] - edges[i]) / len(p)
            e += w * abs(conf - acc); mx = max(mx, abs(conf - acc)); curve.append((conf, acc))
    return float(e), float(mx), curve


def brier(p, y):
    return float(((p - y) ** 2).mean())


def platt(score_cal, y_cal, score):
    """1-D logistic regression (Platt) fitted on CAL scores; returns probabilities for `score`."""
    a, b = torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([a, b], max_iter=200)
    sc, yc = score_cal.double(), y_cal.double()
    def closure():
        opt.zero_grad(); l = F.binary_cross_entropy_with_logits(a.double() * sc + b.double(), yc); l.backward(); return l
    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(a.double() * score.double() + b.double()).float(), float(a), float(b)


def pair_sets(m, img, txt, device, seed):
    """Balanced joint/product pairs on held-out caption 4; product partner = caption 4 of a different image (fixed derangement)."""
    n = len(img); g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g); perm = torch.where(perm == torch.arange(n), (perm + 1) % n, perm)
    with torch.no_grad():
        u, v = m.enc(img.to(device), txt[:, 4].to(device)); u, v = u.cpu(), v.cpu()
        s_pos, c_pos = m.cpu().score(u, v); s_neg, c_neg = m.score(u, v[perm]); m.to(device)
    return torch.cat((s_pos, s_neg)), torch.cat((c_pos, c_neg)), torch.cat((torch.ones(n), torch.zeros(n))), u, v


def threshold_at_fnr(score_pos, fnr):
    return float(torch.quantile(score_pos, fnr))  # accept iff score >= theta -> FNR = fnr on this set


def retrieval(u, v):
    sims = u @ v.T; ranks = (sims > sims.diag()[:, None]).sum(1)
    return float((ranks < 1).float().mean()), float((ranks < 5).float().mean())


def heldout_J(m, img, txt, device, seed, K=8):
    n = len(img); g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        u, v = m.enc(img.to(device), txt[:, 4].to(device)); pool = torch.randint(0, n, (n, K), generator=g)
        v_pool = F.normalize(m.txt(txt[pool, 4].to(device)), dim=-1); loss, st = vcs_loss(m, u, v, v_pool)
    return -float(loss), st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", default="1e-3,3e-4"); ap.add_argument("--epochs", default="5,15,40"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fnr", default="0.05,0.10"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    fd = Path(a.features); splits = {k: load_split(fd, k) for k in ("SRC-FIT", "SRC-CAL", "SRC-EVAL", "TGT-EVAL")}
    lrs = [float(x) for x in a.lrs.split(",")]; epochs = [int(x) for x in a.epochs.split(",")]; seeds = [int(x) for x in a.seeds.split(",")]; fnrs = [float(x) for x in a.fnr.split(",")]
    if a.smoke:
        lrs, epochs, seeds = [1e-3], [1], [0]
        splits = {k: (v[0][:2000], v[1][:2000]) for k, v in splits.items()}
    fit_img, fit_txt = splits["SRC-FIT"]; cal_img, cal_txt = splits["SRC-CAL"]
    results = {"settings": vars(a), "device": str(device), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "methods": {}}
    t0 = time.time()
    for method in ("vcs", "infonce", "logistic"):
        grid = []
        for lr in lrs:
            for ep in epochs:
                m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, lr, ep, 0, device)
                grid.append({"lr": lr, "epochs": ep, "cal_loss": val}); print(f"[{method}] lr={lr:g} ep={ep}: CAL loss {val:.4f} ({time.time() - t0:.0f}s)", flush=True)
        best = min(grid, key=lambda r: r["cal_loss"]); res = {"grid": grid, "selected": best, "seeds": {}}
        for seed in seeds:
            m, val = train(method, fit_img, fit_txt, cal_img, cal_txt, best["lr"], best["epochs"], seed, device)
            r = {"cal_loss": val, "params": {k: float(v) for k, v in m.named_parameters() if v.numel() == 1}}
            # CAL pairs (for Platt and thresholds), EVAL pairs
            cal_s, cal_c, cal_y, _, _ = pair_sets(m, cal_img, cal_txt, device, 11)
            for split in ("SRC-EVAL", "TGT-EVAL"):
                img, txt = splits[split]; s_, c_, y_, u, v = pair_sets(m, img, txt, device, 13)
                row = {"retrieval_R1_R5": retrieval(u, v)}
                # probabilities: native (VCS (1+T)/2; logistic sigma; infonce: none) and Platt-calibrated cosine, Platt-calibrated native score
                probs = {}
                if method == "vcs":
                    probs["native_(1+T)/2"] = (1 + s_) / 2
                if method == "logistic":
                    probs["native_sigmoid"] = torch.sigmoid(s_)
                pc, pa, pb = platt(cal_c, cal_y, c_); probs["cosine+Platt(CAL)"] = pc
                ps, _, _ = platt(cal_s, cal_y, s_); probs["score+Platt(CAL)"] = ps
                row["calibration"] = {}
                for name, p in probs.items():
                    e, mx, curve = ece(p, y_); row["calibration"][name] = {"ECE": e, "max_dev": mx, "Brier": brier(p, y_), "curve": curve}
                # threshold transfer (thresholds set on CAL joint pairs)
                row["threshold"] = {}
                for fnr in fnrs:
                    for sname, cal_sc, ev_sc in (("score", cal_s, s_), ("cosine", cal_c, c_)):
                        th = threshold_at_fnr(cal_sc[cal_y == 1], fnr)
                        row["threshold"][f"{sname}@FNR{fnr:g}"] = {"theta": th, "FNR": float((ev_sc[y_ == 1] < th).float().mean()), "FPR": float((ev_sc[y_ == 0] >= th).float().mean())}
                if method == "vcs":
                    J, st = heldout_J(m, img, txt, device, 17); row["heldout_J"] = J; row["saturation"] = st
                r[split] = row
            res["seeds"][str(seed)] = r
            print(f"[{method}] seed {seed}: " + " | ".join(f"{sp}: R@1 {r[sp]['retrieval_R1_R5'][0]:.3f} ECE " + ", ".join(f"{k}={v['ECE']:.3f}" for k, v in r[sp]["calibration"].items()) for sp in ("SRC-EVAL", "TGT-EVAL")) + (f" | J {r['SRC-EVAL'].get('heldout_J', float('nan')):.3f}" if method == "vcs" else ""), flush=True)
        results["methods"][method] = res
    json.dump(results, open(a.out + ".json", "w"), indent=1, default=float)
    # markdown
    L = [f"# Pre-check A — calibration and threshold transfer on frozen CLIP features (COCO no-animal → animal) — {results['utc']}", "",
         "Adapters Linear(512→256)+L2 per tower; batch 256; captions 0–3 train, caption 4 evaluation; balanced joint/product pairs (product = caption 4 of another image). "
         "ECE: 15 equal-mass bins. Thresholds set on SRC-CAL joint pairs at the stated FNR and applied unchanged. Mean over seeds (SD in JSON).", ""]
    L += ["| method | selected (lr, ep) | split | R@1 | probability | ECE | max dev | Brier |", "|---|---|---|---|---|---|---|---|"]
    for method, res in results["methods"].items():
        for split in ("SRC-EVAL", "TGT-EVAL"):
            names = list(next(iter(res["seeds"].values()))[split]["calibration"].keys())
            for nm in names:
                E = np.mean([r[split]["calibration"][nm]["ECE"] for r in res["seeds"].values()]); M = np.mean([r[split]["calibration"][nm]["max_dev"] for r in res["seeds"].values()]); B = np.mean([r[split]["calibration"][nm]["Brier"] for r in res["seeds"].values()])
                R1 = np.mean([r[split]["retrieval_R1_R5"][0] for r in res["seeds"].values()])
                L.append(f"| {method} | {res['selected']['lr']:g}, {res['selected']['epochs']} | {split} | {R1:.3f} | {nm} | {E:.4f} | {M:.4f} | {B:.4f} |")
    L += ["", "| method | score | FNR target | θ (CAL) | SRC-EVAL FNR / FPR | TGT-EVAL FNR / FPR | drift |FNR| / |FPR| |", "|---|---|---|---|---|---|---|"]
    for method, res in results["methods"].items():
        keys = list(next(iter(res["seeds"].values()))["SRC-EVAL"]["threshold"].keys())
        for k in keys:
            def mean_of(split, f):
                return np.mean([r[split]["threshold"][k][f] for r in res["seeds"].values()])
            sf, sp, tf, tp_ = mean_of("SRC-EVAL", "FNR"), mean_of("SRC-EVAL", "FPR"), mean_of("TGT-EVAL", "FNR"), mean_of("TGT-EVAL", "FPR")
            L.append(f"| {method} | {k} | | {mean_of('SRC-EVAL', 'theta'):.3f} | {sf:.3f} / {sp:.3f} | {tf:.3f} / {tp_:.3f} | {abs(tf - sf):.3f} / {abs(tp_ - sp):.3f} |")
    if "vcs" in results["methods"]:
        L += ["", "VCS held-out J (SRC-EVAL / TGT-EVAL, mean over seeds): " + " / ".join(f"{np.mean([r[sp]['heldout_J'] for r in results['methods']['vcs']['seeds'].values()]):.3f}" for sp in ("SRC-EVAL", "TGT-EVAL"))
              + "; learned (a, b): " + ", ".join(f"seed {s}: a {r['params'].get('a', float('nan')):.2f} b {r['params'].get('b', float('nan')):.2f}" for s, r in results["methods"]["vcs"]["seeds"].items())]
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
