"""V8 Table C (V8-03 kernel_bridges): traditional CS / density-ratio routes on the four threshold events, on the untrained model's pooled image
features (scripts/v8/v8_train.py features).  One deterministic fit per route; one configuration per route jointly over the four thresholds.

    python -I scripts/v8/v8_kernels.py --features /projects/EEG-foundation-model/yinghao/outputs_v8/features --route K-NYSTROM --out DIR [--pca 64]

Per threshold k (H >= k, k = 2..5) on FIT images (natural weights w): P_k mass w y_k, Q_k mass w (1 - y_k), pi_k = sum w y_k / sum w.
  K-KDE      Gaussian KDE of P_k and Q_k (every FIT image a centre, equal bandwidth h = mult * median FIT distance): ell = log p - log q,
             T = tanh(ell / 2)
  K-RULSIF   relative ratio alpha = 1/2 on a Gaussian basis (centres drawn from the P_k mass): r in [0, 2], T = r - 1
  K-NYSTROM  squared regression of the balanced label (P -> +1, Q -> -1) on Nystrom features: o ~ eta, T = clip(o, -1, 1)
Natural tail u_k = pi_k (1 + T) / (1 + (2 pi_k - 1) T); mu = 1 + sum_k u_k (raw, possibly crossing tails) and mu_pava (tails projected onto a
non-increasing sequence by PAVA; secondary).  Grid: mult {0.25, 0.5, 1, 2, 4}, ridge {1e-6, 1e-4, 1e-2, 1}, centres {256, 1024, 4096}
(KDE: mult only); one directional edge extension (mult 0.125 / 8, ridge 1e-8 / 10, centres 8192).  Selection: CAL_SELECT prompt-macro
mean-rating MSE of the raw mu (native criterion recorded as secondary).  Labels: FIT / CAL_SELECT / CAL_PROB / DEV (EvalMuse roles); GenAI-Rank
features carry no label.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROLES_L = ("FIT", "CAL_SELECT", "CAL_PROB", "DEV"); ROLES_U = ("GENAI_RANK",)
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0); RIDGES = (1e-6, 1e-4, 1e-2, 1.0); CENTRES = (256, 1024, 4096)
EXT = {"mult": (0.125, 8.0), "ridge": (1e-8, 10.0), "centres": 8192}; SEED = 17071


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pava_decreasing(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    for n, row in enumerate(v):
        lv, wt = [], []
        for a in row:
            lv.append(float(a)); wt.append(1)
            while len(lv) > 1 and lv[-2] < lv[-1]:
                w = wt[-2] + wt[-1]; z = (lv[-2] * wt[-2] + lv[-1] * wt[-1]) / w; lv[-2:] = [z]; wt[-2:] = [w]
        out[n] = np.concatenate([np.repeat(a, w) for a, w in zip(lv, wt)])
    return out


def mse_prompt_macro(mu, idx):
    by = defaultdict(list)
    for m, r in zip(mu, idx):
        by[r["prompt_id"]].append((m - r["mean"]) ** 2)
    return float(np.mean([np.mean(v) for v in by.values()]))


class Data:
    def __init__(self, fdir: Path, device, pca: int = 0):
        import torch
        self.dev, self.r = device, {}
        for role in ROLES_L + ROLES_U:
            f = fdir / f"feats_{role}.npy"
            if f.exists():
                self.r[role] = {"x": np.load(f).astype(np.float64), "idx": jl(fdir / f"index_{role}.jsonl")}
        F = self.r["FIT"]; w = np.array([r["w"] for r in F["idx"]]); wn = w / w.sum(); mu = wn @ F["x"]; sd = np.sqrt(wn @ (F["x"] - mu) ** 2) + 1e-6
        for v in self.r.values():
            v["z"] = (v["x"] - mu) / sd
        if pca:
            Z = F["z"]; C = (Z * wn[:, None]).T @ Z; vals, vecs = np.linalg.eigh(C); P = vecs[:, ::-1][:, :pca]; self.pca_explained = float(vals[::-1][:pca].sum() / vals.sum())
            for v in self.r.values():
                v["z"] = v["z"] @ P
            s2 = np.sqrt(wn @ F["z"] ** 2) + 1e-6
            for v in self.r.values():
                v["z"] = v["z"] / s2
        for v in self.r.values():
            v["t"] = torch.tensor(v["z"], dtype=torch.float32, device=device)
        self.w = w; self.Y = np.array([r["tail"] for r in F["idx"]]); self.pi = (wn[:, None] * self.Y).sum(0)


def sqd(a, b):
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T).clamp(min=0)


def median_distance(D: Data):
    import torch
    rng = np.random.default_rng(SEED); i = rng.choice(len(D.r["FIT"]["t"]), size=min(4096, len(D.r["FIT"]["t"])), replace=False)
    s = D.r["FIT"]["t"][torch.tensor(i, device=D.dev)]; d = torch.sqrt(sqd(s, s)); iu = torch.triu_indices(len(s), len(s), 1, device=D.dev); return float(d[iu[0], iu[1]].median())


def to_tail(T, pi):
    return pi * (1 + T) / (1 + (2 * pi - 1) * T)


def kde_T(D: Data, role, h, k):
    """T_k on a role for one bandwidth (FIT images are the centres, weighted by the P_k / Q_k masses)."""
    import torch
    C = D.r["FIT"]["t"]; a = D.w * D.Y[:, k]; b = D.w * (1 - D.Y[:, k])
    la = torch.tensor(np.log(np.where(a > 0, a / a.sum(), 1e-300)), device=D.dev); lb = torch.tensor(np.log(np.where(b > 0, b / b.sum(), 1e-300)), device=D.dev)
    ma, mb = torch.tensor(a > 0, device=D.dev), torch.tensor(b > 0, device=D.dev); X = D.r[role]["t"]; out = []
    for i in range(0, len(X), 1024):
        K = -0.5 * sqd(X[i:i + 1024], C).double() / h ** 2
        lp = torch.logsumexp((K + la[None]).masked_fill(~ma[None], -math.inf), 1); lq = torch.logsumexp((K + lb[None]).masked_fill(~mb[None], -math.inf), 1)
        out.append(torch.tanh((lp - lq) / 2).cpu().numpy())
    return np.concatenate(out)


def basis_fit(D: Data, route, nc, h, ridges, k):
    """RuLSIF / Nystrom coefficient vectors for threshold k, one per ridge; returns a function role -> T for each ridge."""
    import torch
    a = D.w * D.Y[:, k]; b = D.w * (1 - D.Y[:, k]); rng = np.random.default_rng(SEED + k)
    mass = a if route == "K-RULSIF" else D.w; p = mass / mass.sum(); ci = rng.choice(len(p), size=min(nc, int((p > 0).sum())), replace=False, p=p)
    Cc = D.r["FIT"]["t"][torch.tensor(ci, device=D.dev)]
    if route == "K-NYSTROM":
        K = torch.exp(-0.5 * sqd(Cc, Cc).double() / h ** 2); eps = 1e-6 * float(K.diagonal().mean())
        vals, vecs = torch.linalg.eigh(K + eps * torch.eye(len(Cc), device=D.dev, dtype=torch.float64)); Wm = (vecs / torch.sqrt(vals)) @ vecs.T
        phi = lambda X: torch.exp(-0.5 * sqd(X, Cc).double() / h ** 2) @ Wm
    else:
        phi = lambda X: torch.exp(-0.5 * sqd(X, Cc).double() / h ** 2)
    Pf = phi(D.r["FIT"]["t"]); wa = torch.tensor(a / a.sum(), device=D.dev); wb = torch.tensor(b / b.sum(), device=D.dev)
    if route == "K-RULSIF":
        H = 0.5 * (Pf.T * wa) @ Pf + 0.5 * (Pf.T * wb) @ Pf; hv = Pf.T @ wa
    else:
        H = 0.5 * ((Pf.T * wa) @ Pf + (Pf.T * wb) @ Pf); hv = 0.5 * (Pf.T @ wa - Pf.T @ wb)
    thetas = {lam: torch.linalg.solve(H + lam * torch.eye(len(H), device=D.dev, dtype=torch.float64), hv) for lam in ridges}

    def T_of(role, lam):
        X = D.r[role]["t"]; o = np.concatenate([(phi(X[i:i + 8192]) @ thetas[lam]).cpu().numpy() for i in range(0, len(X), 8192)])
        return (np.clip(o, 0, 2) - 1, float(np.mean((o < 0) | (o > 2)))) if route == "K-RULSIF" else (np.clip(o, -1, 1), float(np.mean(np.abs(o) > 1)))
    return T_of


def evaluate_cfg(D: Data, route, cfg, h0, roles):
    """Tails for the four thresholds on each role for one configuration -> {role: (u_raw [n, 4], clip fraction)}."""
    res = {role: [] for role in roles}; clip = {role: [] for role in roles}
    for k in range(4):
        if route == "K-KDE":
            for role in roles:
                res[role].append(to_tail(kde_T(D, role, cfg["mult"] * h0, k), D.pi[k])); clip[role].append(0.0)
        else:
            T_of = basis_fit(D, route, cfg["centres"], cfg["mult"] * h0, [cfg["ridge"]], k)
            for role in roles:
                T, c = T_of(role, cfg["ridge"]); res[role].append(to_tail(T, D.pi[k])); clip[role].append(c)
    return {role: (np.stack(res[role], 1), float(np.mean(clip[role]))) for role in roles}


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(); ap.add_argument("--features", required=True); ap.add_argument("--route", required=True, choices=("K-KDE", "K-RULSIF", "K-NYSTROM"))
    ap.add_argument("--out", required=True); ap.add_argument("--pca", type=int, default=0); a = ap.parse_args(); t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; D = Data(Path(a.features), dev, a.pca); h0 = median_distance(D); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cal = D.r["CAL_SELECT"]["idx"]
    grid = [{"mult": m} for m in MULTS] if a.route == "K-KDE" else [{"mult": m, "ridge": r, "centres": c} for c in CENTRES for m in MULTS for r in RIDGES]
    def score(cfg):
        u, clip = evaluate_cfg(D, a.route, cfg, h0, ["CAL_SELECT"])["CAL_SELECT"]
        return {**cfg, "cal_select_mse": mse_prompt_macro(1 + u.sum(1), cal), "cal_select_mse_pava": mse_prompt_macro(1 + pava_decreasing(u).sum(1), cal),
                "crossing_fraction": float(np.mean(np.any(np.diff(u, axis=1) > 1e-12, axis=1))), "clip_fraction": clip}
    rows = [score(g) for g in grid]; sel = min(rows, key=lambda r: r["cal_select_mse"]); ext = []
    if sel["mult"] in (MULTS[0], MULTS[-1]):
        ext.append({**{k: sel[k] for k in grid[0]}, "mult": EXT["mult"][0] if sel["mult"] == MULTS[0] else EXT["mult"][1]})
    if a.route != "K-KDE":
        if sel["ridge"] in (RIDGES[0], RIDGES[-1]):
            ext.append({**{k: sel[k] for k in grid[0]}, "ridge": EXT["ridge"][0] if sel["ridge"] == RIDGES[0] else EXT["ridge"][1]})
        if sel["centres"] == CENTRES[-1]:
            ext.append({**{k: sel[k] for k in grid[0]}, "centres": EXT["centres"]})
    for e in ext:
        r = score(e); r["extension"] = True; rows.append(r)
    sel = min(rows, key=lambda r: r["cal_select_mse"]); roles = [r for r in D.r if r != "FIT"]
    U = evaluate_cfg(D, a.route, {k: sel[k] for k in grid[0]}, h0, roles); metrics = {}
    for role in roles:
        u, clip = U[role]; up = pava_decreasing(u); idx = D.r[role]["idx"]
        with open(out / f"pred_{role}.jsonl", "w") as fh:
            for r, ur, upr in zip(idx, u, up):
                fh.write(json.dumps({**{k: r[k] for k in ("image_id", "prompt_id", "generator_id", "candidate_index") if k in r}, "u_raw": ur.tolist(),
                                     "u_pava": upr.tolist(), "mu": float(1 + ur.sum()), "mu_pava": float(1 + upr.sum()), "score": float(1 + ur.sum())}) + "\n")
        if role in ROLES_L:
            metrics[role] = {"mean_mse_prompt_macro": mse_prompt_macro(1 + u.sum(1), idx), "mean_mse_prompt_macro_pava": mse_prompt_macro(1 + up.sum(1), idx),
                             "crossing_fraction": float(np.mean(np.any(np.diff(u, axis=1) > 1e-12, axis=1))), "clip_fraction": clip}
    rec = {"route": a.route, "pca": a.pca, "pca_explained": getattr(D, "pca_explained", None), "h0_median_distance": h0, "pi_fit": D.pi.tolist(), "grid": rows,
           "selected": sel, "extended": [e for e in ext], "metrics": metrics, "seconds": time.time() - t0, "device": dev}
    (out / "fit.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({"route": a.route, "selected": sel, "metrics": metrics}, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
