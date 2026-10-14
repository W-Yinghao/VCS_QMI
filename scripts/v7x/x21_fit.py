"""V7X-21: the common-feature fits of reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml on the frozen X20 features.
    neural --method {N-VCS-F,N-BCE-F,N-DV,N-NWJ} --lr LR --seed S --out DIR     one neural fit (30 epochs, CAL_SELECT epoch selection)
    kernel --method {K-KDE-S,K-RULSIF,K-NYSTROM} --out DIR                     one deterministic kernel fit over its grid (+ one edge extension)
    cs-native --out DIR                                                        A0: classical CS of the FIT P / Q KDEs per bandwidth (descriptive)
    all --out DIR                                                              tuning (seed 17, 3 lr) -> lr choice -> seeds 0-2; kernels; cs-native
Labels: FIT / CAL_SELECT / CAL_PROB only (the X20 index files of DEV and GENAI_RANK carry no y).  Every prediction is the balanced log ratio ell
and p_nat = sigmoid(ell + logit pi) with the FIT prior; kernel fits also keep their raw outputs (u, T)."""
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "v7"))
import x_estimators as XE  # noqa: E402
import v7_common as VC  # noqa: E402

LABELLED = ("FIT", "CAL_SELECT", "CAL_PROB"); UNLABELLED = ("DEV", "GENAI_RANK")
NEURAL = ("N-VCS-F", "N-BCE-F", "N-DV", "N-NWJ"); KERNEL = ("K-KDE-S", "K-RULSIF", "K-NYSTROM")
LR_GRID = (1e-4, 3e-4, 1e-3); EPOCHS, BATCH, HIDDEN, TUNE_SEED, FORMAL_SEEDS = 30, 4096, 512, 17, (0, 1, 2)
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0); RIDGES = (1e-6, 1e-4, 1e-2, 1.0); CENTRES = (256, 1024, 4096); EXT = {"mult": (0.125, 8.0), "ridge": (1e-8, 10.0), "centres": 8192}
GRID_SEED, H0_SAMPLE, ALPHA = 17071, 4096, 0.5


# ------------------------------------------------------------------------------------------------------------------------------- data
class Feats:
    """X20 features standardised with FIT-only D-weighted moments; D weights w = w_img / n_el; y only for the labelled roles."""

    def __init__(self, x20: Path, device, roles=LABELLED + UNLABELLED, pca=0):
        import torch
        self.dev, self.r = device, {}; man = json.loads((x20 / "manifest.json").read_text()); self.parts = {p["tag"]: p for p in man["parts"]}
        mr = x20 / "manifest_rank.json"
        if mr.exists():
            self.parts.update({p["tag"]: p for p in json.loads(mr.read_text())["parts"]})
        for role in roles:
            if not (x20 / f"feats_{role}.npy").exists():
                continue
            x = np.load(x20 / f"feats_{role}.npy").astype(np.float64); idx = VC.read_jsonl(x20 / f"index_{role}.jsonl")
            if len(idx) != len(x):
                raise SystemExit(f"{role}: index / feature length mismatch")
            if role in UNLABELLED and any("y" in r for r in idx):
                raise SystemExit(f"{role}: an unlabelled role carries y")
            w = np.array([r["w_img"] / r["n_el"] for r in idx]); y = np.array([r["y"] for r in idx]) if role in LABELLED else None
            self.r[role] = {"x": x, "w": w, "y": y, "idx": idx}
        f = self.r["FIT"]; wn = f["w"] / f["w"].sum(); self.mu = wn @ f["x"]; self.sd = np.sqrt(wn @ (f["x"] - self.mu) ** 2) + 1e-6
        for v in self.r.values():
            v["z"] = (v["x"] - self.mu) / self.sd; del v["x"]
        self.pca = pca
        if pca:                                                                # shared sensitivity (V7X_20): FIT-only D-weighted PCA, every method alike
            Z = f["z"]; C = (Z * wn[:, None]).T @ Z; vals, vecs = np.linalg.eigh(C); P = vecs[:, ::-1][:, :pca]
            self.pca_explained = float(vals[::-1][:pca].sum() / vals.sum())
            for v in self.r.values():
                v["z"] = v["z"] @ P
            s2 = np.sqrt(wn @ f["z"] ** 2) + 1e-6
            for v in self.r.values():
                v["z"] = v["z"] / s2
        for v in self.r.values():
            v["t"] = torch.tensor(v["z"], dtype=torch.float32, device=device)
        self.prior = float(wn @ f["y"])
        self.manifest_hash = hashlib.sha256("|".join(f"{k}:{self.parts[k]['sha256']}" for k in sorted(self.parts)).encode()).hexdigest()

    def brier(self, role, p):
        v = self.r[role]; ids = v["idx"]
        return VC.hierarchical_mean([r["prompt_id"] for r in ids], [r["image_id"] for r in ids], VC.expected_brier(p, v["y"]))[0]

    def masses(self, role):
        v = self.r[role]; return v["w"] * v["y"], v["w"] * (1 - v["y"])


def p_nat(ell, prior):
    return VC.natural_probability(ell, prior)


# ----------------------------------------------------------------------------------------------------------------------------- neural
def make_mlp(seed, d_in=768):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))


def neural_loss(method, f, y, w, prior):
    import torch
    import torch.nn.functional as F
    if method in ("N-VCS-F", "N-BCE-F"):
        pos, neg = y / (2 * prior), (1 - y) / (2 * (1 - prior))
        if method == "N-VCS-F":
            d = torch.sigmoid(f); terms = pos * (1 - d).square() + neg * d.square()
        else:
            terms = pos * F.softplus(-f) + neg * F.softplus(f)
        return (w * terms).sum() / w.sum()
    wp, wq = w * y, w * (1 - y); ep = (wp * f).sum() / wp.sum(); m = wq > 0
    if method == "N-DV":
        return -(ep - (torch.logsumexp(f[m] + torch.log(wq[m]), 0) - torch.log(wq.sum())))
    return -(ep - (wq * torch.exp(f - 1)).sum() / wq.sum())                     # N-NWJ


def native_bound(method, f, wp, wq, prior):
    """The method's own criterion on a labelled role (P / Q masses): VCS risk, balanced log-loss, DV bound, NWJ bound."""
    f = np.asarray(f, np.float64); a, b = wp / wp.sum(), wq / wq.sum()
    if method == "N-VCS-F":
        d = 1 / (1 + np.exp(-f)); return float(0.5 * a @ (1 - d) ** 2 + 0.5 * b @ d ** 2)
    if method == "N-BCE-F":
        return float(0.5 * a @ np.logaddexp(0, -f) + 0.5 * b @ np.logaddexp(0, f))
    if method == "N-DV":
        return float(a @ f - XE.dv_log_normaliser(f[b > 0], b[b > 0]))
    return float(a @ f - b @ np.exp(f - 1))


def bridge(method, f, F: Feats, fcal_prob=None):
    """f -> balanced log ratio ell (VCS / BCE: the logit; DV: f - log E_{Q,CAL_PROB} e^f, normaliser from fcal_prob; NWJ: f - 1)."""
    if method in ("N-VCS-F", "N-BCE-F"):
        return np.asarray(f, np.float64), None
    if method == "N-DV":
        _, wq = F.masses("CAL_PROB"); m = wq > 0; ln = XE.dv_log_normaliser(np.asarray(fcal_prob)[m], wq[m])
        return np.asarray(f, np.float64) - ln, ln
    return np.asarray(f, np.float64) - 1.0, None


def predict(model, t):
    import torch
    with torch.no_grad():
        return np.concatenate([model(t[i:i + 65536]).squeeze(-1).double().cpu().numpy() for i in range(0, len(t), 65536)])


def neural_fit(F: Feats, method, lr, seed, out: Path, epochs=EPOCHS):
    import torch
    t0 = time.time(); fit = F.r["FIT"]; dev = F.dev; X = fit["t"]; n = len(X)
    y = torch.tensor(fit["y"], dtype=torch.float32, device=dev); w = torch.tensor(fit["w"], dtype=torch.float32, device=dev)
    model = make_mlp(seed, X.shape[1]).to(dev); opt = torch.optim.Adam(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    n_params = sum(p.numel() for p in model.parameters()); best = (math.inf, None, None); log = []; nonfinite = 0
    wp_cs, wq_cs = F.masses("CAL_SELECT")
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev); model.train(); tl, gn, steps = 0.0, 0.0, 0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            if len(b) < BATCH // 4:
                continue
            loss = neural_loss(method, model(X[b]).squeeze(-1), y[b], w[b], F.prior)
            if not torch.isfinite(loss):
                nonfinite += 1; opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            gn += float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))); opt.step(); tl += float(loss.detach()); steps += 1
        model.eval(); f_cs = predict(model, F.r["CAL_SELECT"]["t"]); f_cp = predict(model, F.r["CAL_PROB"]["t"])
        ok = np.all(np.isfinite(f_cs)) and np.all(np.isfinite(f_cp))
        if ok:
            ell, ln = bridge(method, f_cs, F, f_cp); br = F.brier("CAL_SELECT", p_nat(ell, F.prior)); nb = native_bound(method, f_cs, wp_cs, wq_cs, F.prior)
        else:
            br, nb, ln = math.inf, float("nan"), None
        log.append({"epoch": ep, "train_loss": tl / max(steps, 1), "grad_norm_mean": gn / max(steps, 1), "steps": steps, "cal_select_natural_brier": br,
                    "cal_select_native": nb, "dv_log_normaliser": ln, "finite": bool(ok)})
        if br < best[0]:
            best = (br, ep, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2]); model.eval()
    rec = {"method": method, "lr": lr, "seed": seed, "epochs": epochs, "batch": BATCH, "n_params": n_params, "selected_epoch": best[1],
           "selected_cal_select_brier": best[0], "nonfinite_steps": nonfinite, "epochs_log": log, "prior": F.prior, "fit_seconds": time.time() - t0}
    preds = {role: predict(model, v["t"]) for role, v in F.r.items() if role != "FIT"}
    _, ln = bridge(method, preds["CAL_SELECT"], F, preds["CAL_PROB"]); rec["dv_log_normaliser_frozen"] = ln
    write_preds(F, out, method, {role: bridge(method, f, F, preds["CAL_PROB"])[0] for role, f in preds.items()}, {}, rec)
    return rec


# ---------------------------------------------------------------------------------------------------------------------------- kernels
def sqd(a, b):
    """Squared distances (torch, float32 inputs) clamped at 0."""
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T).clamp(min=0)


def h0_of(F: Feats):
    import torch
    rng = np.random.default_rng(GRID_SEED); i = rng.choice(len(F.r["FIT"]["t"]), size=min(H0_SAMPLE, len(F.r["FIT"]["t"])), replace=False)
    s = F.r["FIT"]["t"][torch.tensor(i, device=F.dev)]; d = torch.sqrt(sqd(s, s)); iu = torch.triu_indices(len(s), len(s), 1, device=F.dev)
    return float(d[iu[0], iu[1]].median())


def kde_logratio(F: Feats, mults, h0, roles):
    """log p_hat - log q_hat (equal bandwidths: constants cancel) and the per-class log densities, every FIT element a centre; blocks of 1,024."""
    import torch
    fit = F.r["FIT"]; C = fit["t"]; a, b = F.masses("FIT"); la = torch.tensor(np.log(np.where(a > 0, a / a.sum(), 1e-300)), device=F.dev)
    lb = torch.tensor(np.log(np.where(b > 0, b / b.sum(), 1e-300)), device=F.dev); ma, mb = torch.tensor(a > 0, device=F.dev), torch.tensor(b > 0, device=F.dev)
    d = C.shape[1]; out = {role: {m: {} for m in mults} for role in roles}; evals = 0
    for role in roles:
        Xq = F.r[role]["t"]; lp = {m: [] for m in mults}; lq = {m: [] for m in mults}
        for i in range(0, len(Xq), 1024):
            D2 = sqd(Xq[i:i + 1024], C).double(); evals += D2.numel()
            for m in mults:
                h = m * h0; K = -0.5 * D2 / h ** 2; const = -d * math.log(h) - 0.5 * d * math.log(2 * math.pi)
                lp[m].append((torch.logsumexp((K + la[None, :]).masked_fill(~ma[None, :], -math.inf), 1) + const).cpu().numpy())
                lq[m].append((torch.logsumexp((K + lb[None, :]).masked_fill(~mb[None, :], -math.inf), 1) + const).cpu().numpy())
        for m in mults:
            P, Q = np.concatenate(lp[m]), np.concatenate(lq[m]); out[role][m] = {"ell": P - Q, "logp": P, "logq": Q}
    return out, evals


def gauss_basis(X, Cc, h):
    return torch_exp(-0.5 * sqd(X, Cc) / h ** 2)


def torch_exp(t):
    import torch
    return torch.exp(t)


def draw_centres(F: Feats, nc, mass):
    rng = np.random.default_rng(GRID_SEED); p = mass / mass.sum(); nz = int((p > 0).sum())
    return rng.choice(len(p), size=min(nc, nz), replace=False, p=p)


def kernel_fit(F: Feats, method, out: Path):
    import torch
    t0 = time.time(); h0 = h0_of(F); a, b = F.masses("FIT"); wfit = F.r["FIT"]["w"]; roles = [r for r in F.r if r != "FIT"]; grid = []; extended = []
    if method == "K-KDE-S":
        mults = list(MULTS); res, evals = kde_logratio(F, mults, h0, ["CAL_SELECT"])
        def score(m):
            e = res["CAL_SELECT"][m]; wp, wq = F.masses("CAL_SELECT")
            return F.brier("CAL_SELECT", p_nat(e["ell"], F.prior)), float(wp @ e["logp"] / wp.sum() + wq @ e["logq"] / wq.sum())
        for m in mults:
            br, ll = score(m); grid.append({"mult": m, "h": m * h0, "cal_select_brier": br, "cal_select_loglik": ll})
        sel = min(grid, key=lambda g: g["cal_select_brier"])
        if sel["mult"] in (mults[0], mults[-1]):
            m2 = EXT["mult"][0] if sel["mult"] == mults[0] else EXT["mult"][1]; r2, e2 = kde_logratio(F, [m2], h0, ["CAL_SELECT"]); res["CAL_SELECT"][m2] = r2["CAL_SELECT"][m2]; evals += e2
            br, ll = score(m2); grid.append({"mult": m2, "h": m2 * h0, "cal_select_brier": br, "cal_select_loglik": ll, "extension": True}); extended.append(m2)
            sel = min(grid, key=lambda g: g["cal_select_brier"])
        nat = max(grid, key=lambda g: g["cal_select_loglik"]); full, e3 = kde_logratio(F, sorted({sel["mult"], nat["mult"]}), h0, roles); evals += e3
        preds = {role: full[role][sel["mult"]]["ell"] for role in roles}; native_preds = {role: full[role][nat["mult"]]["ell"] for role in roles}
        rec = {"selected": sel, "native_selected": nat, "kernel_evaluations": evals}
        write_preds(F, out / "native_cv", method + "-nativeCV", native_preds, {}, {"selected": nat, "note": "native KDE CV (CAL_SELECT log-likelihood)"})
        raw = {}
    else:
        best = None; best_nat = None; evals = 0
        def run_cfg(nc, m, ridges):
            nonlocal evals
            mass = a if method == "K-RULSIF" else wfit; ci = draw_centres(F, nc, mass); Cc = F.r["FIT"]["t"][torch.tensor(ci, device=F.dev)]; h = m * h0
            if method == "K-NYSTROM":
                K = torch.exp(-0.5 * sqd(Cc, Cc).double() / h ** 2); eps = 1e-6 * float(K.diagonal().mean()); vals, vecs = torch.linalg.eigh(K + eps * torch.eye(len(Cc), device=F.dev, dtype=torch.float64))
                Wm = (vecs / torch.sqrt(vals)) @ vecs.T; cond = float(vals.max() / vals.min())
                phi = lambda X: torch.exp(-0.5 * sqd(X, Cc).double() / h ** 2) @ Wm
            else:
                cond = None; phi = lambda X: torch.exp(-0.5 * sqd(X, Cc).double() / h ** 2)
            Pf = phi(F.r["FIT"]["t"]); evals += Pf.numel(); wa = torch.tensor(a / a.sum(), device=F.dev); wb = torch.tensor(b / b.sum(), device=F.dev)
            if method == "K-RULSIF":
                H = ALPHA * (Pf.T * wa) @ Pf + (1 - ALPHA) * (Pf.T * wb) @ Pf; hv = Pf.T @ wa
            else:
                H = 0.5 * ((Pf.T * wa) @ Pf + (Pf.T * wb) @ Pf); hv = 0.5 * (Pf.T @ wa - Pf.T @ wb)
            Pcs = phi(F.r["CAL_SELECT"]["t"]); wp, wq = F.masses("CAL_SELECT"); A, B = wp / wp.sum(), wq / wq.sum(); rows = []
            for lam in ridges:
                Mx = H + lam * torch.eye(len(H), device=F.dev, dtype=torch.float64); th = torch.linalg.solve(Mx, hv); resid = float(torch.linalg.norm(Mx @ th - hv) / torch.linalg.norm(hv))
                o = (Pcs @ th).cpu().numpy()
                if method == "K-RULSIF":
                    r = XE.relative_half_readout(o, F.prior); nat = float(0.5 * (ALPHA * A @ o ** 2 + (1 - ALPHA) * B @ o ** 2) - A @ o)
                else:
                    T = np.clip(o, -1, 1); d = (T + 1) / 2; r = {"p_nat": F.prior * d / (F.prior * d + (1 - F.prior) * (1 - d)), "clip_fraction": float(np.mean(o != T))}
                    nat = float(0.5 * A @ (1 - o) ** 2 + 0.5 * B @ (-1 - o) ** 2)
                rows.append({"centres": len(ci), "mult": m, "h": h, "ridge": lam, "cal_select_brier": F.brier("CAL_SELECT", r["p_nat"]), "cal_select_native": nat,
                             "clip_fraction": r["clip_fraction"], "solve_residual": resid, "cond_KCC": cond, "_th": th, "_ci": ci})
            return rows
        for nc in CENTRES:
            for m in MULTS:
                grid += run_cfg(nc, m, RIDGES)
        sel = min(grid, key=lambda g: g["cal_select_brier"]); edges = []
        if sel["mult"] in (MULTS[0], MULTS[-1]):
            edges.append(("mult", EXT["mult"][0] if sel["mult"] == MULTS[0] else EXT["mult"][1]))
        if sel["ridge"] in (RIDGES[0], RIDGES[-1]):
            edges.append(("ridge", EXT["ridge"][0] if sel["ridge"] == RIDGES[0] else EXT["ridge"][1]))
        if sel["centres"] == max(CENTRES) and len(F.r["FIT"]["t"]) > max(CENTRES):
            edges.append(("centres", EXT["centres"]))
        for kind, v in edges:                                                    # one pre-declared extension per edge, in that direction only
            nc, m, lam = sel["centres"], sel["mult"], sel["ridge"]; nc, m, lam = (v, m, lam) if kind == "centres" else (nc, v, lam) if kind == "mult" else (nc, m, v)
            new = run_cfg(nc, m, [lam]); [g.update(extension=kind) for g in new]; grid += new; extended.append(kind)
        sel = min(grid, key=lambda g: g["cal_select_brier"]); nat_sel = min(grid, key=lambda g: g["cal_select_native"])
        def outputs(cfg):
            ci = cfg["_ci"]; Cc = F.r["FIT"]["t"][torch.tensor(ci, device=F.dev)]; h = cfg["h"]
            if method == "K-NYSTROM":
                K = torch.exp(-0.5 * sqd(Cc, Cc).double() / h ** 2); eps = 1e-6 * float(K.diagonal().mean()); vals, vecs = torch.linalg.eigh(K + eps * torch.eye(len(Cc), device=F.dev, dtype=torch.float64))
                Wm = (vecs / torch.sqrt(vals)) @ vecs.T; phi = lambda X: torch.exp(-0.5 * sqd(X, Cc).double() / h ** 2) @ Wm
            else:
                phi = lambda X: torch.exp(-0.5 * sqd(X, Cc).double() / h ** 2)
            o = {role: np.concatenate([(phi(F.r[role]["t"][i:i + 8192]) @ cfg["_th"]).cpu().numpy() for i in range(0, len(F.r[role]["t"]), 8192)]) for role in roles}
            if method == "K-RULSIF":
                ell = {r_: np.log(np.clip(v, 1e-12, 2 - 1e-12)) - np.log(np.clip(2 - v, 1e-12, 2 - 1e-12)) for r_, v in o.items()}     # d = u/2 -> logit d
            else:
                ell = {r_: np.log(np.clip(1 + v, 1e-12, 2 - 1e-12)) - np.log(np.clip(1 - v, 1e-12, 2 - 1e-12)) for r_, v in o.items()}    # d = (T+1)/2
            return ell, o
        preds, raw = outputs(sel); npreds, nraw = outputs(nat_sel)
        clean = lambda g: {k: v for k, v in g.items() if not k.startswith("_")}
        write_preds(F, out / "native_cv", method + "-nativeCV", npreds, nraw, {"selected": clean(nat_sel), "note": "native criterion on CAL_SELECT"})
        grid = [clean(g) for g in grid]; rec = {"selected": clean(sel), "native_selected": clean(nat_sel), "kernel_evaluations": evals}
    rec.update({"method": method, "h0_median_distance": h0, "grid": grid, "extended": extended, "prior": F.prior, "fit_seconds": time.time() - t0})
    write_preds(F, out, method, preds, raw, rec)
    return rec


def cs_native(F: Feats, out: Path):
    """A0: -2 log C + log A + log B of the FIT P / Q KDEs (equal bandwidth: normalising constants cancel); diagonal pairs included."""
    import torch
    t0 = time.time(); h0 = h0_of(F); X = F.r["FIT"]["t"]; a, b = F.masses("FIT"); wa, wb = torch.tensor(a / a.sum(), device=F.dev), torch.tensor(b / b.sum(), device=F.dev)
    acc = {m: {"A": [], "B": [], "C": []} for m in MULTS}
    for i in range(0, len(X), 2048):
        D2 = sqd(X[i:i + 2048], X).double()
        for m in MULTS:
            K = -0.5 * D2 / (2 * (m * h0) ** 2)                                  # convolution bandwidth sqrt(2) h
            la, lb = torch.log(wa.clamp(min=1e-300)), torch.log(wb.clamp(min=1e-300))
            acc[m]["A"].append(float(torch.logsumexp((K + la[None, :] + la[i:i + 2048, None]).flatten(), 0)))
            acc[m]["B"].append(float(torch.logsumexp((K + lb[None, :] + lb[i:i + 2048, None]).flatten(), 0)))
            acc[m]["C"].append(float(torch.logsumexp((K + lb[None, :] + la[i:i + 2048, None]).flatten(), 0)))
    from scipy.special import logsumexp
    rows = []
    for m in MULTS:
        A, B, C = (float(logsumexp(acc[m][k])) for k in "ABC"); rows.append({"mult": m, "h": m * h0, "D_cs_sqratio": XE.SCALE_FACTOR * (-2 * C + A + B), "logA": A, "logB": B, "logC": C})
    rec = {"panel": "A0_native_CS", "method_id": "K-CS-NATIVE", "scale_factor": XE.SCALE_FACTOR, "rows": rows, "truth": None,
           "note": "natural features: no truth; descriptive only; not a probability row", "feature_manifest_hash": F.manifest_hash, "seconds": time.time() - t0}
    out.mkdir(parents=True, exist_ok=True); (out / "cs_native.json").write_text(json.dumps(rec, indent=1)); return rec


# ----------------------------------------------------------------------------------------------------------------------------- outputs
def write_preds(F: Feats, out: Path, method, ell_by_role, raw_by_role, rec):
    out.mkdir(parents=True, exist_ok=True)
    for role, ell in ell_by_role.items():
        ids = F.r[role]["idx"]; pn = p_nat(ell, F.prior); raw = raw_by_role.get(role)
        with open(out / f"pred_{role}.jsonl", "w") as fh:
            for k, r in enumerate(ids):
                o = {"element_id": r["element_id"], "image_id": r["image_id"], "prompt_id": r["prompt_id"], "ell": float(ell[k]), "p_nat": float(pn[k])}
                if raw is not None:
                    o["raw"] = float(raw[k])
                for c in ("generator_id", "candidate_index"):
                    if c in r:
                        o[c] = r[c]
                fh.write(json.dumps(o) + "\n")
        if role == "GENAI_RANK":
            write_rank(F, out, ell, pn)
    rec = {**rec, "feature_manifest_hash": F.manifest_hash, "pq_id": "evalmuse_element_softvote_v1", "standardization": "FIT-only D-weighted",
           "pca": F.pca, "pca_explained": getattr(F, "pca_explained", None)}
    (out / "log.json").write_text(json.dumps(rec, indent=1, default=float))


def write_rank(F: Feats, out: Path, ell, pn):
    """Per candidate: E = mean p_nat over its parser elements; candidates without elements -> decomposition_failed (no score)."""
    imgs = VC.read_jsonl(F.x20 / "images_GENAI_RANK.jsonl") if hasattr(F, "x20") else []
    per = {}
    for k, r in enumerate(F.r["GENAI_RANK"]["idx"]):
        per.setdefault(r["image_id"], []).append(float(pn[k]))
    with open(out / "genai_rank.jsonl", "w") as fh:
        for im in imgs:
            ps = per.get(im["image_id"], [])
            fh.write(json.dumps({"prompt_id": im["prompt_id"], "generator_id": im["generator_id"], "image_id": im["image_id"], "candidate_index": im["candidate_index"],
                                 "E": (float(np.mean(ps)) if ps else None), "n_elements": len(ps), "decomposition_failed": not ps, "parser_status": im.get("parser_status")}) + "\n")


# --------------------------------------------------------------------------------------------------------------------------------- cli
def load(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; F = Feats(Path(a.x20), dev, pca=a.pca); F.x20 = Path(a.x20)
    print(f"[x21] features {sorted(F.r)}; FIT prior {F.prior:.5f}; manifest {F.manifest_hash[:16]}; device {dev}", flush=True); return F


def cmd_all(a):
    F = load(a); out = Path(a.out); summary = {"tuning": {}, "chosen_lr": {}, "formal": {}, "kernels": {}}
    epochs = a.epochs
    for mth in NEURAL:
        for lr in LR_GRID:
            r = neural_fit(F, mth, lr, TUNE_SEED, out / "tune" / f"{mth}_lr{lr:g}", epochs); summary["tuning"][f"{mth}_lr{lr:g}"] = r["selected_cal_select_brier"]
            print(f"[x21] tune {mth} lr {lr:g}: CAL_SELECT Brier {r['selected_cal_select_brier']:.5f} (epoch {r['selected_epoch']}, {r['fit_seconds']:.0f}s)", flush=True)
        best = min(LR_GRID, key=lambda lr: (summary["tuning"][f"{mth}_lr{lr:g}"], lr)); summary["chosen_lr"][mth] = best
        for s in FORMAL_SEEDS:
            r = neural_fit(F, mth, best, s, out / "formal" / f"{mth}_s{s}", epochs); summary["formal"][f"{mth}_s{s}"] = r["selected_cal_select_brier"]
            print(f"[x21] formal {mth} s{s} lr {best:g}: CAL_SELECT Brier {r['selected_cal_select_brier']:.5f}", flush=True)
    for mth in KERNEL:
        r = kernel_fit(F, mth, out / "kernels" / mth); summary["kernels"][mth] = r["selected"]
        print(f"[x21] {mth}: selected {r['selected']} extended {r['extended']} ({r['fit_seconds']:.0f}s)", flush=True)
    cs = cs_native(F, out / "a0"); summary["cs_native"] = cs["rows"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float)); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("neural", "kernel", "cs-native", "all"):
        p = sub.add_parser(name); p.add_argument("--x20", default=str(VC.BIG_OUT / "x20")); p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--pca", type=int, default=0)
        if name == "neural":
            p.add_argument("--method", choices=NEURAL, required=True); p.add_argument("--lr", type=float, required=True); p.add_argument("--seed", type=int, required=True)
        if name == "kernel":
            p.add_argument("--method", choices=KERNEL, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "all":
        return cmd_all(a)
    F = load(a)
    if a.cmd == "neural":
        neural_fit(F, a.method, a.lr, a.seed, Path(a.out), a.epochs)
    elif a.cmd == "kernel":
        kernel_fit(F, a.method, Path(a.out))
    else:
        cs_native(F, Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
