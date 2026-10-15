"""V9-N2 reliability weights (package §03.B–D, §04.2–04.4, §05.3): fixed teacher score s -> critic T on the A-critic-fit role (P = original
pairs, Q = the specified re-pairing with donors from the same role) -> selection on A-critic-select by the common J -> reference moments
a = E_P T, b = E_Q T on A-reference-moments -> rho_hat = (a - c)/(a - b) from the unlabeled observed U pairs (transductive; interval from the
package's Hoeffding inversion on image-level cluster means, whose independence contract is the role-disjointness check) -> w = clean posterior
(1 - rho)(1 + T) / ((1 - rho)(1 + T) + rho(1 - T)) computed as sigma(logit(1 - rho) + 2 f).

Methods: N3 VCS-Weight (cos-2 critic, quadratic rule), N4 BCE-Weight (same class, matched logistic), N5 RuLSIF-Weight / N6 Nystrom-Weight / KDE
(imported from the N1 module `v9.quant` when present; placeholders raise until then), N7 RankMass (top-k by raw score with N3's total mass),
plus the plain-filter baseline (non-main row).  The package reference (`v9/package_v9_20261016/reference`) is loaded by file path because the
repository already has a top-level `reference` package.  The weight file is locked (sha256) and verified by the trainer before any step.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PKG = Path(__file__).resolve().parents[3] / "v9" / "package_v9_20261016" / "reference"


def _load(name: str):
    import sys
    key = f"v9_pkg_ref_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _PKG / f"{name}.py"); mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod                          # dataclasses resolve the module at class creation
    spec.loader.exec_module(mod)
    return mod


REF = _load("core")
REF_T = _load("torch_losses")

try:                                   # N1's 1-D posterior fitters (fork B, src/v9/quant.py): objects with .T(s) in [-1, 1]; selection on SELECT by the common J
    from v9 import quant as _Q  # type: ignore

    def _quant_fit(method: str, sP, sQ, sP_sel, sQ_sel, seed: int):
        sP, sQ, sPs, sQs = (np.asarray(x, dtype=np.float64) for x in (sP, sQ, sP_sel, sQ_sel))
        if method == "kde":
            obj = _Q.fit_kde(sP, sQ, sPs, sQs)
        elif method == "rulsif":
            obj = _Q.fit_rulsif(sP, sQ, sPs, sQs, seed=seed)
        elif method == "nystrom":
            obj = _Q.fit_nystrom(sP, sQ, sPs, sQs, seed=seed)
        else:
            raise ValueError(method)
        return obj.T
except Exception:                      # pragma: no cover - placeholder until src/v9/quant.py lands
    _quant_fit = None


# ------------------------------------------------------------------------------------------------------------ scalar critics (cos-2 class)
class ScalarCritic:
    """T = tanh(f / 2) with natural logit f = 2 (a s + b), a = softplus(a_raw) > 0 (score direction never flips); matched BCE uses the same f."""

    def __init__(self, a_raw: float = 0.0, b: float = 0.0):
        self.a_raw = torch.tensor(float(a_raw), requires_grad=True); self.b = torch.tensor(float(b), requires_grad=True)

    def logit(self, s: torch.Tensor) -> torch.Tensor:
        return 2.0 * (F.softplus(self.a_raw) * s + self.b)

    def T(self, s) -> np.ndarray:
        with torch.no_grad():
            return torch.tanh(self.logit(torch.as_tensor(np.asarray(s, dtype=np.float64))) / 2).numpy()

    def params(self) -> dict:
        return {"a": float(F.softplus(self.a_raw)), "a_raw": float(self.a_raw), "b": float(self.b)}


def common_j(T_P: np.ndarray, T_Q: np.ndarray) -> float:
    return float(np.mean(T_P - T_P ** 2 / 2) + np.mean(-T_Q - T_Q ** 2 / 2))


def fit_scalar_critic(s_P, s_Q, s_P_sel, s_Q_sel, kind: str, lr_grid=(1e-2, 3e-2, 1e-1), steps: int = 400, seed: int = 0,
                      inits=((0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (0.0, 1.0))) -> dict:
    """Full-batch Adam over a fixed lr grid and deterministic multi-start; selection on the SELECT role by the common J (primary) with each rule's
    native select loss recorded (sensitivity).  kind in {'vcs', 'bce'}.  Returns {'critic': ScalarCritic, 'selected': {...}, 'grid': [...]}."""
    sP, sQ = torch.as_tensor(np.asarray(s_P, dtype=np.float64)), torch.as_tensor(np.asarray(s_Q, dtype=np.float64))
    best, grid = None, []
    for lr in lr_grid:
        for a0, b0 in inits:
            torch.manual_seed(seed); c = ScalarCritic(a0, b0); opt = torch.optim.Adam([c.a_raw, c.b], lr=lr)
            for _ in range(steps):
                loss = REF_T.balanced_fit_loss(c.logit(sP), c.logit(sQ), kind=kind)
                opt.zero_grad(); loss.backward(); opt.step()
            TPs, TQs = c.T(s_P_sel), c.T(s_Q_sel); j_sel = common_j(TPs, TQs)
            with torch.no_grad():
                native_sel = float(REF_T.balanced_fit_loss(c.logit(torch.as_tensor(np.asarray(s_P_sel, dtype=np.float64))), c.logit(torch.as_tensor(np.asarray(s_Q_sel, dtype=np.float64))), kind=kind))
            rec = {"lr": lr, "init": [a0, b0], "fit_loss": float(loss), "select_common_J": j_sel, "select_native": native_sel, **c.params()}
            grid.append(rec)
            if best is None or j_sel > best[0] + 1e-12:
                best = (j_sel, c, rec)
    return {"critic": best[1], "selected": best[2], "grid": grid, "kind": kind, "selection_rule": "common J on SELECT (native loss recorded)"}


# ------------------------------------------------------------------------------------------------------------ reference moments and rho
def repair_within_role(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """The specified Q for a role: every caption gets the observed image of another pair of the SAME role (cyclic shift of a random permutation
    of the images present; a caption never keeps its own image).  Donor images are role-isolated by construction.  This is a conditional Q
    (same-image pairs rejected), disclosed in the protocol."""
    imgs = [r["observed_image_id"] for r in rows]; n = len(rows)
    if n < 2:
        raise ValueError("a role needs >= 2 pairs to build Q")
    perm = rng.permutation(n); shift = int(rng.integers(1, n))
    out = []
    for k, r in enumerate(rows):
        donor = imgs[perm[(int(np.where(perm == k)[0][0]) + shift) % n]]
        if donor == r["observed_image_id"]:
            donor = imgs[perm[(int(np.where(perm == k)[0][0]) + shift + 1) % n]]
        q = dict(r); q["observed_image_id"] = donor; q["q_of"] = r["pair_id"]; out.append(q)
    return out


def cluster_means(values: np.ndarray, cluster_ids) -> tuple[np.ndarray, list]:
    ids = np.asarray([str(c) for c in cluster_ids]); uniq = sorted(set(ids.tolist()))
    return np.array([float(np.mean(values[ids == u])) for u in uniq]), uniq


def estimate_rho(T_P_ref: np.ndarray, T_Q_ref: np.ndarray, T_U: np.ndarray, clusters_P=None, clusters_Q=None, clusters_U=None, alpha: float = 0.05) -> dict:
    """Point estimate from the three means (any fixed score); interval from the package's exact inversion on image-level cluster means when the
    role-disjointness contract passes (otherwise 'interval_unavailable')."""
    a, b, c = float(np.mean(T_P_ref)), float(np.mean(T_Q_ref)), float(np.mean(T_U))
    est = REF.rho_from_means(a, b, c)
    out = {"a": a, "b": b, "c": c, **asdict(est), "n_P": int(len(T_P_ref)), "n_Q": int(len(T_Q_ref)), "n_U": int(len(T_U))}
    if clusters_P is not None and clusters_Q is not None and clusters_U is not None:
        mP, uP = cluster_means(T_P_ref, clusters_P); mQ, uQ = cluster_means(T_Q_ref, clusters_Q); mU, uU = cluster_means(T_U, clusters_U)
        rows = [{"role": "P", "cluster_id": u, "units": [f"P:{u}"]} for u in uP] + [{"role": "Q", "cluster_id": u, "units": [f"Q:{u}"]} for u in uQ] + [{"role": "R", "cluster_id": u, "units": [f"R:{u}"]} for u in uU]
        try:
            REF.role_disjointness(rows); ok = True
        except ValueError as e:                                   # pragma: no cover
            ok = False; out["disjointness_error"] = str(e)
        if ok:
            cs = REF.rho_confidence_set(mP, mQ, mU, alpha=alpha, assumptions_verified=True)
            out["interval"] = asdict(cs); out["interval_note"] = ("image-level cluster means; P and Q clusters share the reference-moments images "
                                                                  "(Q is their re-pairing) so the P/Q cluster sets are the same images in two roles — reported as a diagnostic interval")
            out["n_clusters"] = {"P": len(uP), "Q": len(uQ), "R": len(uU)}
        else:
            out["interval"] = {"status": "interval_unavailable"}
    return out


def posterior_weights(T_U: np.ndarray, rho: float) -> np.ndarray:
    if rho <= 0:
        return np.ones_like(np.asarray(T_U, dtype=np.float64))
    return REF.clean_posterior(np.asarray(T_U, dtype=np.float64), float(rho))


# ------------------------------------------------------------------------------------------------------------ method table (N3..N7 + filter)
def weights_for_method(method: str, s: dict, rho_strategy: str = "estimated", seed: int = 0, lr_grid=(1e-2, 3e-2, 1e-1), steps: int = 400) -> dict:
    """s: {'fit_P','fit_Q','sel_P','sel_Q','ref_P','ref_Q','U'} raw teacher scores (+ optional '*_img' cluster ids).  Returns weights for the U
    pairs (order of s['U']) with the full audit trail.  rho_strategy 'estimated' (deployable) or 'oracle' (diagnostic only; needs s['rho_true'])."""
    rec = {"method": method, "rho_strategy": rho_strategy}
    if method in ("VCS_Weight", "BCE_Weight", "RuLSIF_Weight", "Nystrom_Weight", "KDE_Weight"):
        if method in ("VCS_Weight", "BCE_Weight"):
            fit = fit_scalar_critic(s["fit_P"], s["fit_Q"], s["sel_P"], s["sel_Q"], kind="vcs" if method == "VCS_Weight" else "bce", lr_grid=lr_grid, steps=steps, seed=seed)
            T = fit["critic"].T; rec["critic"] = {"class": "cos2 (a s + b, a = softplus)", **fit["selected"], "grid_size": len(fit["grid"])}
        else:
            if _quant_fit is None:
                raise NotImplementedError(f"{method}: the 1-D posterior fitters live in src/v9/quant.py (V9-N1, fork B); not available yet")
            T = _quant_fit({"RuLSIF_Weight": "rulsif", "Nystrom_Weight": "nystrom", "KDE_Weight": "kde"}[method], s["fit_P"], s["fit_Q"], s["sel_P"], s["sel_Q"], seed)
            rec["critic"] = {"class": method, "source": "v9.quant"}
        T_P_ref, T_Q_ref, T_U = T(s["ref_P"]), T(s["ref_Q"]), T(s["U"])
        rho = estimate_rho(T_P_ref, T_Q_ref, T_U, s.get("ref_P_img"), s.get("ref_Q_img"), s.get("U_img"))
        rec["rho"] = rho
        if rho_strategy == "oracle":
            rho_used = float(s["rho_true"]); rec["rho_used"] = rho_used; rec["deployable"] = False
        else:
            rho_used = rho["projected"] if rho["projected"] is not None else 0.0
            rec["rho_used"] = rho_used; rec["deployable"] = rho["status"] == "point_estimate"
            if rho["status"] == "unidentified_at_numerical_tolerance":
                rec["identification"] = "unidentified: a - b at tolerance; weights set to 1 (no re-weighting) and the unit is flagged"
        w = posterior_weights(T_U, rho_used); rec["T_U_summary"] = {"mean": float(np.mean(T_U)), "min": float(np.min(T_U)), "max": float(np.max(T_U))}
    elif method == "RankMass":
        if "total_mass" not in s:
            raise ValueError("RankMass needs s['total_mass'] = N3's training-side total mass over U (never N3's per-pair ranking)")
        w = REF.rank_mass_weights(np.asarray(s["U"], dtype=np.float64), float(s["total_mass"])); rec["total_mass"] = float(s["total_mass"])
    elif method == "PlainFilter":
        tau = float(np.quantile(np.asarray(s["sel_P"], dtype=np.float64), 0.05)); w = (np.asarray(s["U"], dtype=np.float64) >= tau).astype(np.float64)
        rec.update({"threshold": tau, "rule": "keep U pairs with raw score >= 5th percentile of SELECT true-pair scores (95 % clean retention on SELECT)", "retention_U": float(np.mean(w)), "main_family": False})
    elif method == "Unweighted_A":
        w = np.ones(len(s["U"]), dtype=np.float64)
    else:
        raise ValueError(f"unknown method {method}")
    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, 1.0)
    rec.update({"n_U": int(len(w)), "total_mass": float(w.sum()), "mean_w": float(w.mean()), "ess": REF.effective_sample_size(w) if w.sum() > 0 else 0.0,
                "clipped_fraction": float(np.mean((w <= 1e-12) | (w >= 1 - 1e-12)))})
    return {"w": w, "record": rec}


# ------------------------------------------------------------------------------------------------------------ locked weight file
def write_weight_lock(path: str | Path, pair_ids, w, score, rho_hat, fit_hash: str, reference_hash: str, record: dict) -> dict:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["pair_id", "w", "score", "rho_hat", "fit_hash", "reference_hash"])
        for p, ww, ss in zip(pair_ids, w, score):
            wr.writerow([p, f"{float(ww):.10f}", f"{float(ss):.10f}", f"{float(rho_hat):.8f}" if rho_hat is not None else "", fit_hash, reference_hash])
    lock = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": int(len(w)), "record": record}
    Path(str(path) + ".lock.json").write_text(json.dumps(lock, indent=1))
    return lock


def read_weight_lock(path: str | Path) -> dict:
    path = Path(path); lock = json.loads(Path(str(path) + ".lock.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != lock["sha256"]:
        raise ValueError("weight file does not match its lock (sha256)")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["pair_id"]] = float(row["w"])
    if len(out) != lock["n"]:
        raise ValueError("weight file row count differs from the lock")
    return out
