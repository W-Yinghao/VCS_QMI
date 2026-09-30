"""P101 — package v2 O line (Server Spec v2 §10, Plan §9.2): observation-noise calibration on frozen SSL representations.

    python -m vcs_estim.observation_frozen --run-dir <outputs/RUN> --ckpt epoch_800 --out <json> [--ref-json <reference ckpt JSON>] [--smoke] [--cpu]
    python -m vcs_estim.observation_frozen --stage aggregate --inputs <jsons> --out <md>

Per checkpoint (encoder / projector frozen, eval() on deep copies; the P84 identity roles and view seeds):
  roles        FIT (27k) is split at identity level into FIT-CRITIC (13.5k) and FIT-CAL (13.5k) by a seeded permutation; TUNE is unused;
               SELECT (6k) stops the measurement-critic training; EVAL (6k) is read once for reporting.  Roles are disjoint (asserted).
  noise        u = z + tau * eps / sqrt(d) on the unit-norm projector output z, both sides of every pair, independent draws per pair row, no
               re-normalisation (Spec §10.1).  Subspace variant: u = z + (tau / sqrt(r)) * B eps_r with B the FIT-CRITIC PCA basis (90 % variance,
               r <= 64), computed once from the noise-free FIT-CRITIC z and fixed for every tau.  noise_total_rms = tau; noise_coordinate_sd =
               tau / sqrt(d) (isotropic) or tau / sqrt(r) inside the subspace.  Draws come from seeded generators keyed by (role, tau, mode), so
               every checkpoint sees the same noise realisation (common random numbers).
  calibration  calibration critic = the converged two-parameter cosine C0 (full-batch L-BFGS on FIT-CRITIC pairs, K = 8 cyclic shifts; the
               P84-addendum object), fitted anew on the noisy FIT-CRITIC pairs for every tau; J_proxy(tau) = its J on noisy FIT-CAL blocks.
               J_proxy is a fixed proxy score, never called S (calibration_target = "J_proxy").  Targets {0.95, 0.85, 0.70, 0.50}; grid
               tau in {0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0}; for a target bracketed by two grid points, at most 10 bisection steps until
               |J_proxy - target| <= 0.03; a target outside the grid's J range is recorded as unreachable (no extension of the grid).
               The per-checkpoint tau*(target) is itself reported (the noise needed to bring each representation to the same proxy J).
  measurement  at tau = 0 and at the *common* tau of each target — calibrated on the reference checkpoint (the VCS 4-view 800-epoch recipe
               endpoint, P35 seed 0 epoch 800) and applied unchanged to every checkpoint — VCS and matched-JS critics of the C0 and C2
               classes are trained on noisy FIT-CRITIC pairs (plus the converged C0, the calibration class, fitted to convergence as a separate row) with the P84 budget (Adam 5e-4, 2000 updates, batch 256, SELECT every 100)
               and evaluated once on noisy EVAL blocks: block J (+ SE), matched-JS native value, and kNN top-1 of noisy view-1 z (EVAL queries,
               FIT-CRITIC bank, k 200, T 0.1; labels used for this diagnostic only).
No oracle: S, eta and posterior MSE are absent with a reason.  Aggregation (stage aggregate): per method (VCS, SimCLR) the adjacent-checkpoint
resolution |Delta J| / sqrt(se1^2 + se2^2) at tau = 0 and at each common tau, its ratio to tau = 0, and the kNN change.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import candidates as C
from .fitting import native_loss, outputs, train
from .frozen import Pairs, block_J, block_pairs, encode_role, fit_pairs, load_models
from .frozen_supplement import fit_c0_converged
from .objectives import js_match_loss, js_native
from .pairing import ROLE_SEED, check_disjoint, role_split, split_hash
from .run import git_commit

TARGETS = (0.95, 0.85, 0.70, 0.50)
TAU_GRID = (0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0)
TOL, MAX_REFINE = 0.03, 10
FITCAL_SEED = 20261001
REFERENCE = ("P35_vcs_a5_views4_800ep_seed0", "epoch_800")
ABSENT = {"S_truth": None, "S_truth_absent_reason": "no oracle for image pairs", "posterior_mse": None, "posterior_mse_absent_reason": "eta unknown"}


# ------------------------------------------------------------------------------------------------------------ roles and noise
def o_line_roles(fit_uids, seed: int = ROLE_SEED) -> dict:
    roles = role_split(fit_uids, seed); fit = np.asarray(roles.pop("FIT"))
    perm = np.random.default_rng(FITCAL_SEED).permutation(len(fit)); h = len(fit) // 2
    roles["FIT-CRITIC"] = sorted(fit[perm[:h]].tolist()); roles["FIT-CAL"] = sorted(fit[perm[h:]].tolist())
    check_disjoint(roles)
    return roles


def pca_basis(z: torch.Tensor, frac: float = 0.90, cap: int = 64) -> tuple[torch.Tensor, dict]:
    """Fixed FIT PCA basis B (d x r): the smallest r reaching `frac` explained variance, capped at `cap`."""
    x = z.double() - z.double().mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh(x.T @ x / (len(x) - 1)); evals, evecs = evals.flip(0), evecs.flip(1)
    cum = torch.cumsum(evals, 0) / evals.sum(); r = int(min(cap, int((cum < frac).sum()) + 1))
    return evecs[:, :r].float().contiguous(), {"r": r, "explained": float(cum[r - 1]), "frac_target": frac, "cap": cap,
                                               "basis_sha256": hashlib.sha256(evecs[:, :r].float().numpy().tobytes()).hexdigest()}


def _gen(*key) -> torch.Generator:
    return torch.Generator().manual_seed(int(hashlib.sha256(repr(key).encode()).hexdigest()[:15], 16))


def add_noise(z: torch.Tensor, tau: float, mode: str, gen: torch.Generator, basis: torch.Tensor | None = None) -> torch.Tensor:
    if tau == 0:
        return z.clone()
    if mode == "iso":
        return z + (tau / math.sqrt(z.shape[1])) * torch.randn(z.shape, generator=gen, dtype=z.dtype)
    if mode == "pca":
        r = basis.shape[1]
        return z + (tau / math.sqrt(r)) * (torch.randn((z.shape[0], r), generator=gen, dtype=z.dtype) @ basis.T.to(z.dtype))
    raise ValueError(mode)


def noisy_pairs(P: Pairs, tau: float, mode: str, role: str, basis=None) -> Pairs:
    """Independent noise on each side of every pair row (fresh per row), keyed by (role, tau, mode) for common random numbers."""
    g = _gen("o-line", role, round(tau, 6), mode)
    return Pairs(*(add_noise(t, tau, mode, g, basis) for t in (P.xp, P.yp, P.xq, P.yq)))


def noise_fields(tau: float, mode: str, d: int, r: int | None) -> dict:
    return {"noise_tau": tau, "noise_total_rms": tau, "noise_mode": mode, "d": d, "subspace_rank": r,
            "noise_coordinate_sd": (tau / math.sqrt(d)) if mode == "iso" else (tau / math.sqrt(r)), "calibration_target": "J_proxy",
            "renormalised_after_noise": False}


# ------------------------------------------------------------------------------------------------------------ calibration
def calibrate(j_of_tau, targets=TARGETS, grid=TAU_GRID, tol=TOL, max_refine=MAX_REFINE) -> dict:
    """Scan the grid, then bisect inside the bracketing grid interval (<= max_refine evaluations) until |J - target| <= tol.  The raw curve is
    kept as measured (no monotone smoothing); a target outside [min J, max J] of the scanned curve is recorded unreachable."""
    curve = [(t, j_of_tau(t)) for t in grid]; out = {"grid_curve": curve, "targets": {}}
    for tgt in targets:
        best = min(curve, key=lambda c: abs(c[1] - tgt)); rec = {"target": tgt, "trace": []}
        if abs(best[1] - tgt) <= tol:
            rec.update(tau=best[0], J=best[1], reachable=True, refinements=0); out["targets"][str(tgt)] = rec; continue
        br = next(((curve[i], curve[i + 1]) for i in range(len(curve) - 1) if (curve[i][1] - tgt) * (curve[i + 1][1] - tgt) < 0), None)
        if br is None:
            rec.update(tau=None, J=None, reachable=False, refinements=0, nearest=best,
                       reason="target outside the J range of the scanned grid; grid not extended (Spec §10.2)"); out["targets"][str(tgt)] = rec; continue
        (lo, jlo), (hi, jhi) = br; cand = best
        for _ in range(max_refine):
            mid = 0.5 * (lo + hi); jm = j_of_tau(mid); rec["trace"].append((mid, jm))
            if abs(jm - tgt) < abs(cand[1] - tgt):
                cand = (mid, jm)
            if abs(jm - tgt) <= tol:
                break
            if (jlo - tgt) * (jm - tgt) < 0:
                hi, jhi = mid, jm
            else:
                lo, jlo = mid, jm
        rec.update(tau=cand[0], J=cand[1], reachable=abs(cand[1] - tgt) <= tol, refinements=len(rec["trace"]))
        if not rec["reachable"]:
            rec["reason"] = f"no tau within {max_refine} refinements reached |J - target| <= {tol}; closest recorded"
        out["targets"][str(tgt)] = rec
    return out


def proxy_J(fitc: Pairs, cal: Pairs, tau: float, mode: str, basis, device, rounds: int) -> float:
    Fn, Cn = noisy_pairs(fitc, tau, mode, "FIT-CRITIC", basis), noisy_pairs(cal, tau, mode, "FIT-CAL", basis)
    s = torch.cat([(Fn.xp * Fn.yp).sum(-1), (Fn.xq * Fn.yq).sum(-1)]).double(); std = (float(s.mean()), float(s.std()))
    m, _ = fit_c0_converged(Fn.xp.shape[1], std, Fn, device, max_rounds=rounds)
    tp, tn = torch.tanh(outputs(m, Cn.xp, Cn.yp, device)), torch.tanh(outputs(m, Cn.xq, Cn.yq, device))
    return block_J(tp, tn)["J"]


# ------------------------------------------------------------------------------------------------------------ measurement
def measure(fitc, sel, ev, tau, mode, basis, device, updates, seed, knn, rounds=40) -> dict:
    Fn, Sn, En = (noisy_pairs(P, tau, mode, r, basis) for P, r in ((fitc, "FIT-CRITIC"), (sel, "SELECT"), (ev, "EVAL")))
    s = torch.cat([(Fn.xp * Fn.yp).sum(-1), (Fn.xq * Fn.yq).sum(-1)]).double(); std = (float(s.mean()), float(s.std())); d = Fn.xp.shape[1]
    rows = {}
    for fam in ("C0", "C2"):
        for kind in ("vcs", "js"):
            torch.manual_seed(1000 * seed + {"C0": 1, "C2": 3}[fam] + (0 if kind == "vcs" else 7))
            m, info = train(C.build(fam, d, c0_a0=0.0, c0_std=std), native_loss(kind), Fn, Sn, updates=updates, seed=seed, device=device)
            fp, fn = outputs(m, En.xp, En.yp, device), outputs(m, En.xq, En.yq, device)
            bj = block_J(torch.tanh(fp), torch.tanh(fn))
            rows[f"{kind}_{fam}"] = {**bj, "js_native": js_native(float(js_match_loss(fp.float(), fn.float()))) if kind == "js" else None,
                                     "selected_update": info["selected_update"], "fit_seconds": info["fit_seconds"]}
    mc, cinfo = fit_c0_converged(d, std, Fn, device, max_rounds=rounds)                 # the calibration class, fitted to convergence
    tp, tn = torch.tanh(outputs(mc, En.xp, En.yp, device)), torch.tanh(outputs(mc, En.xq, En.yq, device))
    rows["vcs_C0conv"] = {**block_J(tp, tn), "js_native": None, "selected_update": None, "fit_seconds": cinfo["fit_seconds"],
                          "a_eff": cinfo["a_eff"], "b_eff": cinfo["b_eff"], "note": "converged two-parameter cosine (full-batch L-BFGS), not the 2000-update C0"}
    k = knn(Fn.xp, En.xp)
    return {"tau": tau, "mode": mode, "rows": rows, "knn_top1_noisy_z": k, **ABSENT}


def run_checkpoint(a) -> dict:
    from vcs_ssl.data.cifar import load_cifar10_train
    from vcs_ssl.data.transforms import build_two_view_transform
    from vcs_ssl.diagnostics import knn_eval
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"; t_all = time.time()
    run_dir = Path(a.run_dir); cfg, encoder, projector, _critic, meta = load_models(run_dir, a.ckpt, dev)
    eps = float(cfg["model"]["normalization"]["eps"]); tf = build_two_view_transform(cfg["views"])
    manifest = json.load(open(cfg["data"]["manifest"])); roles = o_line_roles(manifest["fit_uids"])
    if a.smoke:
        roles = {r: v[: {"FIT-CRITIC": 768, "FIT-CAL": 256, "TUNE": 2, "SELECT": 256, "EVAL": 256}[r]] for r, v in roles.items()}
    roles.pop("TUNE")
    data = load_cifar10_train(cfg["data"]["root"]); images, labels = data.data, torch.as_tensor(data.targets)
    seeds = {"FIT-CRITIC": 20260928, "FIT-CAL": 20260928 + 5, "SELECT": 20260928 + 34, "EVAL": 20260928 + 51}
    Z = {r: encode_role(encoder, projector, images, ids, tf, seeds[r], dev, eps, workers=a.workers) for r, ids in roles.items()}
    fitc, fit_info = fit_pairs(Z["FIT-CRITIC"][0], Z["FIT-CRITIC"][1], 8, 20260929)
    cal, _ = block_pairs(Z["FIT-CAL"][0], Z["FIT-CAL"][1], roles["FIT-CAL"], 20261002)
    sel, _ = block_pairs(Z["SELECT"][0], Z["SELECT"][1], roles["SELECT"], 20260931)
    ev, _ = block_pairs(Z["EVAL"][0], Z["EVAL"][1], roles["EVAL"], 20260932)
    d = fitc.xp.shape[1]; basis, pinfo = pca_basis(torch.cat([Z["FIT-CRITIC"][0], Z["FIT-CRITIC"][1]]))
    y_fitc = labels[torch.as_tensor(roles["FIT-CRITIC"])]
    from .pairing import eval_blocks
    anchors = torch.as_tensor(eval_blocks(roles["EVAL"], 20260932)[:, 0]); y_ev = labels[anchors]
    knn = lambda bank, query: knn_eval(bank, y_fitc, query, y_ev, k=min(200, len(bank) - 1), temperature=0.1, device=torch.device(dev),
                                       n_classes=int(labels.max()) + 1)["knn_val_top1_pct"]
    rounds = 8 if a.smoke else 40; R = {"checkpoint": {"run_dir": str(run_dir), "ckpt": a.ckpt, **meta, "method_cfg": cfg["run"]["method"]},
         "code_commit": git_commit(), "roles": {"sizes": {r: len(v) for r, v in roles.items()}, "hash": split_hash(roles), "fitcal_seed": FITCAL_SEED,
         "role_seed": ROLE_SEED}, "pca": pinfo, "d": d, "device": dev, "eval_once": "EVAL read for reporting only; tau chosen on FIT-CAL only",
         "calibration_critic": "converged two-parameter cosine C0 (L-BFGS on noisy FIT-CRITIC pairs, K = 8), J_proxy on noisy FIT-CAL blocks", **ABSENT}
    R["calibration"] = {}
    for mode in ("iso", "pca"):
        cal_res = calibrate(lambda t, m=mode: proxy_J(fitc, cal, t, m, basis, dev, rounds))
        R["calibration"][mode] = {**cal_res, "fields": {str(t): noise_fields(t, mode, d, pinfo["r"]) for t in TAU_GRID}}
    ref = json.load(open(a.ref_json)) if a.ref_json and Path(a.ref_json).exists() else None
    is_ref = (run_dir.name, a.ckpt) == REFERENCE
    src = R if is_ref or ref is None else ref
    common = {mode: {k: v["tau"] for k, v in src["calibration"][mode]["targets"].items()} for mode in ("iso", "pca")}
    R["common_tau"] = {"source": "this checkpoint (reference)" if (is_ref or ref is None) else f"reference {REFERENCE}", "tau": common}
    if ref is None and not is_ref:
        R["common_tau"]["warning"] = "reference JSON missing: common tau taken from this checkpoint (smoke / out-of-order run)"
    R["measurements"] = [measure(fitc, sel, ev, 0.0, "iso", basis, dev, a.updates, a.seed, knn, rounds)]
    for mode in ("iso", "pca"):
        for tgt, tau in common[mode].items():
            if tau is None or tau == 0.0:
                continue
            mres = measure(fitc, sel, ev, float(tau), mode, basis, dev, a.updates, a.seed, knn, rounds); mres["target"] = float(tgt)
            mres.update(noise_fields(float(tau), mode, d, pinfo["r"])); R["measurements"].append(mres)
    R["wall_seconds"] = time.time() - t_all
    return R


# ------------------------------------------------------------------------------------------------------------ aggregate
def aggregate(paths) -> str:
    Rs = [json.load(open(p)) for p in paths]
    key = lambda R: (R["checkpoint"]["method_cfg"], int(str(R["checkpoint"]["ckpt"]).split("_")[-1]) if "epoch" in str(R["checkpoint"]["ckpt"]) else 0)
    Rs.sort(key=key); L = ["# P101 aggregate — O line on frozen representations", "", "## Calibrated tau per checkpoint (J_proxy targets; None = unreachable)", "",
                           "| checkpoint | mode | " + " | ".join(str(t) for t in TARGETS) + " |", "|---|---|" + "---|" * len(TARGETS)]
    for R in Rs:
        for mode in ("iso", "pca"):
            t = R["calibration"][mode]["targets"]
            L.append(f"| {Path(R['checkpoint']['run_dir']).name}:{R['checkpoint']['ckpt']} | {mode} | " +
                     " | ".join(("—" if t[str(x)]["tau"] is None else f"{t[str(x)]['tau']:.3f}") for x in TARGETS) + " |")
    L += ["", "## Resolution between adjacent checkpoints of one method (|ΔJ| / pooled block SE; ratio to tau = 0; kNN change)", ""]
    for method in sorted({R["checkpoint"]["method_cfg"] for R in Rs}):
        seq = [R for R in Rs if R["checkpoint"]["method_cfg"] == method]
        settings = [(m["mode"], m.get("target")) for m in seq[0]["measurements"]]
        L += [f"### {method}", "", "| setting | critic | " + " | ".join(f"{a['checkpoint']['ckpt']}→{b['checkpoint']['ckpt']}" for a, b in zip(seq, seq[1:])) + " | kNN (per ckpt) |",
              "|---|---|" + "---|" * max(1, len(seq) - 1) + "---|"]
        base = {}
        for mode, tgt in settings:
            for crit in ("vcs_C0conv", "vcs_C0", "vcs_C2", "js_C0", "js_C2"):
                cells, kn = [], []
                for a_, b_ in zip(seq, seq[1:]):
                    ma = next(m for m in a_["measurements"] if (m["mode"], m.get("target")) == (mode, tgt))
                    mb = next(m for m in b_["measurements"] if (m["mode"], m.get("target")) == (mode, tgt))
                    ra, rb = ma["rows"][crit], mb["rows"][crit]
                    se = math.sqrt(ra["J_se_block"] ** 2 + rb["J_se_block"] ** 2)
                    res = abs(rb["J"] - ra["J"]) / se if se > 0 else float("nan")          # se = 0 only for a constant critic (e.g. smoke budgets)
                    if tgt is None:
                        base[(crit, a_["checkpoint"]["ckpt"])] = res
                    b0 = base.get((crit, a_["checkpoint"]["ckpt"]), float("nan"))
                    ratio = res / b0 if (b0 == b0 and b0 > 0) else float("nan")
                    cells.append(f"{res:.1f} (×{ratio:.2f})")
                kn = [next(m for m in R["measurements"] if (m["mode"], m.get("target")) == (mode, tgt))["knn_top1_noisy_z"] for R in seq]
                L.append(f"| {mode} {tgt if tgt is not None else 'τ=0'} | {crit} | " + " | ".join(cells) + " | " + " / ".join(f"{k:.1f}" for k in kn) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="run", choices=("run", "aggregate"))
    ap.add_argument("--run-dir"); ap.add_argument("--ckpt"); ap.add_argument("--out", required=True); ap.add_argument("--ref-json", default=None)
    ap.add_argument("--inputs", nargs="*"); ap.add_argument("--updates", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    if a.stage == "aggregate":
        Path(a.out).write_text(aggregate(a.inputs)); print("wrote", a.out); return 0
    if a.smoke:
        a.updates = 60
    R = run_checkpoint(a); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(R, indent=1, default=str)); print("wrote", a.out, f"({R['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
