"""P149 (r3 V0 + V1) — two-view dependence of a frozen encoder: fixture + measurement under coordinate change and a common random channel.

Fixture (`fixture`): per (dataset, run) 20 000 base images from the run's manifest FIT uids, split FIT 10 000 / VAL 2 000 / EVAL 8 000 by base id
(seed 20261002, the P109 / P143 roles); two independent views A1, A2 per base image under the dataset's P122 *standard* measurement law
(law signature checked; augmentation seed 122, per-image seeding (seed, tag, uid) exactly as P122, so images that were in P122's pools get
byte-identical views); frozen eval-mode encoder -> h (512) and z (128), float32, per role and view.  EVAL independent pools (seed 20261007):
P pool = 2 000 base images (A1_i, A2_i); Q-left = 2 000 other base images (A1); Q-right = 2 000 other base images (A2); 2 000 reserved.
Nothing of the official test partition is read.

Measurement (`measure`, representation h): FIT standardisation per view (mean mu_j, scalar gamma_j = sqrt(mean squared deviation per coordinate),
P/Q-fit 2 048 pairs each (P from 2 048 FIT base images; Q from 2 048 + 2 048 disjoint FIT base images), TUNE from VAL (1 000 P pairs, 500 Q pairs).
Settings: identity_t0 | orthogonal_refit_t0 (fixed Haar-orthogonal R1, R2 on view 1 / 2, registered seeds, applied to all roles, critics refitted) |
identity_t0.25 | identity_t1 (Brownian channel W_t ~ N(0, t I) per coordinate on the standardised representation, independent across views and
base images, coupled across t by increments, replayable from (seed, role, uid, view) — the reference_core convention).  Estimators: vcs_mlp
(f = MLP[u; v] -> 256 -> 256 -> 1, loss -J), js_mlp (same net, loss softplus(-2f) + softplus(2f); common T = tanh f, half-logit coordinate),
rff_ridge_tanh (psi = RFF of [u; v], D 1024; the P116 two-stage ridge-tanh solver on [psi, 1]).  Candidates: MLP lr in {1e-4, 5e-4, 2e-3};
RFF bandwidth multiple in {0.5, 1, 2} x FIT median pair distance; <= 1000 full-batch AdamW steps (wd 1e-2), TUNE own risk every 10 steps,
selection on TUNE only; 3 init seeds per (setting, estimator).  EVAL once per selected cell: per-sample T on the P and Q pools (saved), J_common,
independent two-pool SE and Hoeffding radius, S_plugin, T quantiles, saturation, TUNE / FIT J, seconds; nulls: T = 0 and the independent-pair
null (derangement of the P pool) with a B = 200 permutation p-value (shared permutations); pair_var_conditional_batch from 64 re-pairings of the
Q pools with the fixed critic.  Transport check (orthogonal setting, init seed 0): the identity critic with first-layer weights W' = W blockdiag(R1^T,
R2^T) (MLP) / transported frequencies (RFF) must reproduce its outputs on rotated inputs.

    python scripts/p149_twoview.py fixture --dataset cifar10 --run P107_AP3_views4_800ep_seed1 --out outputs/P149_fixtures [--smoke]
    python scripts/p149_twoview.py measure --fixture outputs/P149_fixtures/<run> --out reports/P149/<run> [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vcs_measure import evidence as ev  # noqa: E402  (P122: law_signature, augment, encode, own_risk, j_terms, _train)
from vcs_measure.common import load_run, split_base_ids  # noqa: E402
from vcs_measure.solve import Design, ridge_tanh_calibrated  # noqa: E402  (P116 two-stage solver, used generically on [psi, 1])
from vcs_ssl.data.cifar import load_train_partition  # noqa: E402
from vcs_ssl.data.splits import load_manifest  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

SPLIT_SEED, POOL_SEED, AUG_SEED = 20261002, 20261007, 122
LAW_SOURCES = {"cifar10": "configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml", "cifar100": "configs/cifar100_hpS_vcs_a5_views4_800ep_seed0.yaml"}
LAW_SIG = "22f3b58bfde64c92"  # P122 standard law (CIFAR-10 and CIFAR-100 blocks share it)
MANIFESTS = {"cifar10": "/home/infres/yinwang/CS_QMI/manifests/cifar10_dev45k_val5k.json", "cifar100": "/home/infres/yinwang/CS_QMI/manifests/cifar100_dev45k_val5k.json"}
ROOTS = {"cifar10": "/home/infres/yinwang/CS_QMI/data/cifar10", "cifar100": "/home/infres/yinwang/CS_QMI/data/cifar100"}
ROLES = {"fit": 0, "val": 1, "eval": 2}
TIMES = (0.0, 0.25, 1.0)
SETTINGS = ("identity_t0", "orthogonal_refit_t0", "identity_t0.25", "identity_t1")
ESTIMATORS = ("vcs_mlp", "js_mlp", "rff_ridge_tanh")
LRS, BWS, RFF_D, FIT_SEEDS = (1e-4, 5e-4, 2e-3), (0.5, 1.0, 2.0), 1024, (0, 1, 2)
N_FIT, N_TUNE_P, N_TUNE_Q, N_EVAL = 2048, 1000, 500, 2000
ROT_SEEDS, NOISE_SEED, PERM_SEED, BOOT_SEED = (149001, 149002), 149, 1490, 14900
B_PERM, N_REPAIR = 200, 64
DEV = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------- fixture
def build_fixture(dataset: str, run: str, out: Path, smoke: bool) -> None:
    t0 = time.time()
    views = yaml.safe_load(open(REPO / LAW_SOURCES[dataset]))["views"]
    sig = ev.law_signature(views); assert sig == LAW_SIG, f"law signature {sig} != registered {LAW_SIG}"
    data = load_train_partition(dataset, ROOTS[dataset]); man = load_manifest(MANIFESTS[dataset])
    R = load_run(run, device=DEV); assert R["cfg"]["data"]["name"] == dataset
    sizes = {"fit": 10000, "val": 2000, "eval": 8000} if not smoke else {"fit": 600, "val": 200, "eval": 800}
    sp = split_base_ids(np.asarray(man["fit_uids"], dtype=np.int64), sizes, SPLIT_SEED)
    mean, std = R["cfg"]["views"]["normalize_mean"], R["cfg"]["views"]["normalize_std"]
    store = {"run": run, "dataset": dataset, "law_signature": sig, "law_source": LAW_SOURCES[dataset], "aug_seed": AUG_SEED, "split_seed": SPLIT_SEED,
             "checkpoint_sha256": hashlib.sha256(open(Path(R["cfg"]["run"]["output_root"]) / run / "checkpoints" / "epoch_800.pt", "rb").read()).hexdigest()[:16],
             "normalize": [mean, std], "roles": {}, "features": {}}
    for role, ids in sp.items():
        store["roles"][role] = torch.as_tensor(ids)
        feats = {}
        for view, tag in (("A1", 1), ("A2", 2)):
            imgs = ev.augment(data.data, ids, views, AUG_SEED, tag)
            feats[view] = ev.encode(R, imgs, mean, std, DEV)
        store["features"][role] = feats
        print(f"[{run}] {role}: {len(ids)} base images x 2 views encoded ({time.time() - t0:.0f}s)", flush=True)
    # EVAL independent pools (positions into the eval role), fixed by POOL_SEED
    ne = len(sp["eval"]); perm = np.random.default_rng(POOL_SEED).permutation(ne); k = N_EVAL if not smoke else ne // 4
    store["eval_pools"] = {"P": torch.as_tensor(perm[:k]), "Q_left": torch.as_tensor(perm[k:2 * k]), "Q_right": torch.as_tensor(perm[2 * k:3 * k]),
                           "reserved": torch.as_tensor(perm[3 * k:]), "pool_seed": POOL_SEED}
    out.mkdir(parents=True, exist_ok=True); torch.save(store, out / "fixture.pt")
    atomic_write_json(out / "manifest.json", {"run": run, "dataset": dataset, "law_signature": sig, "law_source": LAW_SOURCES[dataset], "aug_seed": AUG_SEED,
                                              "split_seed": SPLIT_SEED, "pool_seed": POOL_SEED, "sizes": {k: int(len(v)) for k, v in sp.items()},
                                              "eval_pool_size": int(k), "checkpoint_sha256": store["checkpoint_sha256"], "views_per_image": 2,
                                              "representation_sites": ["h", "z"], "dtype": "float32", "utc": utc_now(), "seconds": time.time() - t0,
                                              "pretraining_exposure": "all base images are training-partition images the encoder was pre-trained on (disclosed)"})
    print(f"[{run}] fixture written ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------------------------------------------------- channel / coordinates
def brownian(x: np.ndarray, ids: np.ndarray, role: int, view: int, seed: int = NOISE_SEED) -> dict[float, np.ndarray]:
    """{t: x + W_t} for the fixed grid TIMES, replayable per (seed, role, uid, view) exactly as reference_core.brownian_view (sequential increments)."""
    out = {TIMES[0]: x.copy()}; d = x.shape[1]
    noise = np.zeros_like(x)
    rngs = [np.random.default_rng(np.random.SeedSequence([seed, role, int(u), view, 991])) for u in ids]
    for lvl in range(1, len(TIMES)):
        dt = TIMES[lvl] - TIMES[lvl - 1]
        for i, rng in enumerate(rngs):
            noise[i] += math.sqrt(dt) * rng.standard_normal(d)
        out[TIMES[lvl]] = x + noise
    return out


def haar_orthogonal(d: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q * np.sign(np.diag(r))[None, :]


def standardizer(xfit: np.ndarray) -> tuple[np.ndarray, float]:
    mu = xfit.mean(0); gamma = float(np.sqrt(((xfit - mu) ** 2).mean()))
    if gamma <= 0:
        raise ValueError("collapsed FIT representation")
    return mu, gamma


# ----------------------------------------------------------------------------------------------------------------------- critics
class PairMLP(nn.Module):
    """f(u, v) = MLP([u; v]); first layer linear in the inputs so an orthogonal change of coordinates transports exactly (W' = W blockdiag(R1^T, R2^T))."""

    def __init__(self, d: int, hidden: int = 256, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, u, v):
        return self.net(torch.cat([u, v], 1)).squeeze(1)


def fit_mlp(loss: str, Pf, Qf, Pt, Qt, d: int, seed: int, steps: int) -> dict:
    """Three lr candidates, one init seed; selection by TUNE own risk (min).  Returns the selected candidate (model, meta, all candidate rows)."""
    rows, best = [], None
    for lr in LRS:
        t0 = time.perf_counter(); m = PairMLP(d, seed=seed).to(DEV)
        fn = lambda pairs, cache=None, m=m: (m(pairs[0][0], pairs[0][1]), m(pairs[1][0], pairs[1][1]))
        risk, step = ev._train(fn, list(m.parameters()), (Pf, Qf), (Pt, Qt), loss, lr, seed, steps=steps, every=10)
        rows.append({"lr": lr, "tune_risk": risk, "best_step": step, "seconds": time.perf_counter() - t0, "model": m})
        if best is None or risk < best["tune_risk"]:
            best = rows[-1]
    return {"model": best["model"], "selected": {k: v for k, v in best.items() if k != "model"}, "candidates": [{k: v for k, v in r.items() if k != "model"} for r in rows],
            "fit_seconds": sum(r["seconds"] for r in rows)}


class RFFCritic:
    def __init__(self, W: torch.Tensor, b: torch.Tensor, v: torch.Tensor):
        self.W, self.b, self.v = W, b, v  # W: [2d, D] float64 (frequencies), b: [D], v: [D + 1] (ridge weights incl. bias, tanh scale folded in)

    def psi(self, u, v):
        x = torch.cat([u, v], 1).double()
        return math.sqrt(2.0 / self.W.shape[1]) * torch.cos(x @ self.W + self.b)

    def f(self, u, v):
        p = self.psi(u, v); return torch.cat([p, torch.ones(len(p), 1, dtype=torch.float64, device=p.device)], 1) @ self.v


def fit_rff(Pf, Qf, Pt, Qt, d: int, seed: int) -> dict:
    """Three bandwidth candidates (median FIT pair distance x {0.5, 1, 2}); frequencies from (seed, candidate); the P116 ridge-tanh two-stage solver
    on phi = [psi, 1] with its lambda grid and 25-point tanh scale selected on TUNE (the solver's own VAL role)."""
    t0 = time.perf_counter()
    xf = torch.cat([Pf[0], Pf[1]], 1).double(); sub = xf[torch.randperm(len(xf), generator=torch.Generator().manual_seed(seed))[:1000]]
    med = float(torch.median(torch.cdist(sub, sub)[torch.triu(torch.ones(len(sub), len(sub), dtype=torch.bool), 1)]))
    rows, best = [], None
    for k, mult in enumerate(BWS):
        g = torch.Generator().manual_seed(seed * 1000 + 149 + k)
        W = (torch.randn(2 * d, RFF_D, generator=g, dtype=torch.float64) / (mult * med)).to(DEV); b = (2 * math.pi * torch.rand(RFF_D, generator=g, dtype=torch.float64)).to(DEV)
        cr = RFFCritic(W, b, None)
        one = lambda n: torch.ones(n, 1, dtype=torch.float64, device=DEV)
        phi = lambda pairs: torch.cat([cr.psi(pairs[0], pairs[1]), one(len(pairs[0]))], 1)
        D = Design(phi(Pf), phi(Qf), phi(Pt), phi(Qt))
        lc = ridge_tanh_calibrated(D)
        rows.append({"bw_mult": mult, "median_dist": med, "tune_J": lc.meta["val_J"], "lam": lc.meta["lam"], "c": lc.meta["c"], "W": W, "b": b, "v": lc.v})
        if best is None or lc.meta["val_J"] > best["tune_J"]:
            best = rows[-1]
    crit = RFFCritic(best["W"], best["b"], best["v"])
    meta = {k: v for k, v in best.items() if k not in ("W", "b", "v")}
    return {"model": crit, "selected": {**meta, "tune_risk": -best["tune_J"]}, "candidates": [{k: v for k, v in r.items() if k not in ("W", "b", "v")} for r in rows],
            "fit_seconds": time.perf_counter() - t0}


def critic_f(est: str, model, u, v):
    with torch.no_grad():
        return model(u, v).double() if est != "rff_ridge_tanh" else model.f(u, v)


# ----------------------------------------------------------------------------------------------------------------------- readouts
def readout(tp: np.ndarray, tq: np.ndarray, delta: float = 0.05) -> dict:
    ap, am = tp - 0.5 * tp ** 2, -tq - 0.5 * tq ** 2
    J = float(ap.mean() + am.mean()); se = float(np.sqrt(ap.var(ddof=1) / len(ap) + am.var(ddof=1) / len(am)))
    rad = float(np.sqrt(2 * (1 / len(ap) + 1 / len(am)) * np.log(2 / delta)))
    qs = [5, 25, 50, 75, 95]
    return {"J_common": J, "se_J": se, "hoeffding_radius_J": rad, "conservative_S_lower": max(0.0, J - rad), "S_plugin": float(0.5 * ((tp ** 2).mean() + (tq ** 2).mean())),
            "T_P_quantiles": dict(zip(qs, np.percentile(tp, qs).tolist())), "T_Q_quantiles": dict(zip(qs, np.percentile(tq, qs).tolist())),
            "sat_P": float((np.abs(tp) > 0.95).mean()), "sat_Q": float((np.abs(tq) > 0.95).mean()), "n_P": int(len(tp)), "n_Q": int(len(tq)),
            "approximation_bias_covered": False}


def measure(fix_dir: Path, out: Path, smoke: bool) -> None:
    t_all = time.time()
    S = torch.load(fix_dir / "fixture.pt", map_location="cpu", weights_only=False); man = json.load(open(fix_dir / "manifest.json"))
    steps = 1000 if not smoke else 40
    H = {r: {v: S["features"][r][v]["h"].numpy().astype(np.float64) for v in ("A1", "A2")} for r in ROLES}
    ids = {r: S["roles"][r].numpy() for r in ROLES}
    d = H["fit"]["A1"].shape[1]
    # standardisation per view on FIT (all FIT base images)
    std = {v: standardizer(H["fit"][v]) for v in ("A1", "A2")}
    X = {r: {v: (H[r][v] - std[v][0]) / std[v][1] for v in ("A1", "A2")} for r in ROLES}
    # channel per role / view (coupled across t); role codes 0/1/2
    CH = {r: {v: brownian(X[r][v], ids[r], ROLES[r], 1 if v == "A1" else 2) for v in ("A1", "A2")} for r in ROLES}
    R1, R2 = haar_orthogonal(d, ROT_SEEDS[0]), haar_orthogonal(d, ROT_SEEDS[1])
    # pair index sets
    nf = len(ids["fit"]); gf = np.random.default_rng(SPLIT_SEED + 149); pf = gf.permutation(nf)
    kf = N_FIT if not smoke else nf // 3
    fit_P, fit_QL, fit_QR = pf[:kf], pf[kf:2 * kf], pf[2 * kf:3 * kf]
    nv = len(ids["val"]); kp, kq = (N_TUNE_P, N_TUNE_Q) if not smoke else (nv // 2, nv // 4); pv = np.random.default_rng(SPLIT_SEED + 150).permutation(nv)
    tune_P, tune_QL, tune_QR = pv[:kp], pv[kp:kp + kq], pv[kp + kq:kp + 2 * kq]
    EP, EQL, EQR = (S["eval_pools"][k].numpy() for k in ("P", "Q_left", "Q_right"))
    perm_rng = np.random.default_rng(PERM_SEED)
    derange = []  # shared independent-pair nulls on the P pool: B permutations (rejecting fixed points)
    while len(derange) < (B_PERM if not smoke else 20):
        p = perm_rng.permutation(len(EP))
        if (p != np.arange(len(EP))).all():
            derange.append(p)
    repair = [np.random.default_rng(PERM_SEED + 1 + i).permutation(len(EQR)) for i in range(N_REPAIR if not smoke else 8)]
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=DEV)

    def coords(setting: str, role: str):
        t = {"identity_t0": 0.0, "orthogonal_refit_t0": 0.0, "identity_t0.25": 0.25, "identity_t1": 1.0}[setting]
        a1, a2 = CH[role]["A1"][t], CH[role]["A2"][t]
        if setting == "orthogonal_refit_t0":
            a1, a2 = a1 @ R1.T, a2 @ R2.T
        return a1, a2

    results, arrays = {"run": S["run"], "dataset": S["dataset"], "fixture_manifest": man, "settings": {}, "transport_check": {}}, {}
    results["protocol"] = {"n_fit_P": int(kf), "n_fit_Q": int(kf), "n_tune_P": int(kp), "n_tune_Q": int(kq), "n_eval_P": int(len(EP)), "n_eval_Q": int(len(EQL)),
                           "steps_max": steps, "lrs": LRS, "bw_mults": BWS, "rff_D": RFF_D, "fit_seeds": FIT_SEEDS, "times": TIMES, "rot_seeds": ROT_SEEDS, "noise_seed": NOISE_SEED,
                           "standardizer": {v: {"gamma": std[v][1]} for v in std}, "logit_coordinate": {"vcs_mlp": "f, T = tanh f", "js_mlp": "half_logit_f (D = sigmoid 2f), T = tanh f", "rff_ridge_tanh": "f, T = tanh f"},
                           "permutations": len(derange), "repairings": len(repair), "device": str(DEV)}
    id_models = {}
    for setting in SETTINGS:
        a1f, a2f = coords(setting, "fit"); a1v, a2v = coords(setting, "val"); a1e, a2e = coords(setting, "eval")
        Pf = (T(a1f[fit_P]), T(a2f[fit_P])); Qf = (T(a1f[fit_QL]), T(a2f[fit_QR]))
        Pt = (T(a1v[tune_P]), T(a2v[tune_P])); Qt = (T(a1v[tune_QL]), T(a2v[tune_QR]))
        Pe = (T(a1e[EP]), T(a2e[EP])); Qe_left, Qe_right = T(a1e[EQL]), T(a2e[EQR])
        results["settings"][setting] = {}
        for est in ESTIMATORS:
            for seed in FIT_SEEDS:
                t0 = time.perf_counter()
                fit = fit_mlp("vcs" if est == "vcs_mlp" else "js", Pf, Qf, Pt, Qt, d, seed, steps) if est != "rff_ridge_tanh" else fit_rff(Pf, Qf, Pt, Qt, d, seed)
                m = fit["model"]
                tp = torch.tanh(critic_f(est, m, Pe[0], Pe[1])).cpu().numpy(); tq = torch.tanh(critic_f(est, m, Qe_left, Qe_right)).cpu().numpy()
                r = readout(tp, tq)
                r["J_fit"] = float(ev.j_terms(critic_f(est, m, Pf[0], Pf[1]), critic_f(est, m, Qf[0], Qf[1])).mean())
                r["J_tune"] = float(ev.j_terms(critic_f(est, m, Pt[0], Pt[1]), critic_f(est, m, Qt[0], Qt[1])).mean())
                # independent-pair null on the P pool (derangements), shared permutations -> permutation p for the observed J
                null = []
                for p in derange:
                    tpn = torch.tanh(critic_f(est, m, Pe[0], Pe[1][torch.as_tensor(p, device=DEV)])).cpu().numpy()
                    null.append(float((tpn - 0.5 * tpn ** 2).mean() + (-tq - 0.5 * tq ** 2).mean()))
                null = np.array(null); r["null_J_mean"] = float(null.mean()); r["null_J_sd"] = float(null.std(ddof=1))
                r["perm_p"] = float((1 + (null >= r["J_common"]).sum()) / (1 + len(null)))
                # pair_var_conditional_batch: fixed critic, fixed base images, re-pair the Q pools
                rep = []
                for p in repair:
                    tqr = torch.tanh(critic_f(est, m, Qe_left, Qe_right[torch.as_tensor(p, device=DEV)])).cpu().numpy()
                    rep.append(float((tp - 0.5 * tp ** 2).mean() + (-tqr - 0.5 * tqr ** 2).mean()))
                r["pair_var_conditional_batch"] = float(np.var(rep, ddof=1)); r["repair_J_mean"] = float(np.mean(rep))
                r["selected"] = fit["selected"]; r["candidates"] = fit["candidates"]; r["fit_seconds"] = fit["fit_seconds"]; r["cell_seconds"] = time.perf_counter() - t0
                results["settings"][setting][f"{est}/seed{seed}"] = r
                arrays[f"{setting}/{est}/seed{seed}/T_P"] = tp.astype(np.float32); arrays[f"{setting}/{est}/seed{seed}/T_Q"] = tq.astype(np.float32)
                arrays[f"{setting}/{est}/seed{seed}/null_J"] = null.astype(np.float32)
                if setting == "identity_t0" and seed == 0:
                    id_models[est] = m
                print(f"[{S['run']} {setting} {est} s{seed}] J {r['J_common']:.4f} ± {r['se_J']:.4f} S_plug {r['S_plugin']:.4f} null {r['null_J_mean']:+.4f} p {r['perm_p']:.3f} "
                      f"sat {r['sat_P']:.2f}/{r['sat_Q']:.2f} ({r['cell_seconds']:.0f}s)", flush=True)
            atomic_write_json(out.with_suffix(".partial.json"), results)
        if setting == "orthogonal_refit_t0":  # transport check of the identity_t0 seed-0 critics onto rotated inputs
            Pe0 = (T(coords("identity_t0", "eval")[0][EP]), T(coords("identity_t0", "eval")[1][EP]))
            Rt = torch.as_tensor(np.block([[R1.T, np.zeros((d, d))], [np.zeros((d, d)), R2.T]]), dtype=torch.float32, device=DEV)
            for est, m in id_models.items():
                if est == "rff_ridge_tanh":
                    R1t, R2t = (torch.as_tensor(R, dtype=torch.float64, device=DEV) for R in (R1, R2))
                    mt = RFFCritic(torch.cat([R1t @ m.W[:d], R2t @ m.W[d:]], 0), m.b, m.v)  # (R x)^T W' = x^T W  <=>  W' = R W per block
                    f0 = m.f(Pe0[0], Pe0[1]); f1 = mt.f(Pe[0], Pe[1])
                else:
                    import copy
                    mt = copy.deepcopy(m); W = mt.net[0].weight.data; mt.net[0].weight.data = W @ Rt
                    f0 = m(Pe0[0], Pe0[1]).double(); f1 = mt(Pe[0], Pe[1]).double()
                results["transport_check"][est] = {"max_abs_f_diff": float((f0 - f1).abs().max()), "pass_1e-3": bool(float((f0 - f1).abs().max()) < 1e-3),
                                                   "note": "implementation check only: identity-setting critic with transported first layer / frequencies on rotated inputs"}
            print(f"[{S['run']}] transport check {results['transport_check']}", flush=True)
    results["seconds"] = time.time() - t_all
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), **arrays, eval_P_rows=EP, eval_QL_rows=EQL, eval_QR_rows=EQR, derangements=np.array(derange))
    atomic_write_json(out.with_suffix(".json"), results)
    print(f"[{S['run']}] done ({results['seconds']:.0f}s) -> {out}.json / .npz", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture"); f.add_argument("--dataset", required=True, choices=list(ROOTS)); f.add_argument("--run", required=True); f.add_argument("--out", required=True); f.add_argument("--smoke", action="store_true")
    m = sub.add_parser("measure"); m.add_argument("--fixture", required=True); m.add_argument("--out", required=True); m.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    if a.cmd == "fixture":
        build_fixture(a.dataset, a.run, Path(a.out) / a.run, a.smoke)
    else:
        measure(Path(a.fixture), Path(a.out), a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
