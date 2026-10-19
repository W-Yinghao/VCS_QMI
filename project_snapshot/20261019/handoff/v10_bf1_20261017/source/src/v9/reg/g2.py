"""V9-G2 — the V9-G1 extension (training-sample fraction × training-correspondence mismatch) on the second modality pair, EPFL RGB–NIR
(package v9 docs/06 §2 "EPFL作为第二模态数据的条件扩展"; owner 2026-10-15: every dataset already on the server is cleared for use).

Data and registration machinery = r6 REG-P3 (src/v9/reg/r6_reg_p3.py, verbatim copy): its cache of 420 centred 512-px windows (RGB reference,
NIR moving), manifest (roles, windows -> scene units with near-duplicates merged, 9 categories), perturbations (10 rigid per CAL / DEV window),
Inputs3 (NIR moving through the 'nir' key), unit-pooled lexicographic CAL selection and DEV evaluation (register_one), thresholds err_full < 42 /
< 9 = 10.5 / 2.25 native px.  The training path and the two axes = V9-G1 (src/v9/reg/g1.py), reused by IN-PROCESS parameterisation: inside
`activate()` the g1 module's dataset hooks (unit root, FIT sampler, state hash, r6 roots) are pointed at the EPFL objects and restored on exit —
g1.py itself is not modified (the running G1 jobs import the unmodified file; tests/v9/test_reg_g2.py asserts g1 is byte-identical before / after).

Axes (per unit: method × fit_fraction × train_mismatch × seed):
  fit_fraction   {0.25, 1.0} of REG-P3's 330 FIT units (scenes; a unit = its windows, 336 windows in all), sampled by UNIT with the 9 categories
                 as strata (largest-remainder allocation, ties by category name), seeded by the unit seed; windows / patches are never the sampling
                 unit.  Realised units, windows and counts per category are written into train.json / comir.json.
  train_mismatch {0, 0.2}: g1's generator unchanged — exactly round(ε n) of the positive correspondences of every training step get a displaced
                 moving-side position drawn from the SAME training crop, rejecting positions closer than 2 patch radii (critics 15 px, kernel 4 px,
                 CoMIR 64 px) to the true one; conditional Q (not an unconditional pixel product); the same draws for vcs / logistic / infonce;
                 CoMIR crop-level; no mismatch label reaches any loss, model, selection or registration.
Reuse of r6 REG-P3 units (package: "only if all hashes match"): seed, FIT set with the data hashes, ε = 0, fraction 1, steps / epochs and recipe
must be identical; r6 ran seeds 0–2 and the package fixes 901–903, so nothing is reusable unless the owner maps the seeds (recorded, not done).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np

from . import g1 as G  # (V9-G1, reused; not modified — the same module object as v9.reg.g1 everywhere)
from . import r6_reg_p3 as P3  # (REG-P3, verbatim copy; read-only)

P = G.P; Q = G.Q
assert P3.P is P and P3.Q is Q, "r6_reg_p3 and g1 must share the r6_reg_pilot / r6_reg_p2 module objects (the kernel monkeypatch relies on it)"
P3_ROOT = Path(os.environ.get("V9_REG_P3_ROOT", "/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3"))   # r6 REG-P3 cache / manifest / perturbations / results / runs (read-only)
V9_ROOT = Path(os.environ.get("V9_REG_EPFL_ROOT", "/projects/EEG-foundation-model/yinghao/outputs_v9/reg_epfl"))
METHODS, FRACTIONS, EPSILONS, SEEDS = G.METHODS, G.FRACTIONS, G.EPSILONS, G.SEEDS
PATCH_RADIUS = G.PATCH_RADIUS
R6_SEEDS = (0, 1, 2)
DATASET = "EPFL-RGB-NIR"
THRESHOLDS_NATIVE = {"primary_px": 10.5, "strict_px": 2.25, "err_full_factor": 4.0}        # err_full = 4 x native px (REG-P3's ERR_UNIT)


# ------------------------------------------------------------------------------------------------------------ units
def unit_id(method, frac, eps, seed):
    return f"V9-G2-{method}-frac{frac:g}-eps{eps:g}-s{seed}"


def all_units():
    return [(unit_id(m, f, e, s), m, f, e, s) for m in METHODS for f in FRACTIONS for e in EPSILONS for s in SEEDS]


def kernel_units():
    return [(unit_id("kernel", f, e, 901), "kernel", f, e, 901) for f in FRACTIONS for e in EPSILONS]


def unit_root(uid):
    return V9_ROOT / "units" / uid


def result_path(uid, method, seed, smoke=False):
    name = method + ("" if method in ("mi", "nmi") else f"_s{seed}") + ("_smoke" if smoke else "")
    return unit_root(uid) / "p3" / ("smoke/" if smoke else "") / "results" / f"{name}.json"


# ------------------------------------------------------------------------------------------------------------ manifest helpers (EPFL)
def load_manifest(root=P3_ROOT, smoke=False):
    return P.load_manifest(Path(root), smoke)


def window_unit(manifest):
    return {w: v["unit"] for w, v in manifest["windows"].items()}


def unit_category(manifest):
    return {u: v["category"] for u, v in manifest["units"].items()}


def sample_fit_units(fit_windows, frac, seed, manifest):
    """Stratified (by category) random sample of round(frac × n_units) FIT units among the units that own the given FIT windows (largest-remainder
    allocation, ties by category name), seeded by the unit seed; returns (sorted windows of the chosen units, counts of chosen units per category).
    frac = 1 returns every window.  Windows are never the sampling unit."""
    wu = window_unit(manifest); uc = unit_category(manifest)
    units = sorted(set(wu[w] for w in fit_windows)); n = int(round(frac * len(units)))
    strata = sorted(set(uc[u] for u in units)); members = {s: [u for u in units if uc[u] == s] for s in strata}
    quota = {s: frac * len(members[s]) for s in strata}; alloc = {s: int(math.floor(quota[s])) for s in strata}
    for s in sorted(strata, key=lambda s: (-(quota[s] - alloc[s]), s))[: n - sum(alloc.values())]:
        alloc[s] += 1
    g = np.random.default_rng(int(hashlib.sha256(f"V9-G2|fit|{seed}|{frac:g}".encode()).hexdigest()[:16], 16)); chosen = []
    for s in strata:
        chosen += [members[s][i] for i in sorted(g.choice(len(members[s]), alloc[s], replace=False))]
    chosen = set(chosen); wins = sorted(w for w in fit_windows if wu[w] in chosen)
    return wins, {s: alloc[s] for s in strata}


def state_hash_epfl(manifest, fit_windows, seed, frac, eps, method, steps):
    """As g1.state_hash, with REG-P3's per-pair sha256 (RGB and NIR files) of every FIT window."""
    h = hashlib.sha256()
    for w in sorted(fit_windows):
        pr = manifest["pairs"][manifest["windows"][w]["pair"]]
        h.update(f"{w}|{pr['sha256_rgb']}|{pr['sha256_nir']}|".encode())
    h.update(f"seed={seed}|frac={frac:g}|eps={eps:g}|method={method}|steps={steps}|size={manifest['size']}|down={manifest['down']}|dataset={DATASET}".encode())
    return h.hexdigest()[:16]


def r6_reusable(manifest, fit_full, method, frac, eps, seed, p3_root=P3_ROOT):
    """Package rule: an r6 REG-P3 full-data clean unit is reusable only when seed, FIT set (+ data hashes), ε 0, fraction 1 and recipe match."""
    if method not in ("vcs", "logistic", "infonce", "comir") or frac != 1.0 or eps != 0.0 or seed not in R6_SEEDS:
        return None
    run = Path(p3_root) / "runs" / f"{method}_s{seed}"; meta = run / ("comir.json" if method == "comir" else "train.json")
    if not meta.exists():
        return None
    j = json.load(open(meta))
    if sorted(j.get("fit", [])) != sorted(fit_full) or (method != "comir" and j.get("steps") != 3000) or (method == "comir" and j.get("epochs") != 23):
        return None
    return {"run": str(run), "state_hash": state_hash_epfl(manifest, fit_full, seed, 1.0, 0.0, method, 3000 if method != "comir" else 23)}


def code_hash():
    return hashlib.sha256((inspect.getsource(sample_fit_units) + inspect.getsource(cmd_register) + inspect.getsource(activate) + G.code_hash()).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------------------ in-process parameterisation of g1
_HOOKS = ("unit_root", "sample_fit", "state_hash", "P1_ROOT", "P2_ROOT", "V9_ROOT")


@contextlib.contextmanager
def activate(manifest, p3_root=P3_ROOT):
    """Point g1's dataset hooks at EPFL for the duration of the block; restore them afterwards (g1's source is never touched)."""
    saved = {k: getattr(G, k) for k in _HOOKS}
    try:
        G.unit_root = unit_root; G.V9_ROOT = V9_ROOT; G.P1_ROOT = Path(p3_root); G.P2_ROOT = Path(p3_root)
        G.sample_fit = lambda fit, frac, seed: sample_fit_units(fit, frac, seed, manifest)
        G.state_hash = state_hash_epfl
        yield G
    finally:
        for k, v in saved.items():
            setattr(G, k, v)


# ------------------------------------------------------------------------------------------------------------ unit roots (REG-P3 layout, symlinked read-only)
def ensure_layout(uid, method, seed, p3_root=P3_ROOT, smoke=False):
    """<unit>/p3/{cache -> REG-P3 cache, manifest.json, perturbations.json -> REG-P3's, runs/<method>_s<seed> -> <unit>/run, results/};
    smoke: <unit>/p3/smoke/{cache, manifest_smoke.json, perturbations_smoke.json -> <V9_ROOT>/smoke/p3root/..., runs/..., results/}."""
    u = unit_root(uid); p3 = u / "p3" / ("smoke" if smoke else ""); (p3 / "runs").mkdir(parents=True, exist_ok=True); (p3 / "results").mkdir(exist_ok=True); (u / "run").mkdir(exist_ok=True)
    src = Path(p3_root); suf = "_smoke" if smoke else ""
    for link, target in ((p3 / "cache", src / "cache"), (p3 / f"manifest{suf}.json", src / f"manifest{suf}.json"), (p3 / f"perturbations{suf}.json", src / f"perturbations{suf}.json"),
                         (p3 / "runs" / f"{method}_s{seed}", u / "run")):
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target)
    return u


def make_smoke_root(dst, p3_root=P3_ROOT):
    """A REG-P3-shaped smoke root from the existing cache: REG-P3's smoke_subset (4 FIT windows; 2 CAL; 2 further CAL windows relabelled 'DEV' —
    no DEV window is read), cache files symlinked, manifest_smoke.json written, perturbations_smoke.json by REG-P3's cmd_perturb."""
    dst = Path(dst); m = load_manifest(p3_root, False); sel = P3.smoke_subset(m["roles"], m["windows"]); (dst / "cache").mkdir(parents=True, exist_ok=True)
    for w in sel:
        link = dst / "cache" / f"{w}.npz"
        if not link.is_symlink() and not link.exists():
            link.symlink_to(Path(p3_root) / "cache" / f"{w}.npz")
    sm = dict(m); sm["roles"] = sel; sm["windows"] = {w: m["windows"][w] for w in sel}; sm["smoke"] = True
    (dst / "manifest_smoke.json").write_text(json.dumps(sm, indent=1))
    if not (dst / "perturbations_smoke.json").exists():
        P3.cmd_perturb(argparse.Namespace(root=dst, smoke=True))
    return dst


# ------------------------------------------------------------------------------------------------------------ training (g1's functions on the EPFL objects)
def fit_full_windows(manifest, root):
    return sorted(w for w, r in manifest["roles"].items() if r == "FIT" and (Path(root) / "cache" / f"{w}.npz").exists())


def cmd_train(uid, objective, frac, eps, seed, steps=3000, smoke=False, p3_root=P3_ROOT):
    m = load_manifest(p3_root, smoke)
    with activate(m, p3_root):
        rc = G.cmd_train(uid, objective, frac, eps, seed, steps=steps, smoke=smoke, p1_root=Path(p3_root))
    _annotate_run(uid, "train.json", m); return rc


def cmd_comir(uid, frac, eps, seed, smoke=False, p3_root=P3_ROOT, comir_epochs=0):
    m = load_manifest(p3_root, smoke)
    with activate(m, p3_root):
        rc = G.cmd_comir(uid, frac, eps, seed, smoke=smoke, p1_root=Path(p3_root), comir_epochs=comir_epochs)
    _annotate_run(uid, "comir.json", m); return rc


def _annotate_run(uid, name, manifest):
    """Add the EPFL unit facts to the run record written by g1 (which knows windows only): units, counts per category, dataset, g2 code hash."""
    f = unit_root(uid) / "run" / name
    if f.exists():
        j = json.load(open(f)); wu = window_unit(manifest); units = sorted(set(wu[w] for w in j.get("fit", []) if w in wu))
        j.update({"dataset": DATASET, "fit_units": units, "n_fit_units": len(units), "fit_counts_by_category": j.pop("fit_counts_by_slide", j.get("fit_counts_by_category")), "g2_code_hash": code_hash(),
                  "n_fit_full_units": len(set(wu[w] for w, r in manifest["roles"].items() if r == "FIT"))})
        json.dump(j, open(f, "w"), indent=1)


# ------------------------------------------------------------------------------------------------------------ registration (REG-P3's register_one through the unit root)
def cmd_register(uid, method, seed, smoke=False, n_pert=P.N_PERT, frac=1.0, eps=0.0, p3_root=P3_ROOT):
    """REG-P3's register_one on the unit's run (critics: unit-pooled CAL selection over the 12 checkpoints, DEV; CoMIR: encode-once; kernel: the
    REG-P3 path replayed with g1's fit (sampled FIT units, mismatch) instead of P.fit_kernel; MI / NMI are reused, never re-run here)."""
    if method in ("mi", "nmi"):
        raise ValueError("MI / NMI are reused from REG-P3 (mi_nmi_reuse_check), not registered per unit")
    u = ensure_layout(uid, method, seed, p3_root, smoke); root = u / "p3" / ("smoke" if smoke else ""); m = load_manifest(root, smoke)
    torch, nn, F = P.torch_mods(); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(root=root, smoke=smoke, method=method, seed=seed, n_pert=n_pert, methods=[method], seeds=[seed])
    if method == "kernel":
        orig = P.fit_kernel; fit, counts = sample_fit_units(fit_full_windows(m, root), frac, seed, m)
        P.fit_kernel = lambda torch_, F_, root_, _fit, mult, s, dev_: G.fit_kernel_g1(torch_, F_, root_, fit, mult, s, dev_, eps)
        try:
            rc = P3.register_one(args, torch, nn, F, dev)
        finally:
            P.fit_kernel = orig
        res = result_path(uid, method, seed, smoke); r = json.load(open(res))
        r.update({"unit": uid, "dataset": DATASET, "fit_fraction": frac, "train_mismatch": eps, "fit": fit, "fit_units": sorted(set(window_unit(m)[w] for w in fit)), "fit_counts_by_category": counts,
                  "n_fit_units": sum(counts.values()), "exclusion_radius_px": 2 * PATCH_RADIUS["kernel"], "g2_code_hash": code_hash()}); json.dump(r, open(res, "w"), indent=1); return rc
    rc = P3.register_one(args, torch, nn, F, dev)
    res = result_path(uid, method, seed, smoke); r = json.load(open(res)); rec = json.load(open(u / "run" / ("comir.json" if method == "comir" else "train.json")))
    r.update({"unit": uid, "dataset": DATASET, "fit_fraction": frac, "train_mismatch": eps, "g2_code_hash": code_hash(), "fit": rec.get("fit"), "fit_units": rec.get("fit_units"), "n_fit_units": rec.get("n_fit_units"),
              "fit_counts_by_category": rec.get("fit_counts_by_category"), "realised_mismatch_fraction": rec.get("realised_mismatch_fraction"), "train_state_hash": rec.get("state_hash")})
    json.dump(r, open(res, "w"), indent=1); return rc


def mi_nmi_reuse_check(p3_root=P3_ROOT):
    """MI / NMI have no training: REG-P3's DEV / CAL results are reused iff the perturbations and the manifest they read are the ones on disk."""
    return G.mi_nmi_reuse_check(p2_root=Path(p3_root), p1_root=Path(p3_root))


# ------------------------------------------------------------------------------------------------------------ unit table
def unit_table(p3_root=P3_ROOT):
    m = load_manifest(p3_root, False); fit_full = fit_full_windows(m, p3_root); rows = []
    for uid, method, frac, eps, seed in all_units() + kernel_units():
        fit, counts = sample_fit_units(fit_full, frac, seed, m); reuse = r6_reusable(m, fit_full, method, frac, eps, seed, p3_root); done = result_path(uid, method, seed).exists()
        rows.append({"unit_id": uid, "dataset": DATASET, "method": method, "seed": seed, "fit_fraction": frac, "train_mismatch": eps, "n_fit_units": sum(counts.values()), "n_fit_windows": len(fit),
                     "fit_units_by_category": json.dumps(counts), "reuse": "r6:" + reuse["run"] if reuse else "none_seed_differs" if (frac == 1.0 and eps == 0.0 and method != "kernel") else "not_applicable",
                     "status": "done" if done else "planned_not_submitted", "state_hash": state_hash_epfl(m, fit, seed, frac, eps, method, 3000 if method not in ("comir", "kernel") else 23)})
    return rows
