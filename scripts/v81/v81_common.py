"""Shared paths and helpers for the V8.1 D02 / D03 / D04a diagnostics (CPU only; read-only on every V8 / V7 artefact).

The package reference definitions (v8_1/package_v81_20261015/reference/core.py) are imported by path and used unchanged;
nothing here replaces the frozen V8-04 / V8-05 evaluators.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "v8_1" / "package_v81_20261015"
REF = PKG / "reference"
OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v81")
NORM = OUT / "normalized"
DIAG = OUT / "diag"
RESULTS = REPO / "reports" / "V8_1_DIAGNOSTICS" / "results"
UNITS_V8 = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
KERNELS = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/kernels")
V7_OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v7")
MANIFEST = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data/manifest_images.jsonl")
V8_EVAL_JSON = REPO / "reports" / "V8_UTILITY" / "V8_EVAL_results" / "v8_eval.json"   # V8-04 results (frozen evaluator), cross-check only
DATASET = "EvalMuse-public-train"          # adapt_v8.py's dataset string; every DEV / CAL method must share it
ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"]
SCALE_ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1"]
SEEDS = [0, 1, 2]
TUNE_SEED = 17
TUNE_LRS = ["5e-06", "1e-05", "2e-05"]
ROLES_LABELLED = ["DEV", "CAL_SELECT", "CAL_PROB"]
LEVELS = np.arange(1.0, 6.0)
BOOT_REPS, BOOT_SEED = 2000, 2718
PYTHON = sys.executable


def load_core():
    spec = importlib.util.spec_from_file_location("v81_core", REF / "core.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def csv_write(p, rows, fields=None):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt_cell(r.get(k)) for k in fields})


def fmt_cell(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(fmt_cell(x) for x in v) + "]"
    return v


def manifest(role=None):
    rows = read_jsonl(MANIFEST)
    return rows if role is None else [r for r in rows if r["role"] == role]


def fit_unit(arm, seed):
    return UNITS_V8 / f"V8-FIT-{arm}-s{seed}"


def tune_unit(arm, lr):
    return UNITS_V8 / f"V8-TUNE-{arm}-lr{lr}"


def unit_log(unit):
    return jload(Path(unit) / "log.json")


def ckpt_id(log):
    return f"best_e{log['selected_epoch']}"


def tails(q):
    q = np.asarray(q, float)
    return np.cumsum(q[..., ::-1], axis=-1)[..., ::-1][..., 1:]


def pooled_spearman(x, y):
    """Global average-rank Spearman (the V8-04 / V8-05 pooled readout); None if either side is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def cluster_groups(clusters):
    g = defaultdict(list)
    for i, c in enumerate(clusters):
        g[str(c)].append(i)
    keys = sorted(g)
    return keys, [np.asarray(g[k], int) for k in keys]


def pooled_spearman_cluster_bootstrap(S, y, clusters, reps=BOOT_REPS, seed=BOOT_SEED):
    """Rows of S = one model's scores on the same images; y = human means; clusters = resampling unit per image.
    Every replicate resamples clusters with replacement, rebuilds the paired rows and re-ranks (package 05 §5), so the
    returned replicate matrix (reps x models) supports paired differences between models.  Development interval, no p."""
    S = np.asarray(S, float)
    y = np.asarray(y, float)
    if S.ndim != 2 or S.shape[1] != len(y) or len(clusters) != len(y):
        raise ValueError("S must be (models x images) aligned with y and clusters")
    keys, groups = cluster_groups(clusters)
    level = np.array([pooled_spearman(s, y) if pooled_spearman(s, y) is not None else np.nan for s in S])
    rng = np.random.default_rng(seed)
    boot = np.full((reps, S.shape[0]), np.nan)
    G = len(groups)
    for r in range(reps):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([groups[j] for j in pick])
        ry = rankdata(y[idx])
        if np.all(ry == ry[0]):
            continue
        R = rankdata(S[:, idx], axis=1)
        Rc = R - R.mean(1, keepdims=True)
        yc = ry - ry.mean()
        den = np.sqrt((Rc * Rc).sum(1) * (yc * yc).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[r] = (Rc @ yc) / den
    ci = np.nanquantile(boot, [0.025, 0.975], axis=0).T
    return {"level": level, "boot": boot, "ci95": ci, "n_clusters": G, "reps": reps, "seed": seed,
            "status": "development_interval_no_p_value"}


def macro_decomposition(pool_errors):
    """pool_errors: list of 1-d arrays e_i = mu_i - Hbar_i, one per pool (uniform image weight inside a pool, uniform pool weight).
    MSE_macro = mean_p MSE_p = mean_p b_p^2 + mean_p within_p = [bbar^2 + var_p(b_p)] + mean_p within_p (exact identities)."""
    b = np.array([e.mean() for e in pool_errors])
    within = np.array([((e - e.mean()) ** 2).mean() for e in pool_errors])
    mse = np.array([(e * e).mean() for e in pool_errors])
    out = {"mse_macro": float(mse.mean()), "bias_sq_macro": float((b * b).mean()), "within_macro": float(within.mean()),
           "grand_bias": float(b.mean()), "grand_bias_sq": float(b.mean() ** 2), "between_pool_bias_var": float(b.var()),
           "n_pools": int(len(b))}
    out["identity_residual"] = out["mse_macro"] - out["bias_sq_macro"] - out["within_macro"]
    out["identity_residual_between"] = out["bias_sq_macro"] - out["grand_bias_sq"] - out["between_pool_bias_var"]
    return out


def micro_decomposition(errors):
    e = np.asarray(errors, float)
    b = float(e.mean())
    return {"mse_micro": float((e * e).mean()), "bias_sq_micro": b * b, "within_micro": float(((e - b) ** 2).mean()), "n_images": int(len(e))}


def model_name(m):
    return f"{m['method']}|s{m['seed']}|{m['checkpoint_id']}"


def group_pools(rows):
    """rows of one model -> {pool_id: rows sorted by candidate_index}; raises on inconsistent prompt / cluster inside a pool."""
    pools = defaultdict(list)
    for r in rows:
        pools[r["pool_id"]].append(r)
    for p, rs in pools.items():
        rs.sort(key=lambda r: r["candidate_index"])
        if len({r["prompt_id"] for r in rs}) != 1 or len({r.get("cluster_id", r["prompt_id"]) for r in rs}) != 1:
            raise ValueError(f"inconsistent prompt / cluster in pool {p}")
    return dict(pools)


def split_models(rows):
    models = defaultdict(list)
    for r in rows:
        models[(r["dataset"], r["role"], r["method"], r["seed"], r["checkpoint_id"])].append(r)
    return dict(models)


def human_mean_of(r, core):
    """Exact-tie-consistent human mean: sum(votes) / n (equal rationals -> identical doubles).  The package reference recomputes
    human_mean as hist @ levels, which leaves 1-ulp differences between equal means of different vote patterns (DEV: 101 of 2,855
    exact within-prompt ties are split), so exact ties would enter pair-accuracy denominators; V8.1 D03 uses this exact mean."""
    if "votes" in r:
        v = r["votes"]
        return float(sum(v)) / len(v)
    if "human_mean" in r:
        return float(r["human_mean"])
    if "human_hist" in r:
        return float(np.asarray(r["human_hist"], float) @ LEVELS)
    return float(core.histogram(r["votes"]) @ LEVELS)
