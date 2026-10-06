"""P142 (MV6-I1) — Table 3 completion on a fixed encoder: same-target kernel critic and conditional HSIC beside A1 / A3 (P116).

Per repeat the P116 construction is replayed line for line (fresh N | Y, disjoint FIT / EVAL / POOL of n items, within-class POOL negatives,
B shared within-class permutations of N on EVAL, the FIT 80 / 20 split from `split_seed`).  The numpy stream (`rng`) is consumed only by
these draws, exactly as in P116, so with the P116 seeds the draws are identical; the new methods take randomness from their own generators.
  A1        ridge_tanh_calibrated (P116 A1)                              — reproduction gate against reports/P116/*.json
  A3@B      matched logistic L-BFGS, B in {50, 200, 800} closures (P116) — reproduction gate
  K1        same-target kernel critic: A1's two-stage solver on phi_K = [psi(h)(2N-1), 1], psi = RFF of standardised h (D = 1024,
            Gaussian; bandwidth in {0.5, 1, 2} x FIT median pairwise distance chosen by VAL J); permutation test with the shared PN
  H1        conditional HSIC, class-wise (cond_test_t1.hsic_class_stat: median bandwidth within class, delta kernel on N); shared PN
  H2        conditional HSIC, deep kernel (cond_test_t1.fit_deep_kernel, 300 steps, own numpy generator); shared PN
Nothing is selected on EVAL.  HSIC statistics stay on their native scale.

    python scripts/p142_table3.py --features <P45 dir> --family colour --cells 'colour_s0.2:2000' --repeats 100 --seed 612 \
        --check reports/P116/full_simclr_colour.json --out reports/P142/A_simclr_colour
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

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cond_test_t1 import Family, class_blocks, class_kernels, draw_sample, fit_deep_kernel, hsic_class_stat  # noqa: E402
from precheck_d_tests import DEVICE, within_class_pool  # noqa: E402
from vcs_measure.solve import Design, js_lbfgs, make_split, perm_tests, ridge_tanh_calibrated, risks  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

BUDGETS = (50, 200, 800)
RFF_D = 1024
RFF_BW = (0.5, 1.0, 2.0)
DEEP_STEPS = 300


def rff_map(d: int, sigma: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d, RFF_D, generator=g, dtype=torch.float64) / sigma
    b = 2 * math.pi * torch.rand(RFF_D, generator=g, dtype=torch.float64)
    return lambda z: math.sqrt(2.0 / RFF_D) * torch.cos(z.double().cpu() @ W + b).to(z.device)


def median_dist(z: torch.Tensor, seed: int, m: int = 1000) -> float:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(z), generator=g)[:m]
    d = torch.cdist(z[idx].double().cpu(), z[idx].double().cpu())
    return float(torch.median(d[d > 0]))


def k1_fit(zf, nf, nf_neg, ti, vi, split_seed: int):
    """Kernel A1: the A1 two-stage solver on RFF features; bandwidth by VAL J.  Returns (critic, psi, meta)."""
    t0 = time.perf_counter()
    med = median_dist(zf, split_seed + 11)
    best = None
    per_bw = []
    for k, mult in enumerate(RFF_BW):
        psi = rff_map(zf.shape[1], mult * med, split_seed + 101 + k)
        D = Design.build(psi(zf), nf, nf_neg, ti, vi)
        cr = ridge_tanh_calibrated(D)
        per_bw.append({"bw_mult": mult, "val_J": cr.meta["val_J"], "lam": cr.meta["lam"], "c": cr.meta["c"]})
        if best is None or cr.meta["val_J"] > best[0].meta["val_J"]:
            best = (cr, psi, D, mult)
    cr, psi, D, mult = best
    cr.meta = {**cr.meta, "algo": "K1_rff_ridge_tanh", "rff_D": RFF_D, "median_dist": med, "bw_mult": mult, "per_bw": per_bw,
               "seconds_total": time.perf_counter() - t0}
    return cr, psi, D


def hsic_test(K, blocks, ne, PN, n, delta):
    t0 = time.perf_counter()
    obs = hsic_class_stat(K, blocks, ne, n)
    null = np.array([hsic_class_stat(K, blocks, pn, n) for pn in PN])
    p = float((1 + (null >= obs).sum()) / (1 + len(null)))
    return {"stat": obs, "p": p, "reject": p <= delta, "seconds": time.perf_counter() - t0}


def run_repeat(fam: Family, n: int, rng, *, mode: str, strength: float, delta: float, perms: int, split_seed: int, budgets=BUDGETS,
               deep_steps: int = DEEP_STEPS) -> dict:
    t_rep = time.perf_counter()
    # ---- P116 construction, verbatim order of rng use
    used = set()
    Hf, Yf, Nf, _ = draw_sample(fam, n, rng, mode, strength, used)
    He, Ye, Ne, _ = draw_sample(fam, n, rng, mode, strength, used)
    Hp, Yp, Np, _ = draw_sample(fam, n, rng, mode, strength, used)
    mu, sd = Hf.mean(0), Hf.std(0) + 1e-6
    zf, ze = ((Hf - mu) / sd).to(DEVICE), ((He - mu) / sd).to(DEVICE)
    nf, ne = torch.as_tensor(Nf).to(DEVICE), torch.as_tensor(Ne).to(DEVICE)
    nf_neg = torch.as_tensor(within_class_pool(Yf, Yp, Np, rng)).to(DEVICE); ne_neg = torch.as_tensor(within_class_pool(Ye, Yp, Np, rng)).to(DEVICE)
    pn_list = []
    for _ in range(perms):
        pn = Ne.copy()
        for ii in class_blocks(Ye):
            pn[ii] = pn[ii][rng.permutation(len(ii))]
        pn_list.append(pn)
    PN = torch.as_tensor(np.stack(pn_list)).to(DEVICE)
    ti, vi = make_split(len(zf), split_seed)
    ti, vi = ti.to(DEVICE), vi.to(DEVICE)
    D = Design.build(zf, nf, nf_neg, ti, vi)
    res = {"n": int(n), "n1_frac_eval": float(Ne.mean()), "split_sizes": [int(len(ti)), int(len(vi))], "critics": {}}
    # ---- A1, A3 (P116 algorithms; reproduction gate)
    crit = {"A1": (ridge_tanh_calibrated(D), D, ze)}
    for B, c in js_lbfgs(D, budgets).items():
        crit[f"A3@{B}"] = (c, D, ze)
    # ---- K1
    k1, psi, Dk = k1_fit(zf, nf, nf_neg, ti, vi, split_seed)
    crit["K1"] = (k1, Dk, psi(ze))
    for name, (cr, Dx, zev) in crit.items():
        r = {"meta": cr.meta, **risks(cr, Dx)}
        r["test"] = perm_tests(cr, zev, ne, ne_neg, PN, delta)
        r["Jcommon_eval"] = r["test"]["common"]["stat"]
        res["critics"][name] = r
    # ---- H1, H2 (native statistics; shared permutations)
    blocks = class_blocks(Ye)
    PN_list = [pn for pn in PN]
    t0 = time.perf_counter()
    Ks = class_kernels(ze, blocks)
    h1 = hsic_test(Ks, blocks, ne, PN_list, n, delta); h1["seconds_total"] = time.perf_counter() - t0
    res["critics"]["H1"] = {"meta": {"algo": "H1_hsic_class_median"}, "test": {"own": h1}}
    t0 = time.perf_counter()
    dk = fit_deep_kernel(zf, nf, Yf, deep_steps, split_seed + 2, np.random.default_rng(split_seed + 3))
    with torch.no_grad():
        fe = dk(ze)
    t_fit = time.perf_counter() - t0
    h2 = hsic_test(class_kernels(fe, blocks), blocks, ne, PN_list, n, delta)
    res["critics"]["H2"] = {"meta": {"algo": "H2_hsic_deep_kernel", "steps": deep_steps, "best_step": int(dk.best_step),
                                     "val_score": float(dk.val_score), "seconds_fit": t_fit, "seconds_total": t_fit + h2["seconds"]},
                            "test": {"own": h2}}
    res["seconds_repeat"] = time.perf_counter() - t_rep
    return res


def summarise(inst: list[dict]) -> dict:
    out = {}
    for nm in inst[0]["critics"]:
        rows = [i["critics"][nm] for i in inst]
        own = [r["test"]["own"]["reject"] for r in rows]
        k, R = int(np.sum(own)), len(own)
        z = 1.96; ph = k / R; den = 1 + z * z / R
        lo, hi = (ph + z * z / (2 * R) - z * math.sqrt(ph * (1 - ph) / R + z * z / (4 * R * R))) / den, \
                 (ph + z * z / (2 * R) + z * math.sqrt(ph * (1 - ph) / R + z * z / (4 * R * R))) / den
        s = {"power_own": ph, "power_own_wilson95": [lo, hi],
             "fit_seconds_mean": float(np.mean([r["meta"].get("seconds_total", r["meta"].get("seconds", np.nan)) for r in rows])),
             "perm_seconds_mean": float(np.mean([r["test"]["own"].get("seconds", r["test"].get("seconds", np.nan)) for r in rows]))}
        if "common" in rows[0]["test"]:
            s["power_common"] = float(np.mean([r["test"]["common"]["reject"] for r in rows]))
            s["Jcommon_eval"] = float(np.mean([r["Jcommon_eval"] for r in rows]))
        if nm != "A1":
            a1 = [i["critics"]["A1"]["test"]["own"]["reject"] for i in inst]
            b = sum(1 for x, y in zip(own, a1) if x and not y); c = sum(1 for x, y in zip(own, a1) if y and not x)
            s["vs_A1_discordant"] = {"method_only": b, "A1_only": c}
        out[nm] = s
    return out


def gate(inst: list[dict], ref_inst: list[dict]) -> dict:
    """P116 reproduction: A1 and A3@B own / common statistics and decisions per repeat."""
    worst, mism = 0.0, 0
    for a, b in zip(inst, ref_inst):
        for nm in ("A1", "A3@50", "A3@200", "A3@800"):
            for t in ("own", "common"):
                x, y = a["critics"][nm]["test"][t], b["critics"][nm]["test"][t]
                worst = max(worst, abs(x["stat"] - y["stat"])); mism += int(x["reject"] != y["reject"])
    return {"repeats_compared": min(len(inst), len(ref_inst)), "max_abs_stat_diff": worst, "decision_mismatches": mism,
            "pass": worst <= 1e-6 and mism == 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--family", required=True, choices=("colour", "blur"))
    ap.add_argument("--cells", required=True); ap.add_argument("--repeats", type=int, default=100); ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.05); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="P116 JSON whose A1 / A3 instances must be reproduced"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))); print("device:", DEVICE, flush=True)
    fam = Family(Path(a.features)); assert fam.name == a.family, (fam.name, a.family)
    R, perms, budgets, deep = (2, a.perms, BUDGETS, 60) if a.smoke else (a.repeats, a.perms, BUDGETS, DEEP_STEPS)
    ref = json.load(open(a.check))["cases"] if a.check else None
    cells = []
    for tok in a.cells.split(";"):
        case, n = tok.split(":"); rest = case[len(a.family) + 1:]
        if rest.startswith("s"):
            mode, s = "planted", float(rest[1:])
        elif rest == "null_label_only":
            mode, s = "null_label_only", 0.0
        elif rest.startswith("null_all_planted"):
            mode, s = "null_all_planted", float(rest[len("null_all_planted"):])
        else:
            raise ValueError(tok)
        cells.append((case, mode, s, int(n)))
    rng = np.random.default_rng(a.seed)
    out = {"settings": {**vars(a), "budgets": list(budgets), "rff_D": RFF_D, "rff_bw": list(RFF_BW), "deep_steps": deep},
           "device": str(DEVICE), "utc": utc_now(), "family_dir": str(fam.dir), "cases": {}}
    t0 = time.time()
    for case, mode, s, n in cells:
        inst = [run_repeat(fam, n, rng, mode=mode, strength=s, delta=a.delta, perms=perms, split_seed=a.seed * 1000 + r, budgets=budgets,
                           deep_steps=deep) for r in range(R)]
        cell = {"repeats": R, "summary": summarise(inst), "instances": inst}
        if ref and case in ref and str(n) in ref[case]["by_n"]:
            cell["reproduction_gate"] = gate(inst, ref[case]["by_n"][str(n)]["instances"])
        out["cases"].setdefault(case, {"mode": mode, "strength": s, "by_n": {}})["by_n"][str(n)] = cell
        print(f"[{case} n={n} R={R}] " + " ".join(f"{k}={v['power_own']:.2f}" for k, v in cell["summary"].items())
              + (f" | gate {cell['reproduction_gate']}" if "reproduction_gate" in cell else "")
              + f" | {np.mean([i['seconds_repeat'] for i in inst]):.1f}s/rep ({time.time() - t0:.0f}s)", flush=True)
        atomic_write_json(Path(a.out + ".partial.json"), out)
    atomic_write_json(Path(a.out + ".json"), out)
    print(f"wrote {a.out}.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
