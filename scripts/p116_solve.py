"""P116 (v5 NEXT-E-SOLVE) runner — solver decomposition on the T1 conditional-test construction with ONE shared explicit FIT / VAL split.

Per repeat (T1 construction unchanged: fresh N | Y, disjoint FIT / EVAL / POOL of n items, within-class POOL negatives, B shared within-class
permutations of N on EVAL), on the same draws and the same (ti, vi):
  A1        ridge_tanh_calibrated (= the historical "vcs_closed", two-stage: exact ridge of the raw linear J, then VAL-chosen output scale + tanh)
  A2@B      A1 + continued L-BFGS on the original bounded J of tanh(phi v) with the ridge-grid penalty (lam on VAL), VAL early stopping
            (A1 is a candidate), budget B in {50, 200, 800} closures (B / 4 per lam)
  A2L@B     the same runs without early stopping: last checkpoint within B, lam on VAL (decomposition only)
  A3@B      matched JS L-BFGS on the same phi, lam on VAL, B closures split over the 4 lam values
  JS_P105   scripts/cond_test_t1_ablation.py::exact_js_critic (historical settings: max_iter 200 per lam) on the shared split — reference only
Recorded: own-objective FIT / VAL risk, common bounded-J score J(T) on FIT / VAL / EVAL, permutation power of the own statistic and of the common
score, optimisation residuals and termination, seconds (fit, calibration / continuation, permutation, total).  Nothing is selected on EVAL.
RNG roles: numpy `rng` (seed) = data draws + POOL negatives + permutations (as T1); torch Generator(split_seed) = FIT / VAL split; no
initialisation randomness (A1 deterministic, A2 starts at A1, A3 / JS_P105 start at zero).

    python scripts/p116_solve.py --features <P45 dir> --family colour --cells 'colour_s0.05:1000' --repeats 25 --seed 601 --out reports/P116_...
"""
from __future__ import annotations

import argparse
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

from cond_test_t1 import Family, class_blocks, draw_sample  # noqa: E402
from cond_test_t1_ablation import exact_js_critic  # noqa: E402
from precheck_d_tests import DEVICE, within_class_pool  # noqa: E402
from vcs_measure.solve import (BUDGETS, Design, LinearCritic, js_lbfgs, make_split, perm_tests, ridge_tanh_calibrated,  # noqa: E402
                               ridge_then_bounded_j, risks)
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402


def run_repeat(fam: Family, n: int, rng, *, mode: str, strength: float, delta: float, perms: int, split_seed: int, budgets=BUDGETS) -> dict:
    t_rep = time.perf_counter()
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
    D = Design.build(zf, nf, nf_neg, ti.to(DEVICE), vi.to(DEVICE))
    crit: dict[str, LinearCritic] = {}
    a1 = ridge_tanh_calibrated(D); crit["A1"] = a1
    for B, c in ridge_then_bounded_j(D, a1, budgets).items():
        crit[f"A2L@{B[1]}" if isinstance(B, tuple) else f"A2@{B}"] = c
    for B, c in js_lbfgs(D, budgets).items():
        crit[f"A3@{B}"] = c
    t0 = time.perf_counter()
    ref = exact_js_critic(zf, nf, nf_neg, split_seed, split=(ti.to(DEVICE), vi.to(DEVICE)))
    crit["JS_P105"] = LinearCritic(ref.w.detach().clone(), "js", {"algo": "JS_P105_reference", "lam": ref.lam, "val_JS": float(ref.val_J),
                                                                   "seconds_total": time.perf_counter() - t0, "solver": ref.solver,
                                                                   "closures_used": sum(d["func_evals"] for d in ref.solver["per_lam"])})
    res = {"n": int(n), "n1_frac_eval": float(Ne.mean()), "split_sizes": [int(len(ti)), int(len(vi))], "critics": {}}
    for name, cr in crit.items():
        r = {"meta": cr.meta, **risks(cr, D)}
        pt = perm_tests(cr, ze, ne, ne_neg, PN, delta)
        r["test"] = pt
        with torch.no_grad():
            r["Jcommon_eval"] = pt["common"]["stat"]
        res["critics"][name] = r
    res["seconds_repeat"] = time.perf_counter() - t_rep
    return res


def summarise(inst: list[dict]) -> dict:
    names = list(inst[0]["critics"])
    out = {}
    for nm in names:
        rows = [i["critics"][nm] for i in inst]
        m = rows[0]["meta"]
        sec = [r["meta"].get("seconds_total", r["meta"].get("seconds", np.nan)) for r in rows]
        out[nm] = {"power_own": float(np.mean([r["test"]["own"]["reject"] for r in rows])),
                   "power_common": float(np.mean([r["test"]["common"]["reject"] for r in rows])),
                   **{k: float(np.mean([r[k] for r in rows])) for k in ("own_risk_fit", "own_risk_val", "Jcommon_fit", "Jcommon_val", "Jcommon_eval")},
                   "fit_seconds_mean": float(np.nanmean(sec)), "perm_seconds_mean": float(np.mean([r["test"]["seconds"] for r in rows])),
                   "closures_mean": float(np.mean([r["meta"].get("closures_used", 0) for r in rows])), "algo": m.get("algo")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--family", required=True, choices=("colour", "blur"))
    ap.add_argument("--cells", required=True, help="';'-separated case:n (colour_s0.05:1000; colour_null_label_only:2000; colour_null_all_planted0.2:2000)")
    ap.add_argument("--repeats", type=int, default=100); ap.add_argument("--perms", type=int, default=200); ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))); print("device:", DEVICE, flush=True)
    fam = Family(Path(a.features)); assert fam.name == a.family, (fam.name, a.family)
    R, perms, budgets = (2, 20, (20, 50, 100)) if a.smoke else (a.repeats, a.perms, BUDGETS)
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
        cells.append((case, mode, s, 200 if a.smoke else int(n)))
    rng = np.random.default_rng(a.seed)
    out = {"settings": {**vars(a), "budgets": list(budgets)}, "device": str(DEVICE), "utc": utc_now(), "family_dir": str(fam.dir), "cases": {}}
    t0 = time.time()
    for case, mode, s, n in cells:
        inst = []
        for r in range(R):
            inst.append(run_repeat(fam, n, rng, mode=mode, strength=s, delta=a.delta, perms=perms, split_seed=a.seed * 1000 + r, budgets=budgets))
        summ = summarise(inst)
        out["cases"].setdefault(case, {"mode": mode, "strength": s, "by_n": {}})["by_n"][str(n)] = {"repeats": R, "summary": summ, "instances": inst}
        print(f"[{case} n={n} R={R}] " + " ".join(f"{k}={v['power_own']:.2f}/{v['power_common']:.2f}" for k, v in summ.items())
              + f" | {np.mean([i['seconds_repeat'] for i in inst]):.1f}s/rep ({time.time() - t0:.0f}s)", flush=True)
        atomic_write_json(Path(a.out + ".partial.json"), out)
    atomic_write_json(Path(a.out + ".json"), out)
    print(f"wrote {a.out}.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
