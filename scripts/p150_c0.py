"""P150 (r3 R3-C0) — one crossed diagnostic of detection (P143) vs estimation (P144) on the same data: CIFAR-10 final VCS seed 1, colour 0.1,
h -> O (frozen clean-h head logits) and the label-only null, both P143 and P144 function classes fitted with both objectives and read with ONE
common readout.

Per draw (P143 construction, seed 601 => the first draws coincide with P143 stage 0): FIT / EVAL / POOL samples of n base images from the
regenerated P143 feature store (scripts/p109_i1_features.py --fp32-maps), fresh N | Y with p(N=1|Y) = 0.5 +- 0.3, planted features iff N = 1.
Per site u in {h, O}:  AUD-VCS = ridge-tanh two-stage critic on phi = [u(2N-1), 1] (the P143 'vcs_closed');  AUD-JS = the L-BFGS balanced-logistic
critic on the same phi (P143 'js_exact', logit u -> T = tanh(u/2));  NEST-VCS / NEST-JS = the P144 Y-aware class (linear / MLP picked on VAL,
NS.fit_picked, exact Q with p1, 300 steps).  Common readouts on the SAME EVAL draw: exact-Q J_common (T on the observed (u, N) rows vs the enumerated
(u, 0) / (u, 1) rows weighted (1 - p1, p1)), S_plugin, per-image contributions (saved) -> paired-by-image bootstrap (1 000 reps, shared indices
across classes / objectives), saturation; detection: B = 200 shared within-class permutations of N on EVAL -> p-value of each critic's own
statistic (as P143); fraction of NEST fits selecting step 0.  Sensitivity: NEST refit with 3 optimiser seeds on the same draw (refit spread).
    python scripts/p150_c0.py --features outputs/P150_features --run P107_AP3_views4_800ep_seed1 --out reports/P150/c0 [--repeats 20] [--smoke]
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

from cond_test_t1_ablation import exact_js_critic  # noqa: E402
from precheck_d_tests import closed_form_critic, within_class_pool  # noqa: E402
from vcs_measure import nested as NS  # noqa: E402
from vcs_measure.audit import AuditData, draw, p_n1  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

DEV = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
SITES = {"h": "h", "O": "logits"}
CELLS = {"colour_s0.1": ("planted", "colour_s0.1"), "null": ("null_label_only", "colour_s0.05")}  # null version as P143 (label-only: clean rows)
BOOT, PERMS, N, NEST_STEPS, NEST_SEEDS = 1000, 200, 2000, 300, (0, 1, 2)


def exact_q_readout(T_obs: np.ndarray, T0: np.ndarray, T1: np.ndarray, p1: np.ndarray) -> dict:
    """Per-image contributions: a+(T(u_i, N_i)) and the enumerated a-(T(u_i, n)) weighted by P(n | y_i).  J = mean of their sum (balanced means)."""
    ap = T_obs - 0.5 * T_obs ** 2
    am = (1 - p1) * (-T0 - 0.5 * T0 ** 2) + p1 * (-T1 - 0.5 * T1 ** 2)
    s2 = 0.5 * ((T_obs ** 2).mean() + ((1 - p1) * T0 ** 2 + p1 * T1 ** 2).mean())
    return {"J_common": float(ap.mean() + am.mean()), "S_plugin": float(s2), "sat_obs": float((np.abs(T_obs) > 0.95).mean()), "contrib": ap + am}


def boot_ci(contrib: np.ndarray, idx: np.ndarray) -> list[float]:
    bs = contrib[idx].mean(1); return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=20); ap.add_argument("--seed", type=int, default=601); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    D = AuditData(Path(a.features) / a.run / "features.pt")
    R, n, perms, steps, boot = (2, 200, 20, 30, 50) if a.smoke else (a.repeats, N, PERMS, NEST_STEPS, BOOT)
    rng = np.random.default_rng(a.seed)
    out = {"run": a.run, "utc": utc_now(), "device": str(DEV), "settings": {**vars(a), "n": n, "perms": perms, "nest_steps": steps, "boot": boot, "nest_seeds": NEST_SEEDS}, "cells": {}}
    arrays = {}
    t_all = time.time()
    for cname, (mode, version) in CELLS.items():
        reps = []
        for r in range(R):
            Xf, Yf, Nf = draw(D, "fit", n, rng, mode, version, tuple(SITES.values())); Xe, Ye, Ne = draw(D, "eval", n, rng, mode, version, tuple(SITES.values()))
            _, Yp, Np = draw(D, "val", n, rng, mode, version, ())
            nf, ne = torch.as_tensor(Nf).to(DEV), torch.as_tensor(Ne).to(DEV)
            nf_neg = torch.as_tensor(within_class_pool(Yf, Yp, Np, rng)).to(DEV); ne_neg = torch.as_tensor(within_class_pool(Ye, Yp, Np, rng)).to(DEV)
            perm_N = []
            for _ in range(perms):
                pn = Ne.copy()
                for c in np.unique(Ye):
                    ii = np.where(Ye == c)[0]; pn[ii] = pn[ii][rng.permutation(len(ii))]
                perm_N.append(torch.as_tensor(pn).to(DEV))
            p1e = p_n1(Ye); p1f = torch.as_tensor(p_n1(Yf), dtype=torch.float32).to(DEV)
            bidx = np.random.default_rng(a.seed * 100 + r).integers(0, n, size=(boot, n))  # shared bootstrap indices for this draw
            rec = {"draw": r, "n1_frac_eval": float(Ne.mean()), "sites": {}}
            for site, layer in SITES.items():
                mu, sd = Xf[layer].mean(0), Xf[layer].std(0) + 1e-6
                zf, ze = ((Xf[layer] - mu) / sd).to(DEV), ((Xe[layer] - mu) / sd).to(DEV)
                yf, ye = torch.as_tensor(Yf).to(DEV), torch.as_tensor(Ye).to(DEV)
                zero, one = torch.zeros_like(ne), torch.ones_like(ne)
                crits = {}
                t0 = time.perf_counter(); cv = closed_form_critic(zf, nf, nf_neg, a.seed * 1000 + r); crits["AUD-VCS"] = (lambda z, nn_, y, c=cv: torch.tanh(c(z, nn_)), time.perf_counter() - t0, None)
                t0 = time.perf_counter(); cj = exact_js_critic(zf, nf, nf_neg, a.seed * 1000 + r + 3); crits["AUD-JS"] = (lambda z, nn_, y, c=cj: torch.tanh(0.5 * c(z, nn_)), time.perf_counter() - t0, None)
                nest_models = {}
                for obj in ("vcs", "js"):
                    fits = []
                    for s in NEST_SEEDS:
                        t0 = time.perf_counter()
                        m, pick, vals, bs = NS.fit_picked(zf, nf, yf, {"p1": p1f}, objective=obj, n_classes=int(len(np.unique(Yf))), steps=steps, seed=a.seed * 100 + r * 10 + s)
                        fits.append({"model": m, "picked": pick, "val_J": vals, "best_step": bs, "seconds": time.perf_counter() - t0})
                    nest_models[obj] = fits
                    crits[f"NEST-{obj.upper()}"] = (lambda z, nn_, y, m=fits[0]["model"]: torch.tanh(m(z, nn_, y)), fits[0]["seconds"], fits)
                srec = {}
                for name, (Tfn, secs, fits) in crits.items():
                    with torch.no_grad():
                        To = Tfn(ze, ne, ye).double().cpu().numpy(); T0 = Tfn(ze, zero, ye).double().cpu().numpy(); T1 = Tfn(ze, one, ye).double().cpu().numpy()
                        Tneg = Tfn(ze, ne_neg, ye).double().cpu().numpy()
                        ro = exact_q_readout(To, T0, T1, p1e)
                        # own statistic + shared permutations (as P143): VCS classes use J with sampled within-class negatives; JS classes their JS value
                        def own(Tobs_t, Tneg_t, kind):
                            if kind == "vcs":
                                return float((Tobs_t - 0.5 * Tobs_t ** 2).mean() + (-Tneg_t - 0.5 * Tneg_t ** 2).mean())
                            fo, fn_ = np.arctanh(np.clip(Tobs_t, -0.999999, 0.999999)) * 2, np.arctanh(np.clip(Tneg_t, -0.999999, 0.999999)) * 2
                            return float((-np.logaddexp(0, -fo)).mean() - np.logaddexp(0, fn_).mean() + np.log(4.0))
                        kind = "vcs" if "VCS" in name else "js"
                        obs = own(To, Tneg, kind); null = np.array([own(Tfn(ze, pn, ye).double().cpu().numpy(), Tneg, kind) for pn in perm_N])
                        p = float((1 + (null >= obs).sum()) / (1 + len(null)))
                        sampled_J = float((To - 0.5 * To ** 2).mean() + (-Tneg - 0.5 * Tneg ** 2).mean())
                    srec[name] = {"J_common_exactQ": ro["J_common"], "J_common_sampledQ": sampled_J, "S_plugin": ro["S_plugin"], "sat_obs": ro["sat_obs"],
                                  "boot_ci95_exactQ": boot_ci(ro["contrib"], bidx), "own_stat": obs, "perm_p": p, "reject": p <= 0.05, "fit_seconds": secs}
                    if fits is not None:
                        srec[name]["nest"] = {"picked": [f["picked"] for f in fits], "best_step": [f["best_step"] for f in fits], "step0_selected": [f["best_step"] == 0 for f in fits]}
                        with torch.no_grad():
                            js = [exact_q_readout(torch.tanh(f["model"](ze, ne, ye)).double().cpu().numpy(), torch.tanh(f["model"](ze, zero, ye)).double().cpu().numpy(),
                                                  torch.tanh(f["model"](ze, one, ye)).double().cpu().numpy(), p1e)["J_common"] for f in fits]
                        srec[name]["refit_J_exactQ"] = js; srec[name]["refit_spread_fixed_pool"] = float(np.std(js, ddof=1))
                    arrays[f"{cname}/draw{r}/{site}/{name}/contrib"] = ro["contrib"].astype(np.float32)
                # paired differences across classes / objectives on the shared bootstrap indices
                names = list(srec); pair = {}
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        ca, cb = arrays[f"{cname}/draw{r}/{site}/{names[i]}/contrib"], arrays[f"{cname}/draw{r}/{site}/{names[j]}/contrib"]
                        pair[f"{names[i]}-{names[j]}"] = {"mean": float((ca - cb).mean()), "ci95": boot_ci(ca - cb, bidx)}
                rec["sites"][site] = {"critics": srec, "paired_diff_exactQ": pair}
            reps.append(rec)
            summ = {site: {nm: {"J_exactQ_mean": float(np.mean([x["sites"][site]["critics"][nm]["J_common_exactQ"] for x in reps])),
                                "J_exactQ_sd_over_draws": float(np.std([x["sites"][site]["critics"][nm]["J_common_exactQ"] for x in reps], ddof=1)) if len(reps) > 1 else None,
                                "rejection_rate": float(np.mean([x["sites"][site]["critics"][nm]["reject"] for x in reps])),
                                "S_plugin_mean": float(np.mean([x["sites"][site]["critics"][nm]["S_plugin"] for x in reps]))} for nm in reps[0]["sites"][site]["critics"]} for site in SITES}
            out["cells"][cname] = {"mode": mode, "version": version, "repeats": reps, "summary": summ}
            print(f"[{cname} draw {r}] " + " | ".join(f"{site} " + " ".join(f"{nm}:J{v['J_common_exactQ']:+.4f}/p{v['perm_p']:.2f}" for nm, v in rec["sites"][site]["critics"].items()) for site in SITES)
                  + f" ({time.time() - t_all:.0f}s)", flush=True)
            atomic_write_json(Path(a.out + ".partial.json"), out)
    out["seconds"] = time.time() - t_all
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out + ".npz", **arrays); atomic_write_json(Path(a.out + ".json"), out)
    print(f"wrote {a.out}.json / .npz ({out['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
