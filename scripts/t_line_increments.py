"""P102 — T line: conditional dependence increments between nested information sets on the T1 task (package v2, Spec §11, Plan §8.2).

    python scripts/t_line_increments.py --out reports/P102_t_line_results [--encoders vcs4v800,simclr] [--families colour,blur] [--smoke]

Task (reused from T1 / P73): frozen encoder features h of CIFAR-10 FIT images; nuisance N ∈ {0, 1} class-correlated (P(N = 1 | Y) = ½ ± 0.3 by class
parity), drawn afresh per repeat; N = 1 items carry a planted nuisance of strength s (colour shift or blur, P45 feature files), N = 0 items are clean.
Strength 0 = T1's `null_label_only` case (clean features for everyone): no conditional dependence between H and N given Y.
Conditional experiment: P = P_Y P_{H,N|Y} (observed triples), Q = P_Y P_{H|Y} P_{N|Y} (N replaced by a same-class draw from a disjoint POOL; equal values allowed).

Information sets (all receive N and Y explicitly; A ⊂ B by construction — every coarser set is a deterministic function of the finer one):
    h        standardised 512-d h                                  (B, the full state)
    pca64    first 64 PCA coordinates of h (basis fixed on PROBE)
    pca16    first 16 PCA coordinates (a sub-vector of pca64)
    logit16  10 class logits of a linear probe on pca16 (fixed on PROBE)          chain: h -> pca64 -> pca16 -> logit16
    logit_h  10 class logits of a linear probe on h (fixed on PROBE)              main pair: B = h, A = logit_h
PROBE (standardisation, PCA, both linear probes; clean features, labels used as the task label Y) is a fixed set of images excluded from every repeat.
Critics: VCS and matched JS fitted separately (T1 settings: linear and MLP classes, 300 steps, AdamW 1e-3, wd 1e-2, 80 / 20 split), the class picked by the
selection-set J of T = tanh f for both objectives; everything is then evaluated with the common squared score on EVAL.
Readouts per repeat: J per set, Delta_J / D_T / r_BA for the main pair and every chain step, R_orth of the chain; paired bootstrap by base image on EVAL
(fixed critics); refit variability = spread over repeats.  An exact discrete oracle toy is run first as a self-check (R_orth and r_BA = 0 at the oracle).
No permutation p-values: Delta = 0 is not tested by inheritance from T1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from cond_test_t1 import Family, draw_sample  # noqa: E402
from precheck_d_tests import DEVICE, within_class_pool  # noqa: E402
from vcs_estim import increments as I  # noqa: E402
from vcs_ssl.utils import atomic_write_json, utc_now  # noqa: E402

O = Path("/home/infres/yinwang/CS_QMI/outputs")
FEATURE_DIRS = {("vcs4v800", "colour"): O / "P45_precheck_D_vcs4v800", ("simclr", "colour"): O / "P45_precheck_D_simclr",
                ("vcs4v800", "blur"): O / "P45_precheck_D_vcs4v800_blur", ("simclr", "blur"): O / "P45_precheck_D_simclr_blur"}
STRENGTHS = {"colour": [0.0, 0.05, 0.1, 0.2], "blur": [0.0, 0.25, 0.5, 1.0]}
CHAIN = ["h", "pca64", "pca16", "logit16"]
SETS = CHAIN + ["logit_h"]


# ----------------------------------------------------------------------------------------------------------------------- oracle self-check
def oracle_selfcheck() -> dict:
    p, q = I.toy_joint(n_y=3, n_x=12, seed=0, strength=1.0)
    g1 = np.arange(12) // 2; g2 = np.arange(12) // 4; g3 = np.arange(12) // 12      # 12 -> 6 -> 3 -> 1 states (nested)
    ts = [I.oracle_t(p, q), I.oracle_t(p, q, g1), I.oracle_t(p, q, g2), I.oracle_t(p, q, g3)]
    S = [I.exact_em(p, q, t ** 2) for t in ts]
    d_sq = [I.exact_em(p, q, (ts[l] - ts[l + 1]) ** 2) for l in range(3)]
    inc_ok = max(abs((S[l] - S[l + 1]) - d_sq[l]) for l in range(3))
    rorth = sum(d_sq) - I.exact_em(p, q, (ts[0] - ts[-1]) ** 2)
    jgap = max(abs(I.exact_j(p, q, t) - s) for t, s in zip(ts, S))
    return {"S_chain": S, "increment_identity_max_gap": float(inc_ok), "R_orth_oracle": float(rorth), "J_equals_S_at_oracle_max_gap": float(jgap)}


# ----------------------------------------------------------------------------------------------------------------------- fixed maps on PROBE
def fit_maps(fam: Family, probe_idx: np.ndarray, seed: int, n_classes: int, steps: int = 500) -> dict:
    H = fam.h0[probe_idx]; Y = torch.as_tensor(fam.y[probe_idx])
    mu, sd = H.mean(0), H.std(0) + 1e-6; Z = (H - mu) / sd
    U, S_, V = torch.linalg.svd(Z - Z.mean(0), full_matrices=False); basis = V[:64].T.contiguous()          # [512, 64]
    zc = Z.mean(0)

    def probe(X):
        torch.manual_seed(seed); lin = torch.nn.Linear(X.shape[1], n_classes); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(lin(X), Y); loss.backward(); opt.step()
        lin.eval(); acc = float((lin(X).argmax(1) == Y).float().mean())
        return lin, acc
    p16 = (Z - zc) @ basis[:, :16]
    lin_h, acc_h = probe(Z); lin_16, acc_16 = probe(p16)
    return {"mu": mu, "sd": sd, "zc": zc, "basis": basis, "lin_h": lin_h, "lin_16": lin_16, "probe_acc_h": acc_h, "probe_acc_pca16": acc_16}


@torch.no_grad()
def feature_sets(H: torch.Tensor, maps: dict) -> dict:
    Z = (H - maps["mu"]) / maps["sd"]; P64 = (Z - maps["zc"]) @ maps["basis"]; P16 = P64[:, :16]
    return {"h": Z, "pca64": P64, "pca16": P16, "logit16": maps["lin_16"](P16), "logit_h": maps["lin_h"](Z)}


# ----------------------------------------------------------------------------------------------------------------------- one repeat
def run_repeat(fam: Family, maps: dict, probe_set: set, n: int, rng, *, strength: float, steps: int, seed: int, boot: int, n_classes: int) -> dict:
    mode = "planted" if strength > 0 else "null_label_only"
    s_eff = strength if strength > 0 else min(fam.planted)                # null_label_only ignores the strength
    used = set(probe_set)
    Hf, Yf, Nf, _ = draw_sample(fam, n, rng, mode, s_eff, used)
    He, Ye, Ne, _ = draw_sample(fam, n, rng, mode, s_eff, used)
    Hp, Yp, Np, _ = draw_sample(fam, n, rng, mode, s_eff, used)
    Ff, Fe = feature_sets(Hf, maps), feature_sets(He, maps)
    nf_neg = within_class_pool(Yf, Yp, Np, rng); ne_neg = within_class_pool(Ye, Yp, Np, rng)
    t = lambda a: torch.as_tensor(a).to(DEVICE)
    nf, nfn, yf, ne, nen, ye = t(Nf), t(nf_neg), t(Yf), t(Ne), t(ne_neg), t(Ye)
    res = {"n": n, "strength": strength, "mode": mode, "n1_frac_eval": float(Ne.mean()), "per_objective": {}}
    for obj in ("vcs", "js"):
        tp, tq, picks = {}, {}, {}
        for k, name in enumerate(SETS):
            zf, ze = Ff[name].to(DEVICE), Fe[name].to(DEVICE)
            crit, pick, vals = I.fit_picked(zf, nf, nfn, yf, objective=obj, n_classes=n_classes, steps=steps, seed=seed * 100 + k)
            with torch.no_grad():
                tp[name] = torch.tanh(crit(ze, ne, ye)).double().cpu().numpy(); tq[name] = torch.tanh(crit(ze, nen, ye)).double().cpu().numpy()
            picks[name] = {"picked": pick, "val_J": vals, "best_step": int(crit.best_step)}
        main = I.increment(tp["h"], tq["h"], tp["logit_h"], tq["logit_h"])
        chain = I.chain_summary([tp[c] for c in CHAIN], [tq[c] for c in CHAIN], CHAIN)

        def stat(bp, bq):
            m = I.increment(bp["h"], bq["h"], bp["logit_h"], bq["logit_h"])
            return [m["delta_J"], m["D_T"], m["r_BA"], I.r_orth([bp[c] for c in CHAIN], [bq[c] for c in CHAIN])] + \
                   [I.j_from_t(bp[s], bq[s]) for s in SETS]
        B = I.paired_bootstrap(tp, tq, stat, reps=boot, seed=seed)
        res["per_objective"][obj] = {"main_pair": main, "chain": chain, "critics": picks,
                                     "boot_names": ["delta_J", "D_T", "r_BA", "R_orth"] + [f"J_{s}" for s in SETS],
                                     "boot_ci": {nm: I.ci(B[:, j]) for j, nm in enumerate(["delta_J", "D_T", "r_BA", "R_orth"] + [f"J_{s}" for s in SETS])},
                                     "_boot": B}
    return res


def summarise(reps: list) -> dict:
    out = {}
    for obj in ("vcs", "js"):
        B = np.mean([r["per_objective"][obj]["_boot"] for r in reps], axis=0)          # bootstrap distribution of the repeat mean (fixed critics)
        names = reps[0]["per_objective"][obj]["boot_names"]
        vals = {nm: [r["per_objective"][obj]["main_pair"][nm] for r in reps] for nm in ("delta_J", "D_T", "r_BA")}
        vals["R_orth"] = [r["per_objective"][obj]["chain"]["R_orth"] for r in reps]
        for s in SETS:
            vals[f"J_{s}"] = [r["per_objective"][obj]["chain"]["J"].get(s, r["per_objective"][obj]["main_pair"]["J_A"] if s == "logit_h" else np.nan) for r in reps]
        out[obj] = {nm: {"mean": float(np.mean(v)), "sd_refit": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                         "ci_mean_fixed_critics": I.ci(B[:, names.index(nm)])} for nm, v in vals.items()}
        steps = reps[0]["per_objective"][obj]["chain"]["steps"]
        out[obj]["chain_steps"] = {k: {m: float(np.mean([r["per_objective"][obj]["chain"]["steps"][k][m] for r in reps])) for m in ("delta_J", "D_T", "r_BA")} for k in steps}
    return out


# ----------------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--encoders", default="vcs4v800,simclr"); ap.add_argument("--families", default="colour,blur")
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--probe", type=int, default=8000); ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--strengths", default=None, help="override: comma list applied to every family"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.repeats, a.steps, a.boot, a.probe = 300, 1, 20, 50, 2000
    t0 = time.time()
    results = {"utc": utc_now(), "device": str(DEVICE), "settings": vars(a), "oracle_selfcheck": oracle_selfcheck(), "cells": {}}
    print("oracle self-check:", results["oracle_selfcheck"], flush=True)
    assert results["oracle_selfcheck"]["increment_identity_max_gap"] < 1e-12 and abs(results["oracle_selfcheck"]["R_orth_oracle"]) < 1e-12
    partial = Path(a.out + ".partial.json")
    for enc in a.encoders.split(","):
        for famname in a.families.split(","):
            fam = Family(FEATURE_DIRS[(enc, famname)]); n_classes = int(fam.y.max()) + 1
            prng = np.random.default_rng(a.seed); probe_idx = np.sort(prng.choice(len(fam.y), size=a.probe, replace=False))
            maps = fit_maps(fam, probe_idx, a.seed, n_classes)
            strengths = [float(s) for s in a.strengths.split(",")] if a.strengths else STRENGTHS[famname]
            for s in strengths:
                if s > 0 and s not in fam.planted:
                    print(f"skip {enc}/{famname} s={s}: no planted file", flush=True); continue
                rng = np.random.default_rng(a.seed + int(1000 * s) + (0 if famname == "colour" else 7))
                reps = [run_repeat(fam, maps, set(probe_idx.tolist()), a.n, rng, strength=s, steps=a.steps, seed=a.seed + r, boot=a.boot,
                                   n_classes=n_classes) for r in range(a.repeats)]
                key = f"{enc}/{famname}/s{s:g}"
                summ = summarise(reps)
                for r in reps:
                    for o in r["per_objective"].values():
                        o.pop("_boot", None)
                results["cells"][key] = {"encoder": enc, "family": famname, "strength": s, "probe_acc": {"h": maps["probe_acc_h"], "pca16": maps["probe_acc_pca16"]},
                                         "summary": summ, "repeats": reps}
                v = summ["vcs"]; j = summ["js"]
                print(f"[{key}] VCS dJ {v['delta_J']['mean']:+.4f} {v['delta_J']['ci_mean_fixed_critics']} D_T {v['D_T']['mean']:.4f} r {v['r_BA']['mean']:+.4f} "
                      f"Rorth {v['R_orth']['mean']:+.4f} {v['R_orth']['ci_mean_fixed_critics']} | JS dJ {j['delta_J']['mean']:+.4f} D_T {j['D_T']['mean']:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                atomic_write_json(partial, results)
    atomic_write_json(Path(a.out + ".json"), results)
    Path(a.out + ".md").write_text(markdown(results))
    print(f"wrote {a.out}.json / .md ({time.time() - t0:.0f}s)")
    return 0


def markdown(R: dict) -> str:
    L = [f"# P102 T line — conditional dependence increments ({R['utc']}, {R['device']})", "",
         f"Oracle self-check: {R['oracle_selfcheck']}", "",
         "Main pair B = h vs A = logit_h (both with N, Y).  Delta_J = J_B − J_A; D_T = E_M (T_B − T_A)²; r_BA = Delta_J − D_T.  Chain h → pca64 → pca16 → logit16: R_orth.",
         "Mean over refit repeats (sd over repeats) and the 95 % paired-bootstrap interval of the repeat mean with the critics fixed.", "",
         "| cell | objective | J_h | J_logit_h | Delta_J [CI] | D_T [CI] | r_BA [CI] | R_orth [CI] |", "|---|---|---|---|---|---|---|---|"]
    f = lambda d: f"{d['mean']:+.4f} [{d['ci_mean_fixed_critics'][0]:+.4f}, {d['ci_mean_fixed_critics'][1]:+.4f}]"
    for key, c in R["cells"].items():
        for obj in ("vcs", "js"):
            s = c["summary"][obj]
            L.append(f"| {key} | {obj} | {s['J_h']['mean']:.4f} | {s['J_logit_h']['mean']:.4f} | {f(s['delta_J'])} | {f(s['D_T'])} | {f(s['r_BA'])} | {f(s['R_orth'])} |")
    L += ["", "Chain steps (mean over repeats): Delta_J / D_T / r_BA per step.", ""]
    for key, c in R["cells"].items():
        for obj in ("vcs", "js"):
            st = c["summary"][obj]["chain_steps"]
            L.append(f"- {key} {obj}: " + "; ".join(f"{k}: {v['delta_J']:+.4f} / {v['D_T']:.4f} / {v['r_BA']:+.4f}" for k, v in st.items()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
