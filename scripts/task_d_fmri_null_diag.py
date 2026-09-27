"""P65 addendum 1 — diagnosis of the anti-conservative subject-swapped null at n = 240 (CPU only; frozen results untouched).

    python scripts/task_d_fmri_null_diag.py --prep <prep dir> --out <prefix> [--sizes 200,240,300] [--alpha 0] [--shifts 200] [--steps 300]

For every subject i (z of i, N of i+1 mod S, alpha, lag 0) and every n: the observed statistics of the frozen pipeline (same contiguous split,
same critic fit, same K = 4 product shifts) and then several null constructions on the evaluation block:
  eval_restricted  circular shifts of N_eval drawn from [min_shift, L - min_shift]  (the frozen P65 construction)
  eval_full        circular shifts drawn uniformly from {1, ..., L-1}               (Monte-Carlo over the whole cyclic group)
  orbit_exact      every shift 1..L-1 once, p = (1 + #{>= obs}) / L                 (the exact cyclic-group test)
  window           N shifted over the whole n-window by s in [min_shift, n - min_shift], then the evaluation block is taken (seam mostly outside)
  block_boot       moving-block bootstrap of N_eval (block length 12 TRs)
  phase            phase-randomised surrogate of N_eval (amplitude spectrum preserved)
Statistics: qcfc |r(g, N)|, HSIC (Gaussian kernels), VCS J_eval of the picked critic.  Also: the admissible-shift sets per L, and the seam
diagnostic (mean statistic over the orbit vs at shift 0).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_d_fmri_tests import ClosedForm, fit_mlp, gaussian_kernel, hsic_cont, j_of, pearson, rolled, shifts_for  # noqa: E402

DEV = torch.device("cpu")


def block_bootstrap(n, rng, block=12):
    L = len(n); out = [];
    while sum(len(b) for b in out) < L:
        s = int(rng.integers(0, L - block + 1)); out.append(n[s: s + block])
    return torch.cat(out)[:L]


def phase_surrogate(n, rng):
    x = n.numpy().astype(np.float64); F = np.fft.rfft(x); ph = rng.uniform(0, 2 * np.pi, size=F.shape)
    ph[0] = 0.0
    if len(x) % 2 == 0:
        ph[-1] = 0.0
    y = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x)); return torch.tensor(y, dtype=torch.float32)


def admissible(L, min_shift):
    hi = L - min_shift
    return list(range(1, L)) if hi <= min_shift else list(range(min_shift, hi + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True); ap.add_argument("--sizes", default="200,240,300")
    ap.add_argument("--alpha", default="0"); ap.add_argument("--shifts", type=int, default=200); ap.add_argument("--min-shift", type=int, default=30)
    ap.add_argument("--k-neg", type=int, default=4); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--max-subjects", type=int, default=None)
    a = ap.parse_args()
    prep = Path(a.prep); man = json.load(open(prep / "manifest.json"))
    subs = sorted(s for s, q in man["subjects"].items() if "error" not in q and (prep / f"{s}.pt").exists())
    if a.max_subjects:
        subs = subs[: a.max_subjects]
    data = {s: torch.load(prep / f"{s}.pt", map_location="cpu", weights_only=False) for s in subs}
    ak = next(k for k in data[subs[0]]["z"] if float(k) == float(a.alpha))
    sizes = [int(x) for x in a.sizes.split(",")]; rng = np.random.default_rng(a.seed); t0 = time.time()
    modes = ("eval_restricted", "eval_full", "orbit_exact", "window", "block_boot", "phase")
    res = {"settings": vars(a), "subjects": subs, "admissible": {}, "per_n": {}}
    for L in (36, 60, 72, 90, 142):
        adm = admissible(L, a.min_shift); res["admissible"][str(L)] = {"n_distinct": len(adm), "min": min(adm), "max": max(adm), "fallback_full": (L - a.min_shift) <= a.min_shift}
    print("admissible shifts per evaluation-block length:", json.dumps(res["admissible"]), flush=True)
    for n in sizes:
        L_eval = max(30, int(round(0.3 * n))); n_train = n - L_eval; n_fit = int(round(0.8 * n_train))
        rows = {m: {"qcfc": [], "hsic": [], "vcs": []} for m in modes}; seam = {"r_obs": [], "r_orbit_mean": [], "h_obs": [], "h_orbit_mean": [], "J_obs": [], "J_orbit_mean": []}; picked = []
        distinct_used = []
        for i, s in enumerate(subs):
            d, d2 = data[s], data[subs[(i + 1) % len(subs)]]; T = min(len(d["n"]), len(d2["n"]), n)
            z, nn_, g = d["z"][ak][:T], d2["n"][:T], d["g"][ak][:T]
            zf, nf = z[:n_fit], nn_[:n_fit]; zv, nv = z[n_fit:n_train], nn_[n_fit:n_train]; ze, ne, ge = z[n_train:], nn_[n_train:], g[n_train:]
            L = len(ne)
            negf = [rolled(nf, u) for u in shifts_for(len(nf), a.k_neg, rng, a.min_shift)]; negv = [rolled(nv, u) for u in shifts_for(len(nv), a.k_neg, rng, a.min_shift)]
            seval = shifts_for(L, a.k_neg, rng, a.min_shift)
            cf = ClosedForm(zf, nf, negf, zv, nv, negv); mlp, mval = fit_mlp(zf, nf, negf, zv, nv, negv, a.steps, a.seed * 1000 + i)
            crit = cf if cf.val_j >= mval else mlp; picked.append("closed" if crit is cf else "mlp")
            Kz = gaussian_kernel(ze)
            with torch.no_grad():
                def stats(nsur):
                    J = j_of(crit(ze, nsur), torch.cat([crit(ze, rolled(nsur, u)) for u in seval]))
                    return abs(pearson(ge, nsur)), hsic_cont(Kz, nsur), J
                r0, h0, J0 = stats(ne)
                # full orbit (exact test + seam diagnostic)
                orbit = np.array([stats(rolled(ne, u)) for u in range(1, L)])   # [L-1, 3]
                seam["r_obs"].append(r0); seam["h_obs"].append(h0); seam["J_obs"].append(J0)
                seam["r_orbit_mean"].append(float(orbit[:, 0].mean())); seam["h_orbit_mean"].append(float(orbit[:, 1].mean())); seam["J_orbit_mean"].append(float(orbit[:, 2].mean()))
                for m in modes:
                    if m == "orbit_exact":
                        null = orbit; p = lambda col, obs: float((1 + (null[:, col] >= obs).sum()) / (1 + len(null)))
                    else:
                        if m == "eval_restricted":
                            sh = shifts_for(L, a.shifts, rng, a.min_shift); distinct_used.append(len(set(sh.tolist()))); sur = [rolled(ne, u) for u in sh]
                        elif m == "eval_full":
                            sur = [rolled(ne, u) for u in rng.integers(1, L, size=a.shifts)]
                        elif m == "window":
                            sh = rng.integers(a.min_shift, T - a.min_shift + 1, size=a.shifts); sur = [rolled(nn_, u)[n_train:] for u in sh]
                        elif m == "block_boot":
                            sur = [block_bootstrap(ne, rng) for _ in range(a.shifts)]
                        else:
                            sur = [phase_surrogate(ne, rng) for _ in range(a.shifts)]
                        null = np.array([stats(x) for x in sur]); p = lambda col, obs: float((1 + (null[:, col] >= obs).sum()) / (1 + len(null)))
                    rows[m]["qcfc"].append(p(0, r0) <= a.delta); rows[m]["hsic"].append(p(1, h0) <= a.delta); rows[m]["vcs"].append(p(2, J0) <= a.delta)
            if i % 10 == 9:
                print(f"  n={n}: {i + 1}/{len(subs)} subjects ({time.time() - t0:.0f}s)", flush=True)
        summ = {m: {k: float(np.mean(v)) for k, v in rows[m].items()} for m in modes}
        seam_s = {k: float(np.mean(v)) for k, v in seam.items()}
        res["per_n"][str(n)] = {"L_eval": L_eval, "n_train": n_train, "n_fit": n_fit, "admissible_restricted": res["admissible"].get(str(L_eval), {"n_distinct": len(admissible(L_eval, a.min_shift))}),
                                "distinct_shifts_in_200_draws_mean": float(np.mean(distinct_used)) if distinct_used else None, "rejection": summ, "seam": seam_s,
                                "picked_closed_frac": float(np.mean([p == "closed" for p in picked]))}
        print(f"[n={n} L_eval={L_eval}] " + " | ".join(f"{m}: qcfc {summ[m]['qcfc']:.2f} hsic {summ[m]['hsic']:.2f} vcs {summ[m]['vcs']:.2f}" for m in modes), flush=True)
        print(f"   seam: |r| obs {seam_s['r_obs']:.4f} vs orbit mean {seam_s['r_orbit_mean']:.4f}; HSIC obs {seam_s['h_obs']:.5f} vs {seam_s['h_orbit_mean']:.5f}; J obs {seam_s['J_obs']:.4f} vs {seam_s['J_orbit_mean']:.4f}", flush=True)
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    L = [f"# P65 addendum 1 — null-construction diagnosis on the subject-swapped null (alpha = {a.alpha}, lag 0, {len(subs)} subjects, {a.shifts} draws, delta = {a.delta})", "",
         "Admissible circular shifts of the evaluation block under the frozen rule (min_shift 30; fallback to all shifts when L - 30 <= 30):", "",
         "| L_eval | distinct shifts | range | fallback to full group |", "|---|---|---|---|"]
    for Ls, v in res["admissible"].items():
        L.append(f"| {Ls} | {v['n_distinct']} | {v['min']}–{v['max']} | {v['fallback_full']} |")
    L += ["", "Rejection rate over subjects (true independence by construction; nominal 0.05):", "", "| n | L_eval | distinct in 200 draws | null construction | qcfc | hsic | vcs |", "|---|---|---|---|---|---|---|"]
    for n, v in res["per_n"].items():
        for m in modes:
            L.append(f"| {n} | {v['L_eval']} | {v['distinct_shifts_in_200_draws_mean']:.1f} | {m} | {v['rejection'][m]['qcfc']:.2f} | {v['rejection'][m]['hsic']:.2f} | {v['rejection'][m]['vcs']:.2f} |")
    L += ["", "Seam diagnostic (mean over subjects): statistic at shift 0 vs mean over the full orbit of shifts 1..L-1 of the evaluation block", "", "| n | |r| obs | |r| orbit | HSIC obs | HSIC orbit | J obs | J orbit |", "|---|---|---|---|---|---|---|"]
    for n, v in res["per_n"].items():
        s = v["seam"]; L.append(f"| {n} | {s['r_obs']:.4f} | {s['r_orbit_mean']:.4f} | {s['h_obs']:.5f} | {s['h_orbit_mean']:.5f} | {s['J_obs']:.4f} | {s['J_orbit_mean']:.4f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
