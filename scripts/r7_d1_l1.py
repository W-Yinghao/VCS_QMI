"""r7 R7-30 D1-L1 — CPU finite nested-mask check (this round's light check; frozen protocol coordinator_decisions OPEN-9).
Exact finite laws: C ∈ {−1,+1} with the balanced task law M; the full observation B is 10 binary "tokens" (1,024 joint states) with
class-conditional laws p = λ p0 + (1 − λ) r1, q = λ p0 + (1 − λ) r2 (Dirichlet(0.3) components, λ = 0.6: generic dependence, overlapping
classes).  A mask keeps k of the 10 tokens (the rest are marginalised: package reference pushforward); 3 random token orderings per law
give nested kept sets at k ∈ {1, 2, 4, 6, 8}.
Exact checks (package reference eta_s / pushforward / finite_risk_delta):
  (I1) S_B − S_A = E_M (η_B − η_A)² = E_M (C − η_A)² − E_M (C − η_B)²   (L² projection; theory doc §3)
  (I2) if the full teacher is the true posterior (b = η_B), E_M[h − d] = 2 E_M[(C − b)(b − a)] = 0 for ANY compressed teacher a(A)
       -> a nonzero mean residual measures the FULL teacher's calibration error along (b − a), not the compressed teacher's.
Tables over teachers T = tanh(τ · artanh η) (τ_full, τ_masked ∈ {0.5, 1, 2}) and additive bias T = clip(η + β) (β ∈ {−0.2, 0, 0.2}):
  E_M h, E_M d, E_M (h − d), P(h < 0), corr(h, d) by budget; and, for a calibrated teacher, the ±1.96 sd / √n half-width of the mean
  residual at D1-10's CAL size — a reference for reading "the CI covers 0".
    python scripts/r7_d1_l1.py --out /home/infres/yinwang/CS_QMI/outputs/R7_D1/L1 [--n-cal 6168]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import vcs_vl_reference as REF  # noqa: E402

M_TOK, SEED, LAWS, ORDERS = 10, 20261012, 5, 3
NSTATE = 2 ** M_TOK
BITS = (np.arange(NSTATE)[:, None] >> np.arange(M_TOK)) & 1
BUDGETS = (1, 2, 4, 6, 8)
TAUS = (0.5, 1.0, 2.0)
BIASES = (-0.2, 0.0, 0.2)


def random_law(seed, alpha=0.3, lam=0.6):
    g = np.random.default_rng([SEED, seed]); p0, r1, r2 = (g.dirichlet(np.full(NSTATE, alpha)) for _ in range(3))
    p, q = lam * p0 + (1 - lam) * r1, lam * p0 + (1 - lam) * r2
    return p / p.sum(), q / q.sum()


def coarse_eta(p, q, kept):
    """η_A pulled back to the full states for the observation that keeps tokens `kept`."""
    ids = (BITS[:, list(kept)] * (2 ** np.arange(len(kept)))).sum(1) if len(kept) else np.zeros(NSTATE, dtype=int)
    pa, qa = REF.pushforward(p, q, ids); ea, sa = REF.eta_s(pa, qa)
    _, inv = np.unique(ids, return_inverse=True)
    return ea[inv], sa


def temper(eta, tau):
    return np.tanh(tau * np.arctanh(np.clip(eta, -1 + 1e-12, 1 - 1e-12)))


def expectations(p, q, a, b):
    """Exact E_M over (state, C) of h, d, h − d, 1[h<0] and corr(h, d)."""
    w = np.concatenate([0.5 * p, 0.5 * q]); c = np.concatenate([np.ones(NSTATE), -np.ones(NSTATE)])
    A, B = np.concatenate([a, a]), np.concatenate([b, b]); h, d, res = REF.finite_risk_delta(c, A, B)
    mh, md = float(w @ h), float(w @ d); cov = float(w @ ((h - mh) * (d - md))); sh, sd = float(np.sqrt(w @ (h - mh) ** 2)), float(np.sqrt(w @ (d - md) ** 2))
    return {"E_h": mh, "E_d": md, "E_res": float(w @ res), "sd_res": float(np.sqrt(w @ (res - w @ res) ** 2)), "P_h_neg": float(w @ (h < -1e-15)),
            "corr_h_d": cov / (sh * sd) if sh > 0 and sd > 0 else None, "identity_err": float(np.max(np.abs(h - d - res)))}


def run(n_cal=6168):
    cells, ident = [], []
    for law_id in range(LAWS):
        p, q = random_law(law_id); eb, sb = REF.eta_s(p, q)
        for o in range(ORDERS):
            order = np.random.default_rng([SEED, law_id, o]).permutation(M_TOK)
            for k in BUDGETS:
                ea, sa = coarse_eta(p, q, order[:k])
                w = np.concatenate([0.5 * p, 0.5 * q]); c = np.concatenate([np.ones(NSTATE), -np.ones(NSTATE)])
                gap = float(w @ ((c - np.concatenate([ea, ea])) ** 2 - (c - np.concatenate([eb, eb])) ** 2))
                ident.append({"law": law_id, "order": o, "k": k, "S_B": sb, "S_A": sa, "S_gap": sb - sa,
                              "E_M_eta_diff_sq": float(0.5 * np.sum((p + q) * (eb - ea) ** 2)), "min_risk_gap": gap})
                for tb, ta in itertools.product(TAUS, TAUS):
                    e = expectations(p, q, temper(ea, ta), temper(eb, tb))
                    cells.append({"law": law_id, "order": o, "k": k, "teacher": f"tau_full={tb},tau_masked={ta}", **e})
                for beta in BIASES:
                    e = expectations(p, q, np.clip(ea + beta, -1, 1), np.clip(eb + beta, -1, 1))
                    cells.append({"law": law_id, "order": o, "k": k, "teacher": f"bias={beta:+.1f}", **e})
    worst = max(max(abs(r["S_gap"] - r["E_M_eta_diff_sq"]), abs(r["S_gap"] - r["min_risk_gap"])) for r in ident)
    table = {}
    for c in cells:
        key = (c["teacher"], c["k"]); table.setdefault(key, []).append(c)
    agg = []
    for (t, k), v in sorted(table.items()):
        agg.append({"teacher": t, "k_of_10": k, "E_h": float(np.mean([x["E_h"] for x in v])), "E_d": float(np.mean([x["E_d"] for x in v])),
                    "E_res": float(np.mean([x["E_res"] for x in v])), "E_res_range": [float(min(x["E_res"] for x in v)), float(max(x["E_res"] for x in v))],
                    "P_h_neg": float(np.mean([x["P_h_neg"] for x in v])), "corr_h_d": float(np.mean([x["corr_h_d"] for x in v if x["corr_h_d"] is not None])),
                    "half_width_at_n_cal_if_iid": float(np.mean([1.96 * x["sd_res"] / np.sqrt(n_cal) for x in v]))})
    calibrated_full = [c for c in cells if c["teacher"].startswith("tau_full=1.0")]
    return {"design": {"tokens": M_TOK, "states": NSTATE, "laws": LAWS, "orderings_per_law": ORDERS, "budgets_k_of_10": BUDGETS, "taus": TAUS,
                       "biases": BIASES, "n_cal": n_cal},
            "identity_I1_max_abs_error": worst, "identity_I2_max_abs_mean_residual_calibrated_full": float(max(abs(c["E_res"]) for c in calibrated_full)),
            "max_pointwise_identity_error": float(max(c["identity_err"] for c in cells)), "identities": ident, "table": agg}


def markdown(res):
    L = ["# R7-30 D1-L1 — finite nested-mask check (generated by scripts/r7_d1_l1.py)\n",
         f"I1 S_B − S_A = E_M(η_B − η_A)² = min-risk gap: max abs error {res['identity_I1_max_abs_error']:.1e} over "
         f"{len(res['identities'])} (law, ordering, budget) cells.",
         f"I2 calibrated full teacher (τ_full = 1, any τ_masked): max |E_M(h − d)| = {res['identity_I2_max_abs_mean_residual_calibrated_full']:.1e}.\n",
         "| teacher | k of 10 | E h | E d | E (h − d) [range over laws × orderings] | P(h<0) | corr(h,d) | ±1.96 sd/√n at CAL n (iid) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["table"]:
        L.append(f"| {r['teacher']} | {r['k_of_10']} | {r['E_h']:.4f} | {r['E_d']:.4f} | {r['E_res']:+.4f} [{r['E_res_range'][0]:+.4f}, {r['E_res_range'][1]:+.4f}] | "
                 f"{r['P_h_neg']:.2f} | {r['corr_h_d']:.2f} | {r['half_width_at_n_cal_if_iid']:.4f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-cal", type=int, default=6168)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res = run(a.n_cal)
    (out / "d1_l1.json").write_text(json.dumps(res, indent=1)); (out / "d1_l1.md").write_text(markdown(res)); print(markdown(res))
    return 0 if res["identity_I1_max_abs_error"] < 1e-12 and res["identity_I2_max_abs_mean_residual_calibrated_full"] < 1e-10 else 5


if __name__ == "__main__":
    sys.exit(main())
