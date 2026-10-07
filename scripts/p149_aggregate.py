"""P149 aggregate (frozen reading, prereg §4): per fixture x estimator
  repeatability  refit spread over the 3 init seeds at identity_t0; drift identity_t0 -> orthogonal_refit_t0 by a paired bootstrap over the shared
                 EVAL pools (1 000 reps, shared P / Q indices, seed-paired);
  resolution     adjacent channel levels t 0 -> 0.25 -> 1: ordering probability P(J(t_k) > J(t_k+1)) over seeds x bootstrap reps (seed-paired,
                 shared indices) and the standardised gap |mean dJ| / pooled sd (sd of J over seeds x reps at each level, pooled);
  labels         numerically resolvable (ordering >= 0.95 and gap >= 2 pooled sd at both steps) / not resolvable; estimation-sensitive when the
                 refit spread or the re-pairing sd exceeds the adjacent gap |dJ|;
  context        null J and permutation p, saturation, S_plugin; approximation bias never covered (real images, no oracle).
J on a bootstrap rep = mean over resampled P rows of (T - T^2/2) + mean over resampled Q rows of (-T - T^2/2) (independent two-pool resampling).
    python scripts/p149_aggregate.py [--out reports/P149_results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

O = Path("/home/infres/yinwang/CS_QMI/outputs/P149_twoview")
RUNS = ("P107_AP3_views4_800ep_seed1", "P41_simclr_views4_800ep_seed1", "P107_AP3_c100_views4_800ep_seed1", "P91_c100_simclr_views4_800ep_seed1")
ESTS = ("vcs_mlp", "js_mlp", "rff_ridge_tanh")
LADDER = ("identity_t0", "identity_t0.25", "identity_t1")
SEEDS = (0, 1, 2)
BOOT, BOOT_SEED = 1000, 149_1000


def boot_J(tp: np.ndarray, tq: np.ndarray, ip: np.ndarray, iq: np.ndarray) -> np.ndarray:
    ap, aq = tp - 0.5 * tp ** 2, -tq - 0.5 * tq ** 2
    return ap[ip].mean(1) + aq[iq].mean(1)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/P149_results.json"); a = ap.parse_args()
    res, missing = {}, [r for r in RUNS if not (O / f"measure_{r}.json").exists()]
    for run in RUNS:
        if run in missing:
            continue
        d = json.load(open(O / f"measure_{run}.json")); z = np.load(O / f"measure_{run}.npz")
        nP, nQ = len(z["eval_P_rows"]), len(z["eval_QL_rows"]); g = np.random.default_rng(BOOT_SEED)
        ip, iq = g.integers(0, nP, (BOOT, nP)), g.integers(0, nQ, (BOOT, nQ))  # shared across settings / estimators / seeds
        T = lambda s, e, k, side: z[f"{s}/{e}/seed{k}/T_{side}"].astype(np.float64)
        out = {"transport_check": {e: v["max_abs_f_diff"] for e, v in d["transport_check"].items()}, "estimators": {}}
        for est in ESTS:
            cell = lambda s, k: d["settings"][s][f"{est}/seed{k}"]
            Jb = {s: np.stack([boot_J(T(s, est, k, "P"), T(s, est, k, "Q"), ip, iq) for k in SEEDS]) for s in (*LADDER, "orthogonal_refit_t0")}  # [seed, rep]
            per = {}
            for s in (*LADDER, "orthogonal_refit_t0"):
                J = np.array([cell(s, k)["J_common"] for k in SEEDS])
                per[s] = {"J_mean": float(J.mean()), "J_seeds": J.tolist(), "refit_spread_sd": float(J.std(ddof=1)), "se_two_pool_mean": float(np.mean([cell(s, k)["se_J"] for k in SEEDS])),
                          "boot_sd": float(Jb[s].std(ddof=1)), "S_plugin_mean": float(np.mean([cell(s, k)["S_plugin"] for k in SEEDS])),
                          "repair_sd": float(np.sqrt(np.mean([cell(s, k)["pair_var_conditional_batch"] for k in SEEDS]))),
                          "null_J_mean": float(np.mean([cell(s, k)["null_J_mean"] for k in SEEDS])), "perm_p_max": float(max(cell(s, k)["perm_p"] for k in SEEDS)),
                          "sat_P_mean": float(np.mean([cell(s, k)["sat_P"] for k in SEEDS])), "sat_Q_mean": float(np.mean([cell(s, k)["sat_Q"] for k in SEEDS])),
                          "hoeffding_radius": float(cell(s, 0)["hoeffding_radius_J"]), "conservative_S_lower_mean": float(np.mean([cell(s, k)["conservative_S_lower"] for k in SEEDS]))}
            dr = (Jb["orthogonal_refit_t0"] - Jb["identity_t0"]).ravel()
            drift = {"mean": float(np.mean([per["orthogonal_refit_t0"]["J_seeds"][k] - per["identity_t0"]["J_seeds"][k] for k in SEEDS])),
                     "ci95_paired_boot": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]}
            drift["ci_excludes_0"] = bool(drift["ci95_paired_boot"][0] > 0 or drift["ci95_paired_boot"][1] < 0)
            steps, ok, sens = {}, True, False
            for lo, hi in zip(LADDER[:-1], LADDER[1:]):
                diff = (Jb[lo] - Jb[hi]).ravel(); dJ = per[lo]["J_mean"] - per[hi]["J_mean"]
                pooled = float(np.sqrt(0.5 * (Jb[lo].var(ddof=1) + Jb[hi].var(ddof=1))))
                st = {"dJ": dJ, "ordering_prob": float((diff > 0).mean()), "pooled_sd": pooled, "std_gap": abs(dJ) / pooled if pooled > 0 else None,
                      "max_refit_spread": max(per[lo]["refit_spread_sd"], per[hi]["refit_spread_sd"]), "max_repair_sd": max(per[lo]["repair_sd"], per[hi]["repair_sd"])}
                st["resolvable"] = bool(st["ordering_prob"] >= 0.95 and (st["std_gap"] or 0) >= 2)
                st["estimation_sensitive"] = bool(max(st["max_refit_spread"], st["max_repair_sd"]) > abs(dJ))
                ok &= st["resolvable"]; sens |= st["estimation_sensitive"]; steps[f"{lo}->{hi}"] = st
            label = "numerically resolvable" if ok else "not resolvable"
            out["estimators"][est] = {"per_setting": per, "drift_identity_to_orthogonal": drift, "ladder": steps, "label": label + (" (estimation-sensitive)" if sens else ""),
                                      "approximation_bias_covered": False}
        res[run] = out
    # cross-estimator paired differences on the same fixture at identity_t0 (common readout, shared indices): VCS - JS, VCS - RFF
    json.dump({"runs": res, "missing": missing, "boot": BOOT, "boot_seed": BOOT_SEED}, open(a.out, "w"), indent=1)
    for run, out in res.items():
        print(f"== {run}  transport {out['transport_check']}")
        for est, e in out["estimators"].items():
            p = e["per_setting"]; dft = e["drift_identity_to_orthogonal"]
            lad = "  ".join(f"{k.split('->')[1].replace('identity_', '')}: dJ {v['dJ']:+.3f} P {v['ordering_prob']:.3f} gap {v['std_gap']:.1f}sd" for k, v in e["ladder"].items())
            print(f"  {est:15s} J0 {p['identity_t0']['J_mean']:.3f} (refit sd {p['identity_t0']['refit_spread_sd']:.4f}, se {p['identity_t0']['se_two_pool_mean']:.4f}, sat {p['identity_t0']['sat_P_mean']:.2f}) "
                  f"orth drift {dft['mean']:+.4f} [{dft['ci95_paired_boot'][0]:+.3f},{dft['ci95_paired_boot'][1]:+.3f}] | {lad} | {e['label']}")
    if missing:
        print("missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
