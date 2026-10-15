"""V8.1 D04a — distribution quality and score information (FULL_PLAN_CN 04 §D04, 05 §3), distribution half of D04.

Formal units (8 arms x seeds 0-2) on DEV and CAL_PROB, from the renormalised exports (q renormalised, u = tails(q), mu kept):
  * package distribution_metrics per image: random-rater RPS = CDF-fit MSE + vote constant (identity residual checked),
    histogram log loss (floor 1e-12), mu-MSE, rating variance, per-threshold Brier; prompt-macro (= V8-03 natural weight for
    image-level quantities: 3- and 6-vote images weigh the same) and image-micro;
  * q / u / mu consistency of the raw exports (max deviations, from the prenorm meta);
  * marginals: mean predicted q / u on DEV vs the FIT prior (R1 prior_fit / fit_hist from the unit log; never recomputed from
    DEV) and the DEV empirical vote marginals (descriptive);
  * fixed-rule strata (descriptive only, nothing removed): vote count (3 / 6), human-mean bin, generator, manifest `type`,
    prompt-length tertile with cut points from FIT prompts, observed vote variance;
  * same-seed arm pairs: do close means hide different histograms?  per-image L1(q_a, q_b), max_k |u_a - u_b|, |mu_a - mu_b|;
  * predicted rating spread vs observed vote dispersion (a prediction of the rating distribution, not an uncertainty of mu).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v81_common as C  # noqa: E402

core = C.load_core()
ROLES = ["DEV", "CAL_PROB"]
DM_KEYS = ["rps_random_rater", "cdf_fit_mse", "vote_constant", "hist_log_loss", "mu_mse", "rating_variance", "brier_Hge2", "brier_Hge3", "brier_Hge4", "brier_Hge5"]


def log(*a):
    print(*a, flush=True)


def labels(role):
    recs = {}
    for r in C.manifest(role):
        if r.get("overall_votes"):
            v = np.asarray(r["overall_votes"], float)
            recs[r["image_id"]] = {"hist": core.histogram(r["overall_votes"]), "mean": float(np.mean(v)), "var": float(np.var(v)), "var_unbiased": float(np.var(v, ddof=1)) if len(v) > 1 else None, "n_votes": len(v),
                                   "prompt_id": str(r["prompt_id"]), "generator_id": r.get("generator_id"), "type": r.get("type"), "prompt_len": len(r.get("prompt") or ""),
                                   "element_count": r.get("element_count"),
                                   "cluster": str(r["prompt_group"])}
    return recs


def unit_pred(unit, role):
    pre = C.NORM / "v8fit" / unit.name / f"prenorm_{role}.jsonl"
    rows = C.read_jsonl(pre)
    meta = C.jload(pre.with_suffix(".meta.json"))
    return {r["image_id"]: r for r in rows}, meta


def per_image(pred, recs):
    ids = sorted(recs)
    if set(pred) != set(recs):
        raise SystemExit("prediction / label universes differ")
    M = {k: np.empty(len(ids)) for k in DM_KEYS}
    for n, i in enumerate(ids):
        d = core.distribution_metrics(pred[i]["q"], recs[i]["hist"])
        for k in DM_KEYS:
            M[k][n] = d[k]
        if abs(d["rps_identity_residual"]) > 1e-10:
            raise SystemExit("RPS identity residual above 1e-10")
    return ids, M


def aggregate(ids, M, recs):
    pid = np.array([recs[i]["prompt_id"] for i in ids])
    out = {}
    for k, v in M.items():
        out[f"{k}_micro"] = float(v.mean())
        out[f"{k}_macro"] = float(np.mean([v[pid == p].mean() for p in np.unique(pid)]))
    out["n_images"] = len(ids)
    out["n_prompts"] = int(len(np.unique(pid)))
    return out


def j_readout(ids, pred, recs, prior):
    """CS-native readout per threshold k on the role (package 05 §1): T_k = tanh((logit u_k - logit pi_k) / 2) with the FIT prior pi_k (unit log);
    P_k propto w y_k, Q_k propto w (1 - y_k), w = natural image weight of the role (uniform prompt -> uniform image);
    J_k = E_P T - E_Q T - 1/2 E_P T^2 - 1/2 E_Q T^2 (= S_k - E_M (T - eta)^2, so J differences between models on the same role are differences in
    posterior-ratio fit).  Returns per-k J and the mean."""
    pid = [recs[i]["prompt_id"] for i in ids]
    n_p = {}
    for p_ in pid:
        n_p[p_] = n_p.get(p_, 0) + 1
    w = np.array([1.0 / n_p[p_] for p_ in pid]); w = w / w.sum()
    U = np.clip(np.array([pred[i]["u"] for i in ids], float), 1e-12, 1 - 1e-12)
    Y = C.tails(np.array([recs[i]["hist"] for i in ids], float))
    pi = np.asarray(prior, float)
    T = np.tanh((np.log(U / (1 - U)) - np.log(pi / (1 - pi))) / 2)
    out = {}
    js = []
    for k in range(4):
        wp, wq = w * Y[:, k], w * (1 - Y[:, k])
        ep, eq = (wp @ T[:, k]) / wp.sum(), (wq @ T[:, k]) / wq.sum()
        ep2, eq2 = (wp @ T[:, k] ** 2) / wp.sum(), (wq @ T[:, k] ** 2) / wq.sum()
        j = float(ep - eq - 0.5 * ep2 - 0.5 * eq2)
        out[f"J_Hge{k+2}"] = j
        js.append(j)
    out["J_mean"] = float(np.mean(js))
    out["role_tail_marginal"] = [float(x) for x in (w[:, None] * Y).sum(0)]
    return out


def j_terms(ids, pred, recs, prior):
    """Per-image pieces of J_k so that cluster-resampled J can be rebuilt as ratios of weighted sums."""
    pid = [recs[i]["prompt_id"] for i in ids]
    n_p = {}
    for p_ in pid:
        n_p[p_] = n_p.get(p_, 0) + 1
    w = np.array([1.0 / n_p[p_] for p_ in pid])
    U = np.clip(np.array([pred[i]["u"] for i in ids], float), 1e-12, 1 - 1e-12)
    Y = C.tails(np.array([recs[i]["hist"] for i in ids], float))
    pi = np.asarray(prior, float)
    T = np.tanh((np.log(U / (1 - U)) - np.log(pi / (1 - pi))) / 2)
    return w, Y, T


def j_from_counts(Mult, w, Y, T):
    """Mult: (reps x n_images) cluster multiplicities -> (reps,) J_mean."""
    js = []
    for k in range(4):
        wp, wq = w * Y[:, k], w * (1 - Y[:, k])
        sp, sq = Mult @ wp, Mult @ wq
        ep, eq = (Mult @ (wp * T[:, k])) / sp, (Mult @ (wq * T[:, k])) / sq
        ep2, eq2 = (Mult @ (wp * T[:, k] ** 2)) / sp, (Mult @ (wq * T[:, k] ** 2)) / sq
        js.append(ep - eq - 0.5 * ep2 - 0.5 * eq2)
    return np.mean(js, axis=0)


def paired_distribution_contrasts(per_unit_dev, recs, priors, reps=C.BOOT_REPS, seed=C.BOOT_SEED):
    """U-CS minus every other arm, same seed, DEV: prompt-macro RPS / CDF-fit / log loss / mu-MSE differences (package cluster_bootstrap on
    per-prompt differences, prompt_group clusters) and J_mean difference (cluster multiplicities, same replicates for both models)."""
    ids = per_unit_dev[("U-CS", 0)][0]
    clusters = [recs[i]["cluster"] for i in ids]
    keys = sorted(set(clusters)); cidx = np.array([keys.index(c) for c in clusters])
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)) for _ in range(reps)]).astype(float)
    Mult = counts[:, cidx]
    pid = np.array([recs[i]["prompt_id"] for i in ids])
    prompts = sorted(set(pid)); pcl = [recs[ids[int(np.flatnonzero(pid == p_)[0])]]["cluster"] for p_ in prompts]
    rows = []
    for s in C.SEEDS:
        ia, Ma, pa = per_unit_dev[("U-CS", s)]
        wa, Ya, Ta = j_terms(ia, pa, recs, priors[("U-CS", s)])
        ja = j_from_counts(Mult, wa, Ya, Ta)
        for b in C.ARMS[1:]:
            ib, Mb, pb = per_unit_dev[(b, s)]
            assert ib == ia
            row = {"a": "U-CS", "b": b, "seed": s}
            for key, name in (("rps_random_rater", "rps"), ("cdf_fit_mse", "cdf_fit"), ("hist_log_loss", "log_loss"), ("mu_mse", "mu_mse")):
                d = Ma[key] - Mb[key]
                per_prompt = [float(d[pid == p_].mean()) for p_ in prompts]
                bt = core.cluster_bootstrap(per_prompt, pcl, reps=reps, seed=seed)
                row.update({f"d_{name}": bt["mean"], f"d_{name}_lo": bt["ci95"][0], f"d_{name}_hi": bt["ci95"][1]})
            wb, Yb, Tb = j_terms(ib, pb, recs, priors[(b, s)])
            jb = j_from_counts(Mult, wb, Yb, Tb)
            ones = np.ones((1, len(ia)))
            lvl = float(j_from_counts(ones, wa, Ya, Ta)[0] - j_from_counts(ones, wb, Yb, Tb)[0])
            dj = ja - jb
            row.update({"d_J_mean": lvl, "d_J_mean_lo": float(np.quantile(dj, .025)), "d_J_mean_hi": float(np.quantile(dj, .975))})
            rows.append(row)
    return rows, {"n_clusters": len(keys), "reps": reps, "seed": seed, "status": "development_interval_no_p_value"}


def strata_defs(recs, fit_cuts):
    def f(i):
        r = recs[i]
        m = r["mean"]
        mb = "[1,2)" if m < 2 else "[2,3)" if m < 3 else "[3,4)" if m < 4 else "[4,5]"
        pl = "short" if r["prompt_len"] <= fit_cuts[0] else "mid" if r["prompt_len"] <= fit_cuts[1] else "long"
        vv = "0" if r["var"] == 0 else "(0,1]" if r["var"] <= 1 else ">1"
        ec = r.get("element_count")
        eb = "NA" if ec is None else "low" if ec <= fit_cuts[2] else "mid" if ec <= fit_cuts[3] else "high"
        return {"votes": str(r["n_votes"]), "human_mean_bin": mb, "generator": r["generator_id"], "type": str(r["type"]), "prompt_len_tertile_fit": pl, "vote_variance": vv,
                "element_count_tertile_fit": eb}
    return f


def main():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    fit_len = np.array(sorted({(r["prompt_id"], len(r.get("prompt") or "")) for r in C.manifest("FIT")}), dtype=object)[:, 1].astype(float)
    fit_cuts = [float(np.percentile(fit_len, 100 / 3)), float(np.percentile(fit_len, 200 / 3))]
    fit_ec = np.array([v for v in {r["prompt_id"]: r.get("element_count") for r in C.manifest("FIT")}.values() if v is not None], float)
    fit_cuts += [float(np.percentile(fit_ec, 100 / 3)), float(np.percentile(fit_ec, 200 / 3))]
    recs = {role: labels(role) for role in ROLES}
    unit_rows, consistency, marginals, strata_rows, spread_rows = [], [], [], [], []
    role_marginals = {}
    per_unit_dev = {}
    for a in C.ARMS:
        for s in C.SEEDS:
            unit = C.fit_unit(a, s)
            L = C.unit_log(unit)
            for role in ROLES:
                pred, meta = unit_pred(unit, role)
                ids, M = per_image(pred, recs[role])
                agg = aggregate(ids, M, recs[role])
                jr = j_readout(ids, pred, recs[role], L["prior_fit"])
                unit_rows.append({"method": a, "seed": s, "role": role, "checkpoint_id": C.ckpt_id(L), **agg, **{k: v for k, v in jr.items() if k != "role_tail_marginal"}})
                role_marginals[role] = jr["role_tail_marginal"]
                consistency.append({"method": a, "seed": s, "role": role, **meta["deviations"], "source_sha256": meta["source_sha256"]})
                if role == "DEV":
                    per_unit_dev[(a, s)] = (ids, M, pred)
                    Q = np.array([pred[i]["q"] for i in ids]); U = np.array([pred[i]["u"] for i in ids])
                    H = np.array([recs[role][i]["hist"] for i in ids]); Y = C.tails(H)
                    marginals.append({"method": a, "seed": s, "ref": "predicted_mean_DEV", **{f"q{h+1}": float(x) for h, x in enumerate(Q.mean(0))}, **{f"u{k+2}": float(x) for k, x in enumerate(U.mean(0))}})
                    if s == 0 and a == "U-CS":
                        marginals.append({"method": "reference", "seed": -1, "ref": "DEV_empirical_votes", **{f"q{h+1}": float(x) for h, x in enumerate(H.mean(0))}, **{f"u{k+2}": float(x) for k, x in enumerate(Y.mean(0))}})
                    marginals.append({"method": a, "seed": s, "ref": "FIT_prior_from_unit_log", **{f"q{h+1}": float(x) for h, x in enumerate(L["fit_hist"])}, **{f"u{k+2}": float(x) for k, x in enumerate(L["prior_fit"])}})
                    sd = strata_defs(recs[role], fit_cuts)
                    groups = defaultdict(list)
                    for n, i in enumerate(ids):
                        for kind, val in sd(i).items():
                            groups[(kind, val)].append(n)
                    for (kind, val), idx in sorted(groups.items()):
                        idx = np.array(idx)
                        strata_rows.append({"method": a, "seed": s, "stratum_kind": kind, "stratum": val, "n_images": len(idx), "rps": float(M["rps_random_rater"][idx].mean()),
                                            "mu_mse": float(M["mu_mse"][idx].mean()), "hist_log_loss": float(M["hist_log_loss"][idx].mean()), "cdf_fit_mse": float(M["cdf_fit_mse"][idx].mean()),
                                            "human_mean": float(np.mean([recs[role][ids[n]]["mean"] for n in idx]))})
                    pv = M["rating_variance"]; ov = np.array([recs[role][i]["var"] for i in ids]); ovu = np.array([recs[role][i]["var_unbiased"] for i in ids], float)
                    nv = np.array([recs[role][i]["n_votes"] for i in ids])
                    spread_rows.append({"method": a, "seed": s, "mean_predicted_rating_variance": float(pv.mean()), "mean_observed_vote_variance": float(ov.mean()),
                                        "mean_observed_vote_variance_unbiased": float(ovu.mean()), "mean_pred_var_3votes": float(pv[nv == 3].mean()), "mean_obs_var_unbiased_3votes": float(ovu[nv == 3].mean()),
                                        "mean_pred_var_6votes": float(pv[nv == 6].mean()) if (nv == 6).any() else None, "mean_obs_var_unbiased_6votes": float(ovu[nv == 6].mean()) if (nv == 6).any() else None,
                                        "spearman_pred_var_vs_obs_var": C.pooled_spearman(pv, ov), "share_obs_var_zero": float((ov == 0).mean()),
                                        "mean_pred_var_given_obs_var_zero": float(pv[ov == 0].mean()), "mean_pred_var_given_obs_var_gt1": float(pv[ov > 1].mean()) if (ov > 1).any() else None})
            log(f"[d04a] {unit.name} ok")
    C.csv_write(C.RESULTS / "d04a_unit_roles.csv", unit_rows)
    C.csv_write(C.RESULTS / "distribution.csv", unit_rows)   # package task-table name for the D04 table (same content)
    C.csv_write(C.RESULTS / "d04a_export_consistency.csv", consistency)
    # ---- cross-check against the frozen V8-04 evaluator's JSON (same quantities, raw exports there; expected agreement ~1e-6)
    E = C.jload(C.V8_EVAL_JSON)
    xc = []
    for r in unit_rows:
        ref = E["distribution"][f"V8-FIT-{r['method']}-s{r['seed']}"][r["role"]]["raw"]
        xc.append({"method": r["method"], "seed": r["seed"], "role": r["role"], "v804_mean_mse_prompt_macro": ref["mean_mse_prompt_macro"], "v804_rps_natural": ref["rps_natural"],
                   "v804_hist_logloss_natural": ref.get("hist_logloss_natural"), "d_mu_mse_macro": r["mu_mse_macro"] - ref["mean_mse_prompt_macro"],
                   "d_rps_macro": r["rps_random_rater_macro"] - ref["rps_natural"],
                   "d_hist_log_loss_macro": (r["hist_log_loss_macro"] - ref["hist_logloss_natural"]) if ref.get("hist_logloss_natural") is not None else None,
                   "d_brier_max": max(abs(r[f"brier_Hge{k+2}_macro"] - ref["brier_k_natural"][k]) for k in range(4))})
    C.csv_write(C.RESULTS / "d04a_crosscheck_v804.csv", xc)
    crosscheck = {"source": str(C.V8_EVAL_JSON), "sha256": C.sha256(C.V8_EVAL_JSON),
                  "max_abs": {k: max(abs(x[k]) for x in xc if x[k] is not None) for k in ("d_mu_mse_macro", "d_rps_macro", "d_hist_log_loss_macro", "d_brier_max")}}
    C.csv_write(C.RESULTS / "d04a_marginals.csv", marginals)
    C.csv_write(C.RESULTS / "d04a_strata.csv", strata_rows)
    C.csv_write(C.RESULTS / "d04a_spread.csv", spread_rows)
    # ---- arm means
    def arm_mean(rows, keys, extra=lambda r: True):
        out = []
        for a in C.ARMS:
            for role in ROLES:
                rs = [r for r in rows if r["method"] == a and r.get("role", "DEV") == role and extra(r)]
                if rs:
                    out.append({"method": a, "role": role, "n_seeds": len(rs), **{k: float(np.mean([r[k] for r in rs if r[k] is not None])) for k in keys}})
        return out
    am = arm_mean(unit_rows, [f"{k}_{w}" for k in DM_KEYS for w in ("macro", "micro")] + ["J_Hge2", "J_Hge3", "J_Hge4", "J_Hge5", "J_mean"])
    C.csv_write(C.RESULTS / "d04a_arm_mean.csv", am)
    # ---- strata by arm (mean over seeds)
    st = defaultdict(list)
    for r in strata_rows:
        st[(r["method"], r["stratum_kind"], r["stratum"])].append(r)
    st_rows = [{"method": a, "stratum_kind": k, "stratum": v, "n_images": rs[0]["n_images"], **{m: float(np.mean([r[m] for r in rs])) for m in ("rps", "mu_mse", "hist_log_loss", "cdf_fit_mse", "human_mean")}}
               for (a, k, v), rs in sorted(st.items(), key=lambda kv: (C.ARMS.index(kv[0][0]), kv[0][1], kv[0][2]))]
    C.csv_write(C.RESULTS / "d04a_strata_arm_mean.csv", st_rows)
    # ---- same-seed pairs: histogram differences at close means; cross-seed same-arm as the baseline
    pair_rows = []
    def qdiff(ka, kb):
        ids_a, _, pa = per_unit_dev[ka]; ids_b, _, pb = per_unit_dev[kb]
        assert ids_a == ids_b
        Qa = np.array([pa[i]["q"] for i in ids_a]); Qb = np.array([pb[i]["q"] for i in ids_a])
        Ua, Ub = C.tails(Qa), C.tails(Qb)
        mua, mub = np.array([pa[i]["mu"] for i in ids_a]), np.array([pb[i]["mu"] for i in ids_a])
        l1 = np.abs(Qa - Qb).sum(1); du = np.abs(Ua - Ub).max(1); dmu = np.abs(mua - mub)
        close = dmu < 0.1
        return {"n_images": len(l1), "mean_L1_q": float(l1.mean()), "mean_max_abs_du": float(du.mean()), "mean_abs_dmu": float(dmu.mean()), "share_close_mu_lt_0.1": float(close.mean()),
                "mean_L1_q_given_close_mu": float(l1[close].mean()) if close.any() else None, "share_close_mu_and_L1_gt_0.2": float((close & (l1 > 0.2)).mean()),
                "spearman_abs_dmu_vs_L1": C.pooled_spearman(dmu, l1), "max_L1_q": float(l1.max())}
    for s in C.SEEDS:
        for i, a in enumerate(C.ARMS):
            for b in C.ARMS[i + 1:]:
                pair_rows.append({"kind": "same_seed", "a": a, "b": b, "seed": s, **qdiff((a, s), (b, s))})
    for a in C.ARMS:
        for s1, s2 in ((0, 1), (0, 2), (1, 2)):
            pair_rows.append({"kind": "same_arm_cross_seed", "a": a, "b": a, "seed": f"{s1}-{s2}", **qdiff((a, s1), (a, s2))})
    C.csv_write(C.RESULTS / "d04a_pair_qdiff.csv", pair_rows)
    priors = {(a, s): C.unit_log(C.fit_unit(a, s))["prior_fit"] for a in C.ARMS for s in C.SEEDS}
    contrasts, boot_meta = paired_distribution_contrasts(per_unit_dev, recs["DEV"], priors)
    C.csv_write(C.RESULTS / "d04a_paired_contrasts_dev.csv", contrasts)
    cm = defaultdict(list)
    for r in contrasts:
        cm[r["b"]].append(r)
    contrast_mean = [{"a": "U-CS", "b": b, "n_seeds": len(rs), **{k: float(np.mean([r[k] for r in rs])) for k in ("d_rps", "d_cdf_fit", "d_log_loss", "d_mu_mse", "d_J_mean")},
                      "n_seeds_rps_interval_excludes_0": sum((r["d_rps_lo"] > 0) or (r["d_rps_hi"] < 0) for r in rs),
                      "n_seeds_J_interval_excludes_0": sum((r["d_J_mean_lo"] > 0) or (r["d_J_mean_hi"] < 0) for r in rs),
                      "n_seeds_logloss_interval_excludes_0": sum((r["d_log_loss_lo"] > 0) or (r["d_log_loss_hi"] < 0) for r in rs)} for b, rs in cm.items()]
    C.csv_write(C.RESULTS / "d04a_paired_contrasts_dev_mean.csv", contrast_mean)
    pq = defaultdict(list)
    for r in pair_rows:
        pq[(r["kind"], r["a"], r["b"])].append(r)
    pq_mean = [{"kind": k, "a": a, "b": b, **{m: (float(np.mean([r[m] for r in rs if r[m] is not None])) if any(r[m] is not None for r in rs) else None) for m in ("mean_L1_q", "mean_max_abs_du", "mean_abs_dmu", "share_close_mu_lt_0.1", "mean_L1_q_given_close_mu", "share_close_mu_and_L1_gt_0.2", "spearman_abs_dmu_vs_L1")}}
               for (k, a, b), rs in pq.items()]
    C.csv_write(C.RESULTS / "d04a_pair_qdiff_mean.csv", pq_mean)
    # ---- summary
    ucs = [r for r in am if r["method"] == "U-CS" and r["role"] == "DEV"][0]
    summary = {"generated_by": "scripts/v81/d04a_distribution.py", "definitions": "package reference/core.py distribution_metrics (eps 1e-12); prompt-macro = natural weight (V8-03)",
               "prior_source": "unit log prior_fit / fit_hist (R1, FIT only); never recomputed from DEV / CAL labels",
               "fit_prompt_length_cuts_chars": fit_cuts[:2], "fit_element_count_cuts": fit_cuts[2:], "arm_mean": am,
               "role_tail_marginals_natural": role_marginals,
               "j_definition": "J_k = E_P T - E_Q T - 1/2 E_P T^2 - 1/2 E_Q T^2, T_k = tanh((logit u_k - logit pi_k)/2), pi = FIT prior (unit log), P_k/Q_k = role natural weights x y_k / (1 - y_k)",
               "export_consistency_max": {k: max(r[k] for r in consistency) for k in ("max_abs_sum_q_minus_1", "max_abs_u_minus_tails_q", "max_abs_score_minus_qn_dot_levels", "max_abs_score_minus_1_plus_sum_u")},
               "all_u_monotone": all(r["u_monotone"] for r in consistency),
               "crosscheck_vs_v8_04_eval_json": crosscheck,
               "ucs_dev_arm_mean": ucs,
               "pair_qdiff_mean": pq_mean, "spread": spread_rows, "marginals": marginals,
               "paired_contrasts_dev": {"bootstrap": boot_meta, "by_seed": contrasts, "mean_over_seeds": contrast_mean}}
    C.jdump(summary, C.RESULTS / "d04a_summary.json")
    # ---- markdown
    def md_table(rows, fields, fmt="{:.4f}"):
        L = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
        for r in rows:
            L.append("| " + " | ".join("NA" if r.get(f) is None else (fmt.format(r[f]) if isinstance(r[f], float) else str(r[f])) for f in fields) + " |")
        return "\n".join(L) + "\n"
    md = ["# D04a tables (generated by scripts/v81/d04a_distribution.py; every number from results/d04a_*.csv / d04a_summary.json)", ""]
    md.append("## formal units, mean over seeds: random-rater RPS = CDF-fit MSE + vote constant; histogram log loss; mu-MSE; rating variance; per-threshold Brier (prompt-macro)\n")
    md.append(md_table(am, ["method", "role", "rps_random_rater_macro", "cdf_fit_mse_macro", "vote_constant_macro", "hist_log_loss_macro", "mu_mse_macro", "rating_variance_macro", "brier_Hge2_macro", "brier_Hge3_macro", "brier_Hge4_macro", "brier_Hge5_macro", "rps_random_rater_micro", "mu_mse_micro"]))
    md.append("## CS-native readout J_k (FIT priors in T; role natural weights in P / Q), mean over seeds; higher = closer posterior-ratio fit\n")
    md.append(md_table(am, ["method", "role", "J_Hge2", "J_Hge3", "J_Hge4", "J_Hge5", "J_mean"]))
    md.append(f"role tail marginals (natural weights) vs FIT prior: {role_marginals}\n")
    md.append("## paired contrasts on DEV, U-CS minus each arm at the same seed (prompt_group cluster bootstrap 2000 x 2718; lower RPS / log loss / mu-MSE and higher J are better for U-CS when d_rps < 0, d_J > 0)\n")
    md.append(md_table(contrasts, ["b", "seed", "d_rps", "d_rps_lo", "d_rps_hi", "d_log_loss", "d_log_loss_lo", "d_log_loss_hi", "d_mu_mse", "d_mu_mse_lo", "d_mu_mse_hi", "d_J_mean", "d_J_mean_lo", "d_J_mean_hi"]))
    md.append(md_table(contrast_mean, ["b", "n_seeds", "d_rps", "d_cdf_fit", "d_log_loss", "d_mu_mse", "d_J_mean", "n_seeds_rps_interval_excludes_0", "n_seeds_logloss_interval_excludes_0", "n_seeds_J_interval_excludes_0"]))
    md.append("## per unit (DEV)\n")
    md.append(md_table([r for r in unit_rows if r["role"] == "DEV"], ["method", "seed", "rps_random_rater_macro", "cdf_fit_mse_macro", "hist_log_loss_macro", "mu_mse_macro", "rating_variance_macro", "J_mean"]))
    md.append("## export consistency (raw exports; q renormalised for the package simplex check)\n")
    md.append("```json\n" + __import__("json").dumps(summary["export_consistency_max"], indent=1) + f"\nall_u_monotone: {summary['all_u_monotone']}\n```\n")
    md.append("## cross-check vs the frozen V8-04 evaluator JSON (max |difference| over the 48 unit x role rows)\n")
    md.append("```json\n" + __import__("json").dumps(crosscheck["max_abs"], indent=1) + "\n```\n")
    md.append("## marginals on DEV: predicted mean q / u vs FIT prior (unit log) vs DEV empirical votes\n")
    md.append(md_table(marginals, ["method", "seed", "ref", "q1", "q2", "q3", "q4", "q5", "u2", "u3", "u4", "u5"]))
    md.append("## fixed-rule strata on DEV (image-uniform inside a stratum; arm mean over seeds)\n")
    md.append(md_table(st_rows, ["method", "stratum_kind", "stratum", "n_images", "human_mean", "rps", "cdf_fit_mse", "mu_mse", "hist_log_loss"]))
    md.append("## same-seed arm pairs: histogram differences at close means (mean over seeds); cross-seed same-arm rows are the baseline\n")
    md.append(md_table(pq_mean, ["kind", "a", "b", "mean_L1_q", "mean_max_abs_du", "mean_abs_dmu", "share_close_mu_lt_0.1", "mean_L1_q_given_close_mu", "share_close_mu_and_L1_gt_0.2", "spearman_abs_dmu_vs_L1"]))
    md.append("## predicted rating spread vs observed vote dispersion (DEV; a prediction of the rating distribution, not uncertainty of mu)\n")
    md.append(md_table(spread_rows, ["method", "seed", "mean_predicted_rating_variance", "mean_observed_vote_variance", "mean_observed_vote_variance_unbiased", "mean_pred_var_3votes", "mean_obs_var_unbiased_3votes", "mean_pred_var_6votes", "mean_obs_var_unbiased_6votes", "spearman_pred_var_vs_obs_var", "share_obs_var_zero", "mean_pred_var_given_obs_var_zero", "mean_pred_var_given_obs_var_gt1"]))
    (C.RESULTS / "d04a_tables.md").write_text("\n".join(md), encoding="utf-8")
    log("d04a done; U-CS DEV arm mean:", {k: ucs[k] for k in ("rps_random_rater_macro", "hist_log_loss_macro", "mu_mse_macro")})


if __name__ == "__main__":
    main()
