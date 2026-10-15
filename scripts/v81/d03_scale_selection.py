"""V8.1 D03 — error decomposition and task utility (FULL_PLAN_CN 04 §D03, 05 §2, §4, §5).

For every normalised DEV model (24 formal V8 units, V7 units, external systems, kernel bridges) and the formal units on
CAL_SELECT / CAL_PROB and the tuning units on DEV:
  * prompt-macro MSE = bias_part + within_part per pool (package mse_decomposition, uniform weights), then the exact
    macro identities  mean_p b_p^2 = bbar^2 + var_p(b_p)  (global offset vs per-prompt offsets vs within-prompt error);
    image-micro and natural-weight (V8-03: uniform prompt -> uniform image = prompt-macro for image-level quantities) alongside;
  * within-prompt pair accuracy (macro over defined pools and pooled over informative pairs), top-1 regret, selected quality;
  * global pooled Spearman with a prompt_group cluster bootstrap that rebuilds the paired rows and re-ranks every replicate
    (2000 x seed 2718; paired differences vs U-CS of the same seed); development intervals, no p-values;
  * selection audit from the unit logs: CAL_SELECT epoch chosen by mean-MSE vs the epoch a pairwise-accuracy rule would choose
    (log level only: per-epoch checkpoints were not saved, so DEV at other epochs is unavailable);
  * secondary: affine recalibration fitted on CAL_PROB (a * mu + b) applied to DEV — scale repair without changing any order;
  * tuning units (seed 17, lr grid) as a secondary table.
Non-rating-scale systems (score_kind native_score) get no 1-5 MSE (NA), ranking metrics only.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v81_common as C  # noqa: E402

core = C.load_core()


def log(*a):
    print(*a, flush=True)


def pool_metrics(rows):
    """One model's labelled rows -> (per-pool list, macro dict, decomposition dict, image arrays)."""
    pools = C.group_pools(rows)
    rating = all(r["score_kind"] == "rating_mean" for r in rows)
    per, errs, num, den = [], [], 0.0, 0
    num_pkg, den_pkg, pkg_macro = 0.0, 0, []
    ids, S, Y, CL = [], [], [], []
    for pid, rs in sorted(pools.items()):
        s = np.array([r["score"] for r in rs], float)
        y = np.array([C.human_mean_of(r, core) for r in rs], float)          # exact sum(votes) / n
        y_pkg = np.array([float(r["human_mean"]) for r in rs], float)        # package reference value (hist @ levels)
        d = {"pool_id": pid, "cluster_id": rs[0].get("cluster_id", rs[0]["prompt_id"]), "n_candidates": len(rs)}
        d.update(core.pairwise_accuracy(s, y))
        pk = core.pairwise_accuracy(s, y_pkg)
        d["pair_accuracy_pkg"] = pk["pair_accuracy"]
        d["informative_pairs_pkg"] = pk["informative_pairs"]
        if pk["pair_accuracy"] is not None:
            num_pkg += pk["pair_accuracy"] * pk["informative_pairs"]
            den_pkg += pk["informative_pairs"]
            pkg_macro.append(pk["pair_accuracy"])
        d.update(core.selection_metrics(s, y))
        d["spearman_to_human"] = core.spearman(s, y)
        d["score_variance"] = float(np.var(s))
        d["human_variance"] = float(np.var(y))
        if rating:
            d.update(core.mse_decomposition(s, y))
            errs.append(s - y)
        per.append(d)
        if d["pair_accuracy"] is not None:
            num += d["pair_accuracy"] * d["informative_pairs"]
            den += d["informative_pairs"]
        ids += [r["image_id"] for r in rs]
        S += list(s)
        Y += list(y)
        CL += [d["cluster_id"]] * len(rs)
    macro = {}
    for k in ("pair_accuracy", "regret", "selected_human_mean", "random_expected_human_mean", "observed_pool_max", "spearman_to_human", "mse", "bias_squared", "centered_mse", "score_variance", "human_variance"):
        vals = [p[k] for p in per if p.get(k) is not None]
        macro[f"{k}_macro"] = float(np.mean(vals)) if vals else None
        macro[f"n_def_{k}"] = len(vals)
    macro["pair_accuracy_pooled"] = num / den if den else None
    macro["informative_pairs_total"] = int(den)
    macro["pair_accuracy_pooled_pkg_hist_mean"] = num_pkg / den_pkg if den_pkg else None
    macro["pair_accuracy_macro_pkg_hist_mean"] = float(np.mean(pkg_macro)) if pkg_macro else None
    macro["informative_pairs_total_pkg_hist_mean"] = int(den_pkg)
    macro["n_pools"] = len(per)
    macro["n_images"] = len(ids)
    macro["n_zero_information_pools"] = sum(p["pair_accuracy"] is None for p in per)
    macro["n_single_candidate_pools"] = sum(p["n_candidates"] == 1 for p in per)
    macro["pooled_spearman"] = C.pooled_spearman(S, Y)
    dec = {}
    if rating:
        dec = C.macro_decomposition(errs)
        dec.update(C.micro_decomposition(np.concatenate(errs)))
    return per, macro, dec, {"image_id": ids, "score": np.array(S), "human": np.array(Y), "cluster": CL}


def level_row(key, macro, dec):
    dataset, role, method, seed, ck = key
    return {"dataset": dataset, "role": role, "method": method, "seed": seed, "checkpoint_id": ck, **macro, **dec}


def by_arm_mean(rows, keys):
    out = []
    for a in C.ARMS:
        rs = [r for r in rows if r["method"] == a and r["seed"] in C.SEEDS]
        if not rs:
            continue
        row = {"method": a, "n_seeds": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
            if len(vals) > 1:
                row[f"{k}_sd"] = float(np.std(vals, ddof=1))
        out.append(row)
    return out


def md_table(rows, fields, fmt="{:.4f}"):
    L = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        L.append("| " + " | ".join("NA" if r.get(f) is None else (fmt.format(r[f]) if isinstance(r[f], float) else str(r[f])) for f in fields) + " |")
    return "\n".join(L) + "\n"


def selection_epochs():
    rows = []
    for a in C.ARMS:
        for s in C.SEEDS:
            L = C.unit_log(C.fit_unit(a, s))
            ep = [e["epoch"] for e in L["epochs_log"]]
            mse = np.array([e["cal_select"]["mean_mse_prompt_macro"] for e in L["epochs_log"]])
            pw = np.array([e["cal_select"]["within_prompt_pairwise_acc_macro"] for e in L["epochs_log"]])
            rps = np.array([e["cal_select"]["rps_micro"] for e in L["epochs_log"]])
            i_mse, i_pw = int(np.argmin(mse)), int(np.argmax(pw))
            rows.append({"method": a, "seed": s, "lr": f"{L['lr']:g}", "selected_epoch_log": L["selected_epoch"], "epoch_argmin_cal_mse": ep[i_mse], "epoch_argmax_cal_pairwise": ep[i_pw],
                         "epoch_argmin_cal_rps": ep[int(np.argmin(rps))], "criteria_agree": ep[i_mse] == ep[i_pw],
                         "cal_mse_at_selected": float(mse[i_mse]), "cal_pairwise_at_selected": float(pw[i_mse]), "cal_pairwise_best": float(pw[i_pw]),
                         "cal_pairwise_gap_best_minus_selected": float(pw[i_pw] - pw[i_mse]), "cal_mse_at_pairwise_best": float(mse[i_pw]),
                         "cal_mse_gap_pairwise_best_minus_selected": float(mse[i_pw] - mse[i_mse]), "cal_mse_last_epoch": float(mse[-1]), "cal_pairwise_last_epoch": float(pw[-1]),
                         "cal_mse_by_epoch": [float(x) for x in mse], "cal_pairwise_by_epoch": [float(x) for x in pw], "n_epochs_logged": len(ep)})
    return rows


def recalibration(models_cal_prob, models_dev):
    rows = []
    for a in C.ARMS:
        for s in C.SEEDS:
            kc = next(k for k in models_cal_prob if k[2] == a and k[3] == s)
            kd = next(k for k in models_dev if k[2] == a and k[3] == s)
            cp = models_cal_prob[kc]
            x = np.array([r["score"] for r in cp]); y = np.array([C.human_mean_of(r, core) for r in cp])
            A = np.stack([x, np.ones_like(x)], 1)
            (aa, bb), *_ = np.linalg.lstsq(A, y, rcond=None)
            dev = models_dev[kd]
            pools = C.group_pools(dev)
            errs_raw, errs_cal = [], []
            for pid, rs in sorted(pools.items()):
                sv = np.array([r["score"] for r in rs]); yv = np.array([C.human_mean_of(r, core) for r in rs])
                errs_raw.append(sv - yv)
                errs_cal.append(aa * sv + bb - yv)
            dr, dc = C.macro_decomposition(errs_raw), C.macro_decomposition(errs_cal)
            rows.append({"method": a, "seed": s, "affine_a": float(aa), "affine_b": float(bb), "order_preserved": bool(aa > 0), "n_cal_prob": len(x),
                         "dev_mse_macro_raw": dr["mse_macro"], "dev_mse_macro_recal": dc["mse_macro"], "dev_bias_sq_macro_raw": dr["bias_sq_macro"], "dev_bias_sq_macro_recal": dc["bias_sq_macro"],
                         "dev_within_macro_raw": dr["within_macro"], "dev_within_macro_recal": dc["within_macro"], "dev_grand_bias_raw": dr["grand_bias"], "dev_grand_bias_recal": dc["grand_bias"],
                         "dev_between_pool_bias_var_raw": dr["between_pool_bias_var"], "dev_between_pool_bias_var_recal": dc["between_pool_bias_var"]})
    return rows


def main():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    prov = {}
    files = {"dev_all": C.NORM / "combined/dev_all.jsonl", "cal_select": C.NORM / "combined/cal_select_v8fit.jsonl",
             "cal_prob": C.NORM / "combined/cal_prob_v8fit.jsonl", "dev_tune": C.NORM / "combined/dev_v8tune.jsonl"}
    data = {}
    for k, p in files.items():
        data[k] = C.split_models(C.read_jsonl(p))
        prov[k] = {"path": str(p), "sha256": C.sha256(p), "n_models": len(data[k])}
    # ---- model levels on every role
    levels, arrays, per_pool_fit = [], {}, {}
    for name, models in data.items():
        for key, rows in sorted(models.items(), key=str):
            per, macro, dec, arr = pool_metrics(rows)
            levels.append({"file": name, **level_row(key, macro, dec)})
            if name == "dev_all":
                arrays[key] = arr
                if key[2] in C.ARMS:
                    per_pool_fit[key] = per
            log(f"[{name}] {C.model_name(dict(zip(('dataset','role','method','seed','checkpoint_id'), key)))}: mse_macro={macro.get('mse_macro')} pair_pooled={macro['pair_accuracy_pooled']:.4f} pooled_rho={macro['pooled_spearman']}")
    # ---- pooled Spearman with the joint cluster bootstrap on DEV (all DEV models share images)
    keys = [k for k in arrays]
    ref_ids = arrays[keys[0]]["image_id"]
    for k in keys:
        if arrays[k]["image_id"] != ref_ids:
            raise SystemExit("DEV image order differs between models")
    y = arrays[keys[0]]["human"]; clusters = arrays[keys[0]]["cluster"]
    S = np.stack([arrays[k]["score"] for k in keys])
    B = C.pooled_spearman_cluster_bootstrap(S, y, clusters)
    col = {k: i for i, k in enumerate(keys)}
    ucs = {s: col[next(k for k in keys if k[2] == "U-CS" and k[3] == s)] for s in C.SEEDS}
    ucs_mean_boot = B["boot"][:, [ucs[s] for s in C.SEEDS]].mean(1)
    ucs_mean_level = float(np.mean([B["level"][ucs[s]] for s in C.SEEDS]))
    sp_rows = []
    for k in keys:
        i = col[k]
        row = {"method": k[2], "seed": k[3], "checkpoint_id": k[4], "pooled_spearman": float(B["level"][i]), "ci_lo": float(B["ci95"][i][0]), "ci_hi": float(B["ci95"][i][1])}
        if k[3] in C.SEEDS and k[2] != "U-CS":
            d = B["boot"][:, i] - B["boot"][:, ucs[k[3]]]
            row.update({"diff_vs_ucs": float(B["level"][i] - B["level"][ucs[k[3]]]), "diff_ref": f"U-CS s{k[3]}", "diff_ci_lo": float(np.nanquantile(d, .025)), "diff_ci_hi": float(np.nanquantile(d, .975))})
        elif k[3] == -1:
            d = B["boot"][:, i] - ucs_mean_boot
            row.update({"diff_vs_ucs": float(B["level"][i] - ucs_mean_level), "diff_ref": "U-CS mean of seeds 0-2", "diff_ci_lo": float(np.nanquantile(d, .025)), "diff_ci_hi": float(np.nanquantile(d, .975))})
        sp_rows.append(row)
    sp_rows.sort(key=lambda r: (r["method"] not in C.ARMS, C.ARMS.index(r["method"]) if r["method"] in C.ARMS else 0, r["method"], r["seed"]))
    C.csv_write(C.RESULTS / "d03_pooled_spearman.csv", sp_rows)
    sp_by_arm = by_arm_mean([{"method": r["method"], "seed": r["seed"], "pooled_spearman": r["pooled_spearman"]} for r in sp_rows], ["pooled_spearman"])
    # ---- levels CSV
    levels.sort(key=lambda r: (r["file"], r["role"], r["method"] not in C.ARMS, C.ARMS.index(r["method"]) if r["method"] in C.ARMS else 0, r["method"], r["seed"]))
    C.csv_write(C.RESULTS / "d03_model_levels.csv", levels)
    dev_fit = [r for r in levels if r["file"] == "dev_all" and r["method"] in C.ARMS]
    keys_mean = ["mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "grand_bias_sq", "between_pool_bias_var", "mse_micro", "bias_sq_micro", "within_micro",
                 "pair_accuracy_macro", "pair_accuracy_pooled", "pair_accuracy_pooled_pkg_hist_mean", "informative_pairs_total", "informative_pairs_total_pkg_hist_mean",
                 "regret_macro", "selected_human_mean_macro", "random_expected_human_mean_macro", "observed_pool_max_macro",
                 "spearman_to_human_macro", "pooled_spearman", "score_variance_macro", "human_variance_macro"]
    arm_mean = by_arm_mean(dev_fit, keys_mean)
    C.csv_write(C.RESULTS / "d03_dev_arm_mean.csv", arm_mean)
    cal_rows = [r for r in levels if r["file"] in ("cal_select", "cal_prob")]
    C.csv_write(C.RESULTS / "d03_cal_roles.csv", cal_rows)
    cal_mean = []
    for role in ("CAL_SELECT", "CAL_PROB"):
        for r in by_arm_mean([r for r in cal_rows if r["role"] == role], ["mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "between_pool_bias_var", "pair_accuracy_pooled", "pair_accuracy_macro", "regret_macro", "pooled_spearman"]):
            cal_mean.append({"role": role, **r})
    C.csv_write(C.RESULTS / "d03_cal_arm_mean.csv", cal_mean)
    # ---- cross-check against the package diagnose per_pool for the formal units (same definitions -> identical)
    D = C.jload(C.DIAG / "dev_all.json")
    diag_pool = {(r["method"], r["seed"], r["pool_id"]): r for r in D["per_pool"] if r["role"] == "DEV" and r["method"] in C.ARMS}
    maxdiff = 0.0
    for key, per in per_pool_fit.items():
        for p in per:
            d = diag_pool[(key[2], key[3], p["pool_id"])]
            for k in ("mse", "bias_squared", "centered_mse", "regret", "selected_human_mean"):
                maxdiff = max(maxdiff, abs(d[k] - p[k]))
            if p["pair_accuracy_pkg"] is not None:
                maxdiff = max(maxdiff, abs(d["pair_accuracy"] - p["pair_accuracy_pkg"]))   # package per_pool uses the hist-based mean
    # ---- cross-check against the frozen V8-04 evaluator JSON (mean-rating MSE prompt-macro on DEV / CAL_PROB, raw exports)
    E = C.jload(C.V8_EVAL_JSON)
    xc804 = []
    for r in levels:
        if r["method"] in C.ARMS and r["seed"] in C.SEEDS and r["role"] in ("DEV", "CAL_PROB"):
            ref = E["distribution"][f"V8-FIT-{r['method']}-s{r['seed']}"][r["role"]]["raw"]
            xc804.append({"method": r["method"], "seed": r["seed"], "role": r["role"], "mse_macro": r["mse_macro"], "v804_mean_mse_prompt_macro": ref["mean_mse_prompt_macro"],
                          "diff": r["mse_macro"] - ref["mean_mse_prompt_macro"], "v804_mean_mse_natural": ref.get("mean_mse_natural"),
                          "diff_vs_natural": (r["mse_macro"] - ref["mean_mse_natural"]) if ref.get("mean_mse_natural") is not None else None})
    C.csv_write(C.RESULTS / "d03_crosscheck_v804.csv", xc804)
    crosscheck804 = {"source": str(C.V8_EVAL_JSON), "sha256": C.sha256(C.V8_EVAL_JSON), "n_rows": len(xc804), "max_abs_diff_macro": max(abs(x["diff"]) for x in xc804),
                     "max_abs_diff_vs_natural": max(abs(x["diff_vs_natural"]) for x in xc804 if x["diff_vs_natural"] is not None) if any(x["diff_vs_natural"] is not None for x in xc804) else None}
    # ---- selection audit, recalibration, tuning grid
    sel = selection_epochs()
    C.csv_write(C.RESULTS / "d03_selection_epochs.csv", sel)
    dev_by = {(r["method"], r["seed"]): r for r in dev_fit}
    sp_by = {(r["method"], r["seed"]): r for r in sp_rows}
    xs = [C.unit_log(C.fit_unit(a, s))["selected_cal_select_mse"] for a in C.ARMS for s in C.SEEDS]
    rel = {}
    for name, getter in (("dev_mse_macro", lambda a, s: dev_by[(a, s)]["mse_macro"]), ("dev_pooled_spearman", lambda a, s: sp_by[(a, s)]["pooled_spearman"]),
                         ("dev_pair_accuracy_pooled", lambda a, s: dev_by[(a, s)]["pair_accuracy_pooled"]), ("dev_regret_macro", lambda a, s: dev_by[(a, s)]["regret_macro"])):
        ys = [getter(a, s) for a in C.ARMS for s in C.SEEDS]
        rel[name] = {"spearman_over_24_units_vs_selected_cal_select_mse": C.pooled_spearman(xs, ys),
                     "pearson_over_24_units": float(np.corrcoef(xs, ys)[0, 1]),
                     "scale6_only_spearman": C.pooled_spearman([x for x, a in zip(xs, [a for a in C.ARMS for _ in C.SEEDS]) if a in C.SCALE_ARMS],
                                                               [v for v, a in zip(ys, [a for a in C.ARMS for _ in C.SEEDS]) if a in C.SCALE_ARMS])}
    recal = recalibration(data["cal_prob"], data["dev_all"])
    C.csv_write(C.RESULTS / "d03_recalibration.csv", recal)
    tune = [r for r in levels if r["file"] == "dev_tune"]
    sel_lr = {a: C.unit_log(C.fit_unit(a, 0))["lr"] for a in C.ARMS}
    for r in tune:
        arm, lr = r["method"].rsplit("-lr", 1)
        r["arm"], r["lr"] = arm, lr
        r["is_selected_lr"] = abs(float(lr) - float(sel_lr[arm])) < 1e-12
    tune.sort(key=lambda r: (C.ARMS.index(r["arm"]), float(r["lr"])))
    C.csv_write(C.RESULTS / "d03_tune_lr.csv", tune, ["arm", "lr", "is_selected_lr", "mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "between_pool_bias_var", "pair_accuracy_pooled", "pair_accuracy_macro", "regret_macro", "pooled_spearman"])
    # ---- export vs training-time record: the exported CAL_SELECT predictions must reproduce the log at the selected epoch
    cal_by = {(r["method"], r["seed"]): r for r in levels if r["file"] == "cal_select"}
    export_check = []
    for a in C.ARMS:
        for s in C.SEEDS:
            L = C.unit_log(C.fit_unit(a, s))
            e = next(x for x in L["epochs_log"] if x["epoch"] == L["selected_epoch"])["cal_select"]
            r = cal_by[(a, s)]
            export_check.append({"method": a, "seed": s, "selected_epoch": L["selected_epoch"], "log_cal_mse_at_selected": e["mean_mse_prompt_macro"],
                                 "export_cal_mse_macro": r["mse_macro"], "diff_mse": r["mse_macro"] - e["mean_mse_prompt_macro"],
                                 "log_cal_pairwise_macro_at_selected": e["within_prompt_pairwise_acc_macro"], "export_cal_pairwise_macro": r["pair_accuracy_macro"],
                                 "diff_pairwise": r["pair_accuracy_macro"] - e["within_prompt_pairwise_acc_macro"], "log_selected_cal_select_mse": L["selected_cal_select_mse"]})
    C.csv_write(C.RESULTS / "d03_export_vs_log.csv", export_check)
    lr_check = []
    for a in C.ARMS:
        rs = [r for r in tune if r["arm"] == a]
        selr = next(r for r in rs if r["is_selected_lr"])
        bm = min(rs, key=lambda r: r["mse_macro"]); br = max(rs, key=lambda r: r["pooled_spearman"])
        lr_check.append({"arm": a, "selected_lr": selr["lr"], "dev_best_lr_by_mse": bm["lr"], "dev_best_lr_by_pooled_spearman": br["lr"], "dev_mse_selected": selr["mse_macro"],
                         "dev_mse_best": bm["mse_macro"], "dev_rho_selected": selr["pooled_spearman"], "dev_rho_best": br["pooled_spearman"]})
    C.csv_write(C.RESULTS / "d03_tune_lr_check.csv", lr_check)
    # ---- epoch-index-0 pattern (0-based index: model after the FIRST completed epoch, package 03 §2): e0-selected units vs the same arm's other seeds
    e0_rows = []
    for a in C.ARMS:
        units = [(s_, C.unit_log(C.fit_unit(a, s_))["selected_epoch"]) for s_ in C.SEEDS]
        e0 = [s_ for s_, e in units if e == 0]
        rest = [s_ for s_, e in units if e != 0]
        if not e0 or not rest:
            continue
        for s_ in e0:
            e0_rows.append({"method": a, "e0_seed": s_, "other_seeds": "+".join(str(x) for x in rest), "other_epochs": "+".join(str(e) for x, e in units if x in rest),
                            "dev_pooled_spearman_e0": sp_by[(a, s_)]["pooled_spearman"], "dev_pooled_spearman_others_mean": float(np.mean([sp_by[(a, x)]["pooled_spearman"] for x in rest])),
                            "diff_pooled_spearman": sp_by[(a, s_)]["pooled_spearman"] - float(np.mean([sp_by[(a, x)]["pooled_spearman"] for x in rest])),
                            "dev_mse_e0": dev_by[(a, s_)]["mse_macro"], "dev_mse_others_mean": float(np.mean([dev_by[(a, x)]["mse_macro"] for x in rest])),
                            "diff_mse": dev_by[(a, s_)]["mse_macro"] - float(np.mean([dev_by[(a, x)]["mse_macro"] for x in rest])),
                            "dev_pair_pooled_e0": dev_by[(a, s_)]["pair_accuracy_pooled"], "dev_pair_pooled_others_mean": float(np.mean([dev_by[(a, x)]["pair_accuracy_pooled"] for x in rest])),
                            "diff_pair_pooled": dev_by[(a, s_)]["pair_accuracy_pooled"] - float(np.mean([dev_by[(a, x)]["pair_accuracy_pooled"] for x in rest]))})
    C.csv_write(C.RESULTS / "d03_epoch0_pattern.csv", e0_rows)
    epoch_hist = {}
    for r in sel:
        epoch_hist[str(r["selected_epoch_log"])] = epoch_hist.get(str(r["selected_epoch_log"]), 0) + 1
    seed_epochs = {f"s{s_}": [r["selected_epoch_log"] for r in sel if r["seed"] == s_] for s_ in C.SEEDS}
    # ---- summary
    ucs_rows = [r for r in dev_fit if r["method"] == "U-CS"]
    summary = {"generated_by": "scripts/v81/d03_scale_selection.py", "provenance": prov, "cluster": "prompt_group", "bootstrap": {"reps": B["reps"], "seed": B["seed"], "n_clusters": B["n_clusters"], "status": B["status"]},
               "crosscheck_vs_package_diagnose_per_pool_max_abs_diff": maxdiff,
               "human_mean_rule": "exact sum(votes)/n for every labelled quantity; the package reference recomputes hist @ levels and splits 1-ulp ties (columns *_pkg_hist_mean keep its value)",
               "crosscheck_vs_v8_04_eval_json": crosscheck804,
               "natural_weight_note": "V8-03 natural sampling = uniform prompt -> uniform valid image; for image-level quantities this equals the prompt-macro column (no separate number)",
               "arm_mean_dev": arm_mean, "pooled_spearman_by_arm": sp_by_arm,
               "ucs_dev_per_seed": [{k: r[k] for k in ("seed", "mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "between_pool_bias_var", "pair_accuracy_pooled", "regret_macro", "pooled_spearman")} for r in ucs_rows],
               "selection_audit": {"n_units": len(sel), "n_units_selected_epoch_equals_argmin_cal_mse": sum(r["selected_epoch_log"] == r["epoch_argmin_cal_mse"] for r in sel),
                                   "n_units_mse_and_pairwise_criteria_agree": sum(r["criteria_agree"] for r in sel),
                                   "mean_cal_pairwise_gap_best_minus_selected": float(np.mean([r["cal_pairwise_gap_best_minus_selected"] for r in sel])),
                                   "max_cal_pairwise_gap_best_minus_selected": float(np.max([r["cal_pairwise_gap_best_minus_selected"] for r in sel])),
                                   "n_units_selected_is_last_epoch": sum(r["selected_epoch_log"] == max(C.unit_log(C.fit_unit(r["method"], r["seed"]))["epochs_log"][-1]["epoch"], r["selected_epoch_log"]) for r in sel),
                                   "selected_epochs": {f"{r['method']}-s{r['seed']}": r["selected_epoch_log"] for r in sel}},
               "selection_vs_dev_over_units": rel,
               "selected_epoch_histogram_0based": epoch_hist, "selected_epochs_by_seed_in_arm_order": seed_epochs,
               "epoch0_pattern": {"n_e0_units_with_non_e0_sibling": len(e0_rows), "n_lower_dev_pooled_spearman": sum(r["diff_pooled_spearman"] < 0 for r in e0_rows),
                                  "n_higher_dev_mse": sum(r["diff_mse"] > 0 for r in e0_rows), "n_lower_dev_pair_pooled": sum(r["diff_pair_pooled"] < 0 for r in e0_rows), "rows": e0_rows},
               "export_vs_log": {"max_abs_diff_mse": max(abs(x["diff_mse"]) for x in export_check), "max_abs_diff_pairwise_macro": max(abs(x["diff_pairwise"]) for x in export_check),
                                 "max_abs_diff_log_selected_vs_epoch_entry": max(abs(x["log_selected_cal_select_mse"] - x["log_cal_mse_at_selected"]) for x in export_check)},
               "tune_lr_check": {"n_arms": len(lr_check), "n_dev_best_mse_lr_equals_selected": sum(x["dev_best_lr_by_mse"] == x["selected_lr"] for x in lr_check),
                                 "n_dev_best_rho_lr_equals_selected": sum(x["dev_best_lr_by_pooled_spearman"] == x["selected_lr"] for x in lr_check), "rows": lr_check},
               "recalibration": {"mean_affine_a": float(np.mean([r["affine_a"] for r in recal])), "mean_affine_b": float(np.mean([r["affine_b"] for r in recal])),
                                 "all_order_preserved": all(r["order_preserved"] for r in recal),
                                 "mean_dev_mse_macro_raw": float(np.mean([r["dev_mse_macro_raw"] for r in recal])), "mean_dev_mse_macro_recal": float(np.mean([r["dev_mse_macro_recal"] for r in recal])),
                                 "by_arm": by_arm_mean(recal, ["affine_a", "affine_b", "dev_mse_macro_raw", "dev_mse_macro_recal", "dev_bias_sq_macro_raw", "dev_bias_sq_macro_recal", "dev_within_macro_raw", "dev_within_macro_recal", "dev_grand_bias_raw", "dev_between_pool_bias_var_raw", "dev_between_pool_bias_var_recal"])}}
    C.jdump(summary, C.RESULTS / "d03_summary.json")
    # ---- markdown
    md = ["# D03 tables (generated by scripts/v81/d03_scale_selection.py; every number from results/d03_*.csv / d03_summary.json)", ""]
    md.append("## DEV formal units, mean over seeds 0-2 (prompt-macro unless named micro; mse = bias_sq + within per pool; bias_sq_macro = grand_bias^2 + between_pool_bias_var)\n")
    md.append(md_table(arm_mean, ["method", "mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "between_pool_bias_var", "mse_micro", "pair_accuracy_pooled", "pair_accuracy_macro", "regret_macro", "selected_human_mean_macro", "pooled_spearman"]))
    md.append("## exact human-mean ties vs the package reference's hist-based mean (pair accuracy; DEV arm means)\n")
    md.append(md_table(arm_mean, ["method", "pair_accuracy_pooled", "pair_accuracy_pooled_pkg_hist_mean", "informative_pairs_total", "informative_pairs_total_pkg_hist_mean"]))
    md.append("## DEV within-pool spread (prompt-macro variance of model scores vs of human means) and selection references\n")
    md.append(md_table(arm_mean, ["method", "score_variance_macro", "human_variance_macro", "random_expected_human_mean_macro", "selected_human_mean_macro", "observed_pool_max_macro", "regret_macro"]))
    md.append("## epoch-index-0 pattern: units whose CAL_SELECT minimum fell at index 0 (after the first epoch) vs the same arm's other seeds (DEV)\n")
    md.append(md_table(e0_rows, ["method", "e0_seed", "other_seeds", "other_epochs", "dev_pooled_spearman_e0", "dev_pooled_spearman_others_mean", "diff_pooled_spearman", "dev_mse_e0", "dev_mse_others_mean", "diff_mse", "diff_pair_pooled"]))
    md.append(f"selected epoch histogram (0-based): {epoch_hist}; by seed in arm order {C.ARMS}: {seed_epochs}\n")
    md.append("## DEV per unit (formal)\n")
    md.append(md_table(dev_fit, ["method", "seed", "checkpoint_id", "mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "between_pool_bias_var", "pair_accuracy_pooled", "regret_macro", "pooled_spearman", "n_zero_information_pools"]))
    md.append("## DEV pooled Spearman with prompt_group cluster bootstrap (2000 x 2718) and paired differences vs U-CS\n")
    md.append(md_table(sp_rows, ["method", "seed", "pooled_spearman", "ci_lo", "ci_hi", "diff_vs_ucs", "diff_ref", "diff_ci_lo", "diff_ci_hi"]))
    md.append("## other DEV systems (V7 units, external, kernels): ranking metrics; MSE only for 1-5 regressors\n")
    md.append(md_table([r for r in levels if r["file"] == "dev_all" and r["method"] not in C.ARMS], ["method", "seed", "mse_macro", "bias_sq_macro", "within_macro", "pair_accuracy_pooled", "regret_macro", "pooled_spearman", "n_zero_information_pools"]))
    md.append("## formal units on CAL_SELECT / CAL_PROB, arm means over seeds\n")
    md.append(md_table(cal_mean, ["role", "method", "mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "between_pool_bias_var", "pair_accuracy_pooled", "pair_accuracy_macro", "regret_macro", "pooled_spearman"]))
    md.append("## formal units on CAL_SELECT / CAL_PROB\n")
    md.append(md_table(cal_rows, ["role", "method", "seed", "mse_macro", "bias_sq_macro", "within_macro", "pair_accuracy_pooled", "pair_accuracy_macro", "regret_macro", "pooled_spearman"]))
    md.append("## export vs training-time selection record (CAL_SELECT, exported best.pt predictions vs the unit log at the selected epoch)\n")
    md.append(md_table(export_check, ["method", "seed", "selected_epoch", "log_cal_mse_at_selected", "export_cal_mse_macro", "diff_mse", "log_cal_pairwise_macro_at_selected", "export_cal_pairwise_macro", "diff_pairwise"], fmt="{:.6g}"))
    md.append("## tuning grid (seed 17): CAL_SELECT-selected lr vs the DEV-best lr (descriptive; DEV is never used for selection)\n")
    md.append(md_table(lr_check, ["arm", "selected_lr", "dev_best_lr_by_mse", "dev_best_lr_by_pooled_spearman", "dev_mse_selected", "dev_mse_best", "dev_rho_selected", "dev_rho_best"]))
    md.append("## selection audit from the unit logs (CAL_SELECT per epoch; mean-MSE rule = frozen V8-03 selection)\n")
    md.append(md_table(sel, ["method", "seed", "lr", "selected_epoch_log", "epoch_argmin_cal_mse", "epoch_argmax_cal_pairwise", "epoch_argmin_cal_rps", "cal_mse_at_selected", "cal_pairwise_at_selected", "cal_pairwise_best", "cal_pairwise_gap_best_minus_selected", "cal_mse_gap_pairwise_best_minus_selected", "cal_mse_last_epoch", "cal_pairwise_last_epoch"]))
    md.append("## selection criterion vs DEV outcome across the 24 units\n")
    md.append("```json\n" + __import__("json").dumps(rel, indent=1) + "\n```\n")
    md.append("## secondary: affine recalibration fitted on CAL_PROB, applied to DEV (order preserved when a > 0)\n")
    md.append(md_table(summary["recalibration"]["by_arm"], ["method", "affine_a", "affine_b", "dev_mse_macro_raw", "dev_mse_macro_recal", "dev_bias_sq_macro_raw", "dev_bias_sq_macro_recal", "dev_within_macro_raw", "dev_within_macro_recal", "dev_grand_bias_raw", "dev_between_pool_bias_var_raw", "dev_between_pool_bias_var_recal"]))
    md.append("## secondary: tuning units (seed 17, lr grid) on DEV\n")
    md.append(md_table(tune, ["arm", "lr", "is_selected_lr", "mse_macro", "bias_sq_macro", "within_macro", "grand_bias", "pair_accuracy_pooled", "regret_macro", "pooled_spearman"]))
    (C.RESULTS / "d03_tables.md").write_text("\n".join(md), encoding="utf-8")
    log("d03 done; crosscheck max abs diff vs diagnose per_pool:", maxdiff)


if __name__ == "__main__":
    main()
