"""Derived numbers cited in V81_02 / V81_03 / V81_04a (ranges, counts, shares) and a number check of the three reports.

  numbers : results/d02_d04a_report_numbers.json from the d02_* / d03_* / d04a_* results (no other input)
  verify  : every decimal number written in the three reports must equal (at its printed precision) some number in
            results/d02_* / d03_* / d04a_* / distribution.csv / d02_d04a_report_numbers.json; unmatched tokens are listed
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v81_common as C  # noqa: E402

R = C.RESULTS
REPORTS = [R.parent / "V81_02_DECISION_AGREEMENT.md", R.parent / "V81_03_SCALE_VS_SELECTION.md", R.parent / "V81_04a_DISTRIBUTION.md"]


def rcsv(name):
    return list(csv.DictReader(open(R / name, encoding="utf-8")))


def f(x):
    return None if x in (None, "NA", "") else float(x)


def numbers():
    out = {}
    pairs = rcsv("d02_dev_pairs.csv")
    same = [r for r in pairs if r["kind"] == "v8_same_seed"]
    cross = [r for r in pairs if r["kind"] == "v8_same_arm_cross_seed"]
    t1 = [f(r["top1_agreement"]) for r in same]
    out["d02_same_seed_top1_min_max_per_seed"] = [min(t1), max(t1)]
    out["d02_cross_seed_top1_min_max_per_pair"] = [min(f(r["top1_agreement"]) for r in cross), max(f(r["top1_agreement"]) for r in cross)]
    M = defaultdict(list)
    for r in same:
        M[(r["a_method"], r["b_method"])].append(r)
    mean_t1 = {k: float(np.mean([f(r["top1_agreement"]) for r in v])) for k, v in M.items()}
    mean_sp = {k: float(np.mean([f(r["spearman"]) for r in v])) for k, v in M.items()}
    out["d02_same_seed_top1_seedmean_min"] = min(mean_t1.values()); out["d02_same_seed_top1_seedmean_argmin"] = "/".join(min(mean_t1, key=mean_t1.get))
    out["d02_same_seed_top1_seedmean_max"] = max(mean_t1.values()); out["d02_same_seed_top1_seedmean_argmax"] = "/".join(max(mean_t1, key=mean_t1.get))
    out["d02_same_seed_spearman_seedmean_min_max"] = [min(mean_sp.values()), max(mean_sp.values())]
    out["d02_same_seed_n_pairs"] = len(same)
    out["d02_dev_same_seed_all_top1_mean"] = json.load(open(R / "d02_summary.json"))["headline"]["dev_same_seed_all_pairs"]["top1_agreement"]
    excl = [r for r in same if f(r["delta_ci_lo"]) > 0 or f(r["delta_ci_hi"]) < 0]
    out["d02_same_seed_n_interval_excludes_0"] = len(excl)
    out["d02_same_seed_n_interval_excludes_0_involving_list"] = sum("U-LIST" in (r["a_method"], r["b_method"]) for r in excl)
    out["d02_same_seed_n_interval_excludes_0_list_better"] = sum((r["b_method"] == "U-LIST" and f(r["overall_delta_a_minus_b"]) < 0) or (r["a_method"] == "U-LIST" and f(r["overall_delta_a_minus_b"]) > 0) for r in excl)
    ucs = [r for r in same if r["a_method"] == "U-CS"]
    out["d02_ucs_same_seed_n"] = len(ucs)
    out["d02_ucs_same_seed_n_interval_excludes_0"] = sum(f(r["delta_ci_lo"]) > 0 or f(r["delta_ci_hi"]) < 0 for r in ucs)
    out["d02_ucs_same_seed_delta_min_max"] = [min(f(r["overall_delta_a_minus_b"]) for r in ucs), max(f(r["overall_delta_a_minus_b"]) for r in ucs)]
    out["d02_swap_identity_residual_max_abs"] = max(abs(f(r["swap_identity_residual"])) for r in pairs if f(r["swap_identity_residual"]) is not None)
    cons = rcsv("d02_pool_consensus.csv")
    for k in ("share_all8_same_image", "share_scale6_same_image", "mean_distinct_selections", "share_ge3_distinct", "oracle_minus_ucs", "ucs_regret_macro"):
        v = [f(r[k]) for r in cons]
        out[f"d02_consensus_{k}_min_max"] = [min(v), max(v)]
    ext = rcsv("d02_external_by_arm.csv")
    ue = [r for r in ext if r["arm"] == "U-CS"]
    for prefix, sel in (("kernels", lambda r: r["other"].startswith("K-")), ("v7units", lambda r: r["other"].startswith("V7-")),
                        ("vqa_like", lambda r: r["other"] in ("B1_VQAScore", "B2_VQAScore_Qwen25VL7B", "B5_HPSv3", "B4T_TIFA", "B4D_DSG", "B3_PNVQA_own", "B3_PNVQA_evalmuse"))):
        rows = [r for r in ue if sel(r)]
        for k in ("top1_agreement", "spearman", "overall_delta_a_minus_b"):
            v = [f(r[k]) for r in rows]
            out[f"d02_ucs_vs_{prefix}_{k}_min_max"] = [min(v), max(v)]
    # ---- D03
    am = {r["method"]: r for r in rcsv("d03_dev_arm_mean.csv")}
    scale = C.SCALE_ARMS
    for k in ("mse_macro", "bias_sq_macro", "within_macro", "between_pool_bias_var", "grand_bias", "pair_accuracy_pooled", "regret_macro", "pooled_spearman", "score_variance_macro"):
        v = [f(am[a][k]) for a in scale]
        out[f"d03_scale6_{k}_min_max"] = [min(v), max(v)]
        v8 = [f(am[a][k]) for a in C.ARMS]
        out[f"d03_all8_{k}_min_max"] = [min(v8), max(v8)]
    out["d03_ucs_bias_share_of_mse"] = f(am["U-CS"]["bias_sq_macro"]) / f(am["U-CS"]["mse_macro"])
    out["d03_ucs_within_share_of_mse"] = f(am["U-CS"]["within_macro"]) / f(am["U-CS"]["mse_macro"])
    out["d03_scale6_bias_share_min_max"] = [min(f(am[a]["bias_sq_macro"]) / f(am[a]["mse_macro"]) for a in scale), max(f(am[a]["bias_sq_macro"]) / f(am[a]["mse_macro"]) for a in scale)]
    out["d03_pair_accuracy_pkg_minus_exact_min_max"] = [min(f(am[a]["pair_accuracy_pooled_pkg_hist_mean"]) - f(am[a]["pair_accuracy_pooled"]) for a in C.ARMS),
                                                        max(f(am[a]["pair_accuracy_pooled_pkg_hist_mean"]) - f(am[a]["pair_accuracy_pooled"]) for a in C.ARMS)]
    sp = rcsv("d03_pooled_spearman.csv")
    v8sp = [r for r in sp if r["method"] in C.ARMS and r["method"] != "U-CS"]
    out["d03_n_v8_diffs_vs_ucs"] = len(v8sp)
    out["d03_n_v8_diffs_interval_excludes_0"] = sum(f(r["diff_ci_lo"]) > 0 or f(r["diff_ci_hi"]) < 0 for r in v8sp)
    out["d03_n_v8_diffs_interval_excludes_0_non_rank"] = sum((f(r["diff_ci_lo"]) > 0 or f(r["diff_ci_hi"]) < 0) and r["method"] != "U-RANK" for r in v8sp)
    out["d03_n_v8_diffs_interval_above_0"] = sum(f(r["diff_ci_lo"]) > 0 for r in v8sp)
    bce = [r for r in sp if r["method"] == "U-BCE"]
    out["d03_bce_minus_ucs_min_max"] = [min(f(r["diff_vs_ucs"]) for r in bce), max(f(r["diff_vs_ucs"]) for r in bce)]
    rank = [r for r in sp if r["method"] == "U-RANK"]
    out["d03_rank_minus_ucs_min_max"] = [min(f(r["diff_vs_ucs"]) for r in rank), max(f(r["diff_vs_ucs"]) for r in rank)]
    v7 = [r for r in sp if r["method"].startswith("V7-")]
    out["d03_v7_minus_ucs_min_max"] = [min(f(r["diff_vs_ucs"]) for r in v7), max(f(r["diff_vs_ucs"]) for r in v7)]
    out["d03_v7_n_interval_excludes_0"] = sum(f(r["diff_ci_lo"]) > 0 or f(r["diff_ci_hi"]) < 0 for r in v7)
    out["d03_v7_pooled_spearman_min_max"] = [min(f(r["pooled_spearman"]) for r in v7), max(f(r["pooled_spearman"]) for r in v7)]
    ucs_sp = [f(r["pooled_spearman"]) for r in sp if r["method"] == "U-CS"]
    out["d03_ucs_pooled_spearman_min_max"] = [min(ucs_sp), max(ucs_sp)]
    rec = rcsv("d03_recalibration.csv")
    sc = [r for r in rec if r["method"] in scale]
    out["d03_recal_scale6_mse_change_min_max"] = [min(f(r["dev_mse_macro_recal"]) - f(r["dev_mse_macro_raw"]) for r in sc), max(f(r["dev_mse_macro_recal"]) - f(r["dev_mse_macro_raw"]) for r in sc)]
    out["d03_recal_affine_a_min_max"] = [min(f(r["affine_a"]) for r in rec), max(f(r["affine_a"]) for r in rec)]
    d3 = json.load(open(R / "d03_summary.json"))
    out["d03_epoch0_n"] = d3["epoch0_pattern"]["n_e0_units_with_non_e0_sibling"]
    e0 = d3["epoch0_pattern"]["rows"]
    out["d03_epoch0_diff_spearman_min_max"] = [min(r["diff_pooled_spearman"] for r in e0), max(r["diff_pooled_spearman"] for r in e0)]
    out["d03_epoch0_diff_mse_min_max"] = [min(r["diff_mse"] for r in e0), max(r["diff_mse"] for r in e0)]
    sel = rcsv("d03_selection_epochs.csv")
    out["d03_n_seed1_e0"] = sum(r["seed"] == "1" and r["selected_epoch_log"] == "0" for r in sel)
    out["d03_n_seed02_e0_non_rank"] = sum(r["seed"] in ("0", "2") and r["selected_epoch_log"] == "0" and r["method"] != "U-RANK" for r in sel)
    tl = rcsv("d03_tune_lr_check.csv")
    out["d03_tune_n_dev_best_mse_lr_equals_selected"] = sum(r["dev_best_lr_by_mse"] == r["selected_lr"] for r in tl)
    out["d03_tune_n_dev_best_rho_lr_equals_selected"] = sum(r["dev_best_lr_by_pooled_spearman"] == r["selected_lr"] for r in tl)
    # ---- D04a
    a4 = {(r["method"], r["role"]): r for r in rcsv("d04a_arm_mean.csv")}
    for k in ("rps_random_rater_macro", "cdf_fit_mse_macro", "hist_log_loss_macro", "mu_mse_macro", "rating_variance_macro", "J_mean"):
        v = [f(a4[(a, "DEV")][k]) for a in ("U-CS", "U-BCE", "U-RPS", "U-CE")]
        out[f"d04a_dev_dist4_{k}_min_max"] = [min(v), max(v)]
        v = [f(a4[(a, "DEV")][k]) for a in ("U-MSE", "U-L1", "U-RANK", "U-LIST")]
        out[f"d04a_dev_point4_{k}_min_max"] = [min(v), max(v)]
    out["d04a_dev_vote_constant_share_of_ucs_rps"] = f(a4[("U-CS", "DEV")]["vote_constant_macro"]) / f(a4[("U-CS", "DEV")]["rps_random_rater_macro"])
    pc = rcsv("d04a_paired_contrasts_dev.csv")
    bce = [r for r in pc if r["b"] == "U-BCE"]
    out["d04a_ucs_vs_bce_n_intervals_excluding_0"] = sum((f(r[f"d_{m}_lo"]) > 0 or f(r[f"d_{m}_hi"]) < 0) for r in bce for m in ("rps", "log_loss", "mu_mse", "J_mean"))
    out["d04a_ucs_vs_bce_n_intervals"] = 4 * len(bce)
    pq = rcsv("d04a_pair_qdiff_mean.csv")
    dist = [r for r in pq if r["kind"] == "same_seed" and r["a"] in ("U-CS", "U-BCE", "U-RPS", "U-CE") and r["b"] in ("U-CS", "U-BCE", "U-RPS", "U-CE")]
    pt = [r for r in pq if r["kind"] == "same_seed" and ((r["a"] in ("U-CS", "U-BCE", "U-RPS", "U-CE")) != (r["b"] in ("U-CS", "U-BCE", "U-RPS", "U-CE")))]
    cs = [r for r in pq if r["kind"] == "same_arm_cross_seed"]
    for name, rows in (("dist_vs_dist", dist), ("dist_vs_point", pt), ("cross_seed", cs)):
        out[f"d04a_qdiff_{name}_L1_close_min_max"] = [min(f(r["mean_L1_q_given_close_mu"]) for r in rows), max(f(r["mean_L1_q_given_close_mu"]) for r in rows)]
        out[f"d04a_qdiff_{name}_L1_min_max"] = [min(f(r["mean_L1_q"]) for r in rows), max(f(r["mean_L1_q"]) for r in rows)]
    spr = rcsv("d04a_spread.csv")
    for grp, arms in (("dist4", ("U-CS", "U-BCE", "U-RPS", "U-CE")), ("point4", ("U-MSE", "U-L1", "U-RANK", "U-LIST"))):
        rows = [r for r in spr if r["method"] in arms]
        for k in ("mean_predicted_rating_variance", "mean_pred_var_3votes", "mean_pred_var_6votes", "spearman_pred_var_vs_obs_var"):
            out[f"d04a_spread_{grp}_{k}_min_max"] = [min(f(r[k]) for r in rows), max(f(r[k]) for r in rows)]
    # ---- independent cross-check: V7C-40 DEV table (V7 scorer, manifest overall_mean) vs D03 pooled Spearman / pooled pair accuracy
    x40p = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v7/reports/V7_CLOSEOUT/V7C_DEV_results/x40_dev_external.json")
    x40 = json.load(open(x40p))["overall"]
    lv = {(r["method"], r["seed"]): r for r in rcsv("d03_model_levels.csv") if r["file"] == "dev_all"}
    name_map = {"B0_FGA_released_O (exposed)": ("B0_FGA_released_exposed", "-1"), "B1_VQAScore": ("B1_VQAScore", "-1"), "B2_VQAScore_Qwen25VL7B": ("B2_VQAScore_Qwen25VL7B", "-1"),
                "B5_HPSv3": ("B5_HPSv3", "-1"), "B4T_TIFA": ("B4T_TIFA", "-1"), "B4D_DSG": ("B4D_DSG", "-1"), "B3_PNVQA_own": ("B3_PNVQA_own", "-1"),
                "B3_PNVQA_evalmuse_elements": ("B3_PNVQA_evalmuse", "-1")}
    for s_ in C.SEEDS:
        name_map[f"N0_s{s_}_O"] = ("V7-N0_native_l1", str(s_))
        for tag in ("M0_semantic_l1", "M1_semantic_logistic", "M2_semantic_vcs"):
            name_map[f"{tag}-s{s_}_O"] = (f"V7-{tag}", str(s_))
    xs = []
    for r in x40:
        if r["system"] in name_map and name_map[r["system"]] in lv:
            m = lv[name_map[r["system"]]]
            xs.append({"system": r["system"], "d_srcc": f(m["pooled_spearman"]) - r["srcc"], "d_pairwise": f(m["pair_accuracy_pooled"]) - r["pairwise_acc"],
                       "d_pairwise_pkg_hist_mean": f(m["pair_accuracy_pooled_pkg_hist_mean"]) - r["pairwise_acc"]})
    out["crosscheck_x40_source"] = str(x40p)
    out["crosscheck_x40_n_systems"] = len(xs)
    out["crosscheck_x40_max_abs_d_srcc"] = max(abs(x["d_srcc"]) for x in xs)
    out["crosscheck_x40_max_abs_d_pairwise_exact"] = max(abs(x["d_pairwise"]) for x in xs)
    out["crosscheck_x40_max_abs_d_pairwise_pkg_hist_mean"] = max(abs(x["d_pairwise_pkg_hist_mean"]) for x in xs)
    C.jdump({"generated_by": "scripts/v81/d02_d04a_report_numbers.py numbers", "numbers": out}, R / "d02_d04a_report_numbers.json")
    print(json.dumps(out, indent=1))


def all_result_numbers():
    vals = []
    files = [p for p in R.glob("d02_*") if p.suffix in (".csv", ".json")] + [p for p in R.glob("d03_*") if p.suffix in (".csv", ".json")] + \
            [p for p in R.glob("d04a_*") if p.suffix in (".csv", ".json")] + [R / "distribution.csv"]
    num = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+")
    for p in files:
        if p.name == "d02_agreement_rows_ucs.csv":
            continue   # per-prompt detail: not cited
        txt = p.read_text(encoding="utf-8")
        for t in num.findall(txt):
            try:
                vals.append(float(t))
            except ValueError:
                pass
    return np.unique(np.array(vals))


def verify():
    vals = all_result_numbers()
    token = re.compile(r"(?<![\w.])[-+−]?\d+\.\d+(?![\w.])")
    bad_total = 0
    for rep in REPORTS:
        if not rep.exists():
            print(f"missing report {rep}")
            continue
        txt = rep.read_text(encoding="utf-8")
        txt = re.sub(r"`[^`]*`", " ", txt)                         # code spans (paths, ids) are not numbers
        txt = re.sub(r"\d{4}-\d{2}-\d{2}T[\d:]+Z?", " ", txt)        # timestamps
        txt = "\n".join("" if l.lstrip().startswith("#") else l for l in txt.split("\n"))   # section headings (numbers like 2.1)
        txt = re.sub(r"§\s*\d+(?:\.\d+)*", " ", txt)                    # section references
        bad = []
        for m in token.finditer(txt):
            s = m.group(0).replace("−", "-").replace("+", "")
            x = float(s)
            dec = len(s.split(".")[1])
            tol = 0.5 * 10 ** (-dec) + 1e-12
            ok = np.any(np.abs(vals - x) <= tol) or np.any(np.abs(np.abs(vals) - abs(x)) <= tol)
            if not ok:
                line = txt.count("\n", 0, m.start()) + 1
                bad.append((line, m.group(0)))
        bad_total += len(bad)
        print(f"{rep.name}: {len(token.findall(txt))} numbers checked, {len(bad)} unmatched {bad[:40]}")
    return bad_total


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "numbers"
    if mode == "numbers":
        numbers()
    else:
        sys.exit(1 if verify() else 0)
