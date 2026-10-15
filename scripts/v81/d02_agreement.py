"""V8.1 D02 — candidate decision agreement (FULL_PLAN_CN 04 §D02).

`run`       : package reference/diagnose.py (unchanged) on the combined normalised files -> outputs_v81/diag/<name>.json
`summarise` : compact results under reports/V8_1_DIAGNOSTICS/results/d02_* (matrices, cross-seed stability, pool consensus,
              U-CS vs external systems / kernels / V7 units, label-free GenAI-Rank + TIFA160 agreement) + d02_tables.md.
All agreement numbers are the package's own definitions (core.decision_agreement / swap_decomposition / cluster_bootstrap):
exact top-1 on the frozen candidate order, top-set Jaccard, within-pool Spearman / Kendall tau-b, selected-quality delta
(development labels only), prompt_group cluster bootstrap 2000 x seed 2718, no p-values.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v81_common as C  # noqa: E402

RUNS = {"dev_all": ("combined/dev_all.jsonl", "development"),
        "cal_select_v8fit": ("combined/cal_select_v8fit.jsonl", "development"),
        "cal_prob_v8fit": ("combined/cal_prob_v8fit.jsonl", "development"),
        "genai_rank_v8fit": ("labelfree/genai_rank_v8fit.jsonl", "none"),
        "tifa160_v8fit": ("labelfree/tifa160_v8fit.jsonl", "none")}
PAIR_METRICS = ["top1_agreement", "topset_intersects", "topset_jaccard", "top_tie_a", "top_tie_b", "spearman", "kendall_tau_b"]


def log(*a):
    print(*a, flush=True)


def run():
    C.DIAG.mkdir(parents=True, exist_ok=True)
    for name, (rel, mode) in RUNS.items():
        out = C.DIAG / f"{name}.json"
        if out.exists():
            log(f"[diag] {name} exists, skipped")
            continue
        rows = C.NORM / rel
        cmd = [C.PYTHON, str(C.REF / "diagnose.py"), "--rows", str(rows), "--out", str(out), "--label-mode", mode]
        log("[diag]", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"diagnose failed for {name}: {r.stderr[-3000:]}")
        log(r.stdout.strip())


# ------------------------------------------------------------------------------------------------------------- orientation
def arm_rank(m):
    return C.ARMS.index(m) if m in C.ARMS else 100


def oriented(ps):
    """Return (a, b, metrics, swap, ci) with the V8 arm (or the earlier arm / lower seed) as side a; deltas follow a - b."""
    a, b = ps["a"], ps["b"]
    flip = False
    ra, rb = arm_rank(a["method"]), arm_rank(b["method"])
    if ra != rb:
        flip = rb < ra
    elif a["method"] == b["method"]:
        flip = b["seed"] < a["seed"]
    else:
        flip = str(b["method"]) < str(a["method"])
    pm = dict(ps["pool_macro"])
    swap = dict(ps.get("swap", {}))
    ci = dict(ps.get("quality_delta_ci", {}))
    if flip:
        a, b = b, a
        pm["top_tie_a"], pm["top_tie_b"] = pm.get("top_tie_b"), pm.get("top_tie_a")
        for k in ("delta_when_disagree", "overall_delta"):
            if swap.get(k) is not None:
                swap[k] = -swap[k]
        if ci.get("mean") is not None:
            ci["mean"] = -ci["mean"]
        if ci.get("ci95"):
            ci["ci95"] = [-ci["ci95"][1], -ci["ci95"][0]]
    return a, b, pm, swap, ci, ps


def pair_row(ps):
    a, b, pm, swap, ci, raw = oriented(ps)
    row = {"a_method": a["method"], "a_seed": a["seed"], "a_ckpt": a["checkpoint_id"], "b_method": b["method"], "b_seed": b["seed"], "b_ckpt": b["checkpoint_id"],
           "n_pools": raw["n_pools"]}
    for k in PAIR_METRICS:
        row[k] = pm.get(k)
        row[f"n_def_{k}"] = raw["n_defined"].get(k)
    if swap:
        row.update({"disagreement_rate": swap["disagreement_rate"], "delta_when_disagree": swap["delta_when_disagree"], "overall_delta_a_minus_b": swap["overall_delta"],
                    "swap_identity_residual": swap["identity_residual"], "delta_ci_lo": ci["ci95"][0] if ci.get("ci95") else None,
                    "delta_ci_hi": ci["ci95"][1] if ci.get("ci95") else None, "delta_n_clusters": ci.get("n_clusters"), "delta_boot_status": ci.get("status")})
    return row


def pair_kind(r):
    a_in, b_in = r["a_method"] in C.ARMS, r["b_method"] in C.ARMS
    if a_in and b_in:
        if r["a_method"] == r["b_method"]:
            return "v8_same_arm_cross_seed"
        return "v8_same_seed" if r["a_seed"] == r["b_seed"] else "v8_cross"
    if a_in != b_in:
        return "v8_vs_other"
    return "other_vs_other"


# -------------------------------------------------------------------------------------------------------------- matrices
def matrices(rows, metric, seeds=C.SEEDS):
    out = {}
    for s in seeds:
        M = np.full((8, 8), np.nan)
        for r in rows:
            if pair_kind(r) == "v8_same_seed" and r["a_seed"] == s and r.get(metric) is not None:
                i, j = C.ARMS.index(r["a_method"]), C.ARMS.index(r["b_method"])
                M[i, j] = r[metric]
                M[j, i] = -r[metric] if "delta" in metric else r[metric]   # a - b deltas are antisymmetric
        out[f"s{s}"] = M
    out["mean"] = np.nanmean(np.stack([out[f"s{s}"] for s in seeds]), axis=0)
    return out


def matrix_rows(mats, metric):
    rows = []
    for key, M in mats.items():
        for i, a in enumerate(C.ARMS):
            rows.append({"metric": metric, "seed": key, "arm": a, **{b: (None if np.isnan(M[i, j]) else float(M[i, j])) for j, b in enumerate(C.ARMS)}})
    return rows


def md_matrix(M, title, fmt="{:.3f}"):
    L = [f"**{title}**", "", "| arm | " + " | ".join(C.ARMS) + " |", "|---|" + "---|" * 8]
    for i, a in enumerate(C.ARMS):
        L.append(f"| {a} | " + " | ".join("—" if i == j or np.isnan(M[i, j]) else fmt.format(M[i, j]) for j in range(8)) + " |")
    return "\n".join(L) + "\n"


def md_table(rows, fields, headers=None, fmt="{:.3f}"):
    headers = headers or fields
    L = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        cells = []
        for f in fields:
            v = r.get(f)
            cells.append("NA" if v is None else (fmt.format(v) if isinstance(v, float) else str(v)))
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------------------------------------------------- consensus
def consensus(per_pool):
    """Per seed: how many distinct images the 8 arms select per DEV pool; oracle-over-arm-choices reference (labels fixed, not deployable)."""
    sel = defaultdict(dict)   # (seed, pool) -> arm -> (image, human_mean, pool_max)
    for r in per_pool:
        if r["role"] == "DEV" and r["method"] in C.ARMS:
            sel[(r["seed"], r["pool_id"])][r["method"]] = (r["selected_image_id"], r.get("selected_human_mean"), r.get("observed_pool_max"), r.get("n_candidates"))
    out = []
    for s in C.SEEDS:
        pools = [v for (ss, p), v in sel.items() if ss == s and len(v) == 8]
        n_distinct = np.array([len({v[a][0] for a in C.ARMS}) for v in pools])
        scale_same = np.array([len({v[a][0] for a in C.SCALE_ARMS}) == 1 for v in pools])
        oracle = np.array([max(v[a][1] for a in C.ARMS) for v in pools])
        ucs = np.array([v["U-CS"][1] for v in pools])
        best_single = {a: float(np.mean([v[a][1] for v in pools])) for a in C.ARMS}
        pool_max = np.array([v["U-CS"][2] for v in pools])
        single_cand = np.array([v["U-CS"][3] == 1 for v in pools])
        out.append({"seed": s, "n_pools": len(pools), "n_single_candidate_pools": int(single_cand.sum()),
                    "share_all8_same_image": float((n_distinct == 1).mean()), "share_scale6_same_image": float(scale_same.mean()),
                    "mean_distinct_selections": float(n_distinct.mean()), "max_distinct_selections": int(n_distinct.max()),
                    "share_ge3_distinct": float((n_distinct >= 3).mean()),
                    "ucs_selected_human_mean": float(ucs.mean()), "best_single_arm": max(best_single, key=best_single.get),
                    "best_single_arm_selected_human_mean": max(best_single.values()),
                    "oracle_over_arm_choices_selected_human_mean": float(oracle.mean()), "oracle_minus_ucs": float((oracle - ucs).mean()),
                    "pool_max_human_mean": float(pool_max.mean()), "ucs_regret_macro": float((pool_max - ucs).mean())})
    return out


# -------------------------------------------------------------------------------------------------------------- summarise
def summarise():
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    prov = {}
    # ---- labelled DEV (dev_all)
    D = C.jload(C.DIAG / "dev_all.json")
    prov["dev_all"] = {"diag": str(C.DIAG / "dev_all.json"), "input": D["input"], "n_rows": D["n_rows"], "n_models": len(D["model_levels"])}
    pairs = [pair_row(ps) for ps in D["pair_summaries"]]
    for r in pairs:
        r["kind"] = pair_kind(r)
    pairs.sort(key=lambda r: (r["kind"], arm_rank(r["a_method"]), r["a_method"], r["a_seed"], arm_rank(r["b_method"]), r["b_method"], r["b_seed"]))
    C.csv_write(C.RESULTS / "d02_dev_pairs.csv", pairs)
    mat_rows, mats = [], {}
    for metric in ("top1_agreement", "topset_jaccard", "spearman", "kendall_tau_b", "disagreement_rate", "overall_delta_a_minus_b"):
        mats[metric] = matrices(pairs, metric)
        mat_rows += matrix_rows(mats[metric], metric)
    C.csv_write(C.RESULTS / "d02_matrix_same_seed.csv", mat_rows)
    cross = defaultdict(list)
    for r in pairs:
        if r["kind"] == "v8_same_arm_cross_seed":
            cross[r["a_method"]].append(r)
    cross_rows = [{"arm": a, "n_pairs": len(rs), **{k: float(np.mean([r[k] for r in rs if r[k] is not None])) for k in ("top1_agreement", "topset_jaccard", "spearman", "kendall_tau_b", "disagreement_rate")},
                   "abs_overall_delta_mean": float(np.mean([abs(r["overall_delta_a_minus_b"]) for r in rs]))} for a, rs in cross.items()]
    C.csv_write(C.RESULTS / "d02_cross_seed_stability.csv", cross_rows)
    cons = consensus(D["per_pool"])
    C.csv_write(C.RESULTS / "d02_pool_consensus.csv", cons)
    # ---- U-CS (and every arm) vs non-V8 systems
    ext = [r for r in pairs if r["kind"] == "v8_vs_other"]
    ext_mean = defaultdict(list)
    for r in ext:
        ext_mean[(r["a_method"], r["b_method"], r["b_seed"] if r["b_seed"] == -1 else "same")].append(r)
    ext_rows = []
    for (a, b, sd), rs in sorted(ext_mean.items(), key=lambda kv: (arm_rank(kv[0][0]), str(kv[0][1]))):
        row = {"arm": a, "other": b, "other_seed": sd, "n_pairs": len(rs)}
        for k in ("top1_agreement", "topset_jaccard", "spearman", "kendall_tau_b", "top_tie_b", "disagreement_rate", "delta_when_disagree", "overall_delta_a_minus_b"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        los = [r["delta_ci_lo"] for r in rs if r.get("delta_ci_lo") is not None]
        his = [r["delta_ci_hi"] for r in rs if r.get("delta_ci_hi") is not None]
        row["delta_ci_lo_min"] = min(los) if los else None
        row["delta_ci_hi_max"] = max(his) if his else None
        ext_rows.append(row)
    C.csv_write(C.RESULTS / "d02_external_by_arm.csv", ext_rows)
    other_levels = [{"method": m["method"], "seed": m["seed"], "checkpoint_id": m["checkpoint_id"], "n_pools": m["n_pools"], "n_images": m["n_images"],
                     **{k: m["pool_macro"].get(k) for k in ("pair_accuracy", "regret", "selected_human_mean", "spearman_to_human", "mse", "top_tie_count")},
                     "n_def_spearman": m["n_defined"].get("spearman_to_human")} for m in D["model_levels"]]
    C.csv_write(C.RESULTS / "d02_dev_model_levels.csv", other_levels)
    # ---- per-prompt detail for the main U-CS contrasts (same seed: the other 7 arms, V7-N0 / V7-M2; released: B0, B1); full detail stays in the diag JSON
    keep_other = set(C.ARMS) | {"V7-N0_native_l1", "V7-M2_semantic_vcs", "B0_FGA_released_exposed", "B1_VQAScore"}
    detail = []
    for r in D["agreement_rows"]:
        a, b = r["a"], r["b"]
        if b["method"] == "U-CS" and a["method"] != "U-CS":
            a, b = b, a
            sign = -1.0
        else:
            sign = 1.0
        if a["method"] != "U-CS" or b["method"] not in keep_other or b["method"] == "U-CS":
            continue
        if b["seed"] != -1 and b["seed"] != a["seed"]:
            continue
        dq = r.get("selected_quality_delta")
        detail.append({"a_method": a["method"], "a_seed": a["seed"], "b_method": b["method"], "b_seed": b["seed"], "pool_id": r["pool_id"], "cluster_id": r["cluster_id"],
                       "top1_agreement": r["top1_agreement"], "topset_jaccard": r["topset_jaccard"], "spearman": r["spearman"], "kendall_tau_b": r["kendall_tau_b"],
                       "selected_quality_delta_a_minus_b": None if dq is None else sign * dq})
    detail.sort(key=lambda r: (r["b_method"], r["a_seed"], r["pool_id"]))
    C.csv_write(C.DIAG / "d02_agreement_rows_ucs.csv", detail)   # per-prompt rows carry label-derived deltas: kept off the public repo
    # ---- label-free
    lf_rows, lf_mats, lf_ties = [], {}, []
    for name in ("genai_rank_v8fit", "tifa160_v8fit"):
        L = C.jload(C.DIAG / f"{name}.json")
        prov[name] = {"diag": str(C.DIAG / f"{name}.json"), "input": L["input"], "n_rows": L["n_rows"], "labelled": L["labelled"]}
        prs = [pair_row(ps) for ps in L["pair_summaries"]]
        for r in prs:
            r["kind"] = pair_kind(r)
            r["dataset"] = name
        lf_rows += prs
        lf_mats[name] = {m: matrices(prs, m) for m in ("top1_agreement", "topset_jaccard", "spearman", "kendall_tau_b")}
        ties = defaultdict(list)
        for p in L["per_pool"]:
            ties[(p["method"], p["seed"])].append(p["top_tie_count"] > 1)
        for (m, s), v in sorted(ties.items(), key=lambda kv: (arm_rank(kv[0][0]), kv[0][1])):
            lf_ties.append({"dataset": name, "method": m, "seed": s, "n_pools": len(v), "share_pools_with_top_tie": float(np.mean(v))})
        cs = [r for r in prs if r["kind"] == "v8_same_arm_cross_seed"]
        for a in C.ARMS:
            rs = [r for r in cs if r["a_method"] == a]
            lf_ties.append({"dataset": name, "method": a, "seed": "cross-seed", "n_pools": rs[0]["n_pools"] if rs else None,
                            "share_pools_with_top_tie": None, "cross_seed_top1": float(np.mean([r["top1_agreement"] for r in rs])) if rs else None,
                            "cross_seed_spearman": float(np.mean([r["spearman"] for r in rs if r["spearman"] is not None])) if rs else None})
    C.csv_write(C.RESULTS / "d02_labelfree_pairs.csv", lf_rows)
    C.csv_write(C.RESULTS / "d02_labelfree_ties.csv", lf_ties)
    lf_mat_rows = []
    for name, mm in lf_mats.items():
        for metric, mats_ in mm.items():
            for r in matrix_rows(mats_, metric):
                lf_mat_rows.append({"dataset": name, **r})
    C.csv_write(C.RESULTS / "d02_labelfree_matrix.csv", lf_mat_rows)

    # ---- CAL_SELECT / CAL_PROB (development roles): same summaries as DEV for the V8 arms
    cal = {}
    for name in ("cal_select_v8fit", "cal_prob_v8fit"):
        Lc = C.jload(C.DIAG / f"{name}.json")
        prov[name] = {"diag": str(C.DIAG / f"{name}.json"), "input": Lc["input"], "n_rows": Lc["n_rows"]}
        prs = [pair_row(ps) for ps in Lc["pair_summaries"]]
        for r in prs:
            r["kind"] = pair_kind(r)
        sc = lambda r: r["a_method"] in C.SCALE_ARMS and r["b_method"] in C.SCALE_ARMS
        def m_(cond, key, rows=prs):
            v = [r[key] for r in rows if cond(r) and r.get(key) is not None]
            return float(np.mean(v)) if v else None
        cal[name] = {"n_pools": prs[0]["n_pools"],
                     "same_seed_all_top1": m_(lambda r: r["kind"] == "v8_same_seed", "top1_agreement"),
                     "same_seed_scale6_top1": m_(lambda r: r["kind"] == "v8_same_seed" and sc(r), "top1_agreement"),
                     "same_seed_rank_or_list_top1": m_(lambda r: r["kind"] == "v8_same_seed" and not sc(r), "top1_agreement"),
                     "cross_seed_top1": m_(lambda r: r["kind"] == "v8_same_arm_cross_seed", "top1_agreement"),
                     "same_seed_all_spearman": m_(lambda r: r["kind"] == "v8_same_seed", "spearman"),
                     "cross_seed_spearman": m_(lambda r: r["kind"] == "v8_same_arm_cross_seed", "spearman"),
                     "ucs_bce_top1": m_(lambda r: r["kind"] == "v8_same_seed" and {r["a_method"], r["b_method"]} == {"U-CS", "U-BCE"}, "top1_agreement")}
        C.csv_write(C.RESULTS / f"d02_{name}_pairs.csv", prs)
    # ---- headline summary
    def mean_of(rows, cond, key):
        v = [r[key] for r in rows if cond(r) and r.get(key) is not None]
        return float(np.mean(v)) if v else None

    same = [r for r in pairs if r["kind"] == "v8_same_seed"]
    scale = lambda r: r["a_method"] in C.SCALE_ARMS and r["b_method"] in C.SCALE_ARMS
    rank = lambda r: not scale(r)
    headline = {
        "dev_same_seed_all_pairs": {"n": len(same), **{k: mean_of(same, lambda r: True, k) for k in ("top1_agreement", "topset_jaccard", "spearman", "disagreement_rate")}},
        "dev_same_seed_scale6_pairs": {"n": sum(scale(r) for r in same), **{k: mean_of(same, scale, k) for k in ("top1_agreement", "topset_jaccard", "spearman", "disagreement_rate")}},
        "cal_roles": cal,
        "dev_same_seed_pairs_with_rank_or_list": {"n": sum(rank(r) for r in same), **{k: mean_of(same, rank, k) for k in ("top1_agreement", "topset_jaccard", "spearman", "disagreement_rate")}},
        "dev_cross_seed_same_arm": {"n": len([r for r in pairs if r["kind"] == "v8_same_arm_cross_seed"]),
                                    **{k: mean_of(pairs, lambda r: r["kind"] == "v8_same_arm_cross_seed", k) for k in ("top1_agreement", "topset_jaccard", "spearman", "disagreement_rate")}},
        "dev_ucs_vs_ubce": {k: mean_of(pairs, lambda r: {r["a_method"], r["b_method"]} == {"U-CS", "U-BCE"} and r["a_seed"] == r["b_seed"], k)
                            for k in ("top1_agreement", "topset_jaccard", "spearman", "disagreement_rate", "overall_delta_a_minus_b")},
        "dev_ucs_vs_external": {r["other"]: {k: r[k] for k in ("top1_agreement", "topset_jaccard", "spearman", "disagreement_rate", "delta_when_disagree", "overall_delta_a_minus_b", "delta_ci_lo_min", "delta_ci_hi_max")}
                                for r in ext_rows if r["arm"] == "U-CS"},
        "dev_pool_consensus": cons,
        "labelfree": {name: {"same_seed_top1_mean": float(np.nanmean(lf_mats[name]["top1_agreement"]["mean"][np.triu_indices(8, 1)])),
                             "same_seed_spearman_mean": float(np.nanmean(lf_mats[name]["spearman"]["mean"][np.triu_indices(8, 1)])),
                             "scale6_top1_mean": float(np.nanmean(lf_mats[name]["top1_agreement"]["mean"][:6, :6][np.triu_indices(6, 1)])),
                             "ucs_vs_ubce_top1": float(lf_mats[name]["top1_agreement"]["mean"][0, 1]),
                             "ucs_vs_urank_top1": float(lf_mats[name]["top1_agreement"]["mean"][0, 6]),
                             "cross_seed_top1_mean": mean_of(lf_rows, lambda r: r["dataset"] == name and r["kind"] == "v8_same_arm_cross_seed", "top1_agreement")}
                      for name in lf_mats},
    }
    C.jdump({"generated_by": "scripts/v81/d02_agreement.py summarise", "definitions": "package reference/core.py (decision_agreement, swap_decomposition, cluster_bootstrap 2000 x 2718)",
             "cluster": "prompt_group (V7 near-duplicate component) via component_id in the derived manifest", "provenance": prov, "headline": headline},
            C.RESULTS / "d02_summary.json")

    # ---- markdown tables for the report
    md = ["# D02 tables (generated by scripts/v81/d02_agreement.py summarise; every number from results/d02_*.csv / d02_summary.json)", ""]
    md.append("## DEV, 24 formal units: same-seed 8 x 8 matrices, mean over seeds 0-2 (prompt-macro over 307 pools)\n")
    for metric, title in (("top1_agreement", "exact top-1 agreement"), ("topset_jaccard", "top-set Jaccard"), ("spearman", "within-pool Spearman"),
                          ("kendall_tau_b", "within-pool Kendall tau-b"), ("overall_delta_a_minus_b", "selected-quality delta, row arm minus column arm (human mean 1-5)")):
        md.append(md_matrix(mats[metric]["mean"], title, "{:+.4f}" if "delta" in metric else "{:.3f}"))
    md.append("## per seed: top-1 agreement\n")
    for s in C.SEEDS:
        md.append(md_matrix(mats["top1_agreement"][f"s{s}"], f"seed {s}"))
    md.append("## cross-seed stability (same arm, seeds 0-1 / 0-2 / 1-2; mean of the 3 pairs)\n")
    md.append(md_table(cross_rows, ["arm", "top1_agreement", "topset_jaccard", "spearman", "kendall_tau_b", "disagreement_rate", "abs_overall_delta_mean"]))
    md.append("## pool consensus (DEV, per seed)\n")
    md.append(md_table(cons, ["seed", "n_pools", "n_single_candidate_pools", "share_all8_same_image", "share_scale6_same_image", "mean_distinct_selections", "share_ge3_distinct",
                              "ucs_selected_human_mean", "best_single_arm", "best_single_arm_selected_human_mean", "oracle_over_arm_choices_selected_human_mean", "oracle_minus_ucs", "pool_max_human_mean", "ucs_regret_macro"]))
    md.append("## U-CS vs external systems, kernel bridges and V7 units (mean over the U-CS seeds; other_seed = same means same-seed V7 unit)\n")
    md.append(md_table([r for r in ext_rows if r["arm"] == "U-CS"], ["other", "other_seed", "n_pairs", "top1_agreement", "topset_jaccard", "spearman", "kendall_tau_b", "top_tie_b", "disagreement_rate",
                                                                      "delta_when_disagree", "overall_delta_a_minus_b", "delta_ci_lo_min", "delta_ci_hi_max"]))
    md.append("## CAL_SELECT / CAL_PROB (development roles): same-seed and cross-seed agreement of the V8 arms\n")
    md.append(md_table([{"role": k, **v} for k, v in cal.items()], ["role", "n_pools", "same_seed_all_top1", "same_seed_scale6_top1", "same_seed_rank_or_list_top1", "cross_seed_top1", "same_seed_all_spearman", "cross_seed_spearman", "ucs_bce_top1"]))
    md.append("## label-free agreement (no label read): same-seed top-1 matrices, mean over seeds\n")
    for name in lf_mats:
        md.append(md_matrix(lf_mats[name]["top1_agreement"]["mean"], f"{name}: exact top-1 agreement"))
        md.append(md_matrix(lf_mats[name]["spearman"]["mean"], f"{name}: within-pool Spearman"))
    md.append("## label-free: top-tie share per model and cross-seed agreement\n")
    md.append(md_table(lf_ties, ["dataset", "method", "seed", "n_pools", "share_pools_with_top_tie", "cross_seed_top1", "cross_seed_spearman"]))
    (C.RESULTS / "d02_tables.md").write_text("\n".join(md), encoding="utf-8")
    log("d02 summarise done:", {k: v for k, v in headline.items() if k in ("dev_same_seed_scale6_pairs", "dev_cross_seed_same_arm", "dev_ucs_vs_ubce")})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["run", "summarise"])
    a = ap.parse_args()
    run() if a.mode == "run" else summarise()


if __name__ == "__main__":
    main()
