"""V7C-01 / V7C-02 (closeout package docs/03): CAL / DEV diagnostics from the per-image readouts of scripts/v7x/v7c_readouts.py.  No GenAI label.

    python -I scripts/v7x/v7c_linkage.py --readouts /projects/EEG-foundation-model/yinghao/outputs_v7/x50_secondary --out reports/V7_CLOSEOUT/V7C_01_02

Tables (role, unit, arm, seed and counts on every row):
  within_prompt_metrics.csv   per readout (U_ann = mean human element label, an annotation-assisted reference; E_gt, E_auto, O, H_gt, Ecal_gt):
                              prompt-macro within-prompt pairwise accuracy (human-tied pairs excluded, model ties 1/2), Kendall tau-b, selected Hbar
                              (first-index ties; 'expected' = mean over tied maxima), observed regret, pooled SRCC / PLCC (background only); pools =
                              every prompt with >= 2 scored candidates; standard-size sensitivity: N = 9 (a seeded 9-subset of every prompt with >= 9
                              images) and N = 3 (its three disjoint triples), N = 1 = random expectation
  tail_error.csv              per prompt then averaged: mean element Brier, element Brier of the E_gt-selected image, top-1 / top-2 margin, regret,
                              max positive error (E_gt - U_ann); strata by FIT tertiles of the main element count and of the prompt length
  parser_coverage.csv         parser status counts per role, E_auto coverage, auto vs human element counts
  label_noise_sensitivity.csv votes of each image split once (seed 2718) into two halves -> within-prompt pairwise agreement of the halves' means,
                              3- and 6-vote images separately (array resampling, not independent raters)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

DATA = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data")
READOUTS = ("U_ann", "E_gt", "E_auto", "O", "H_gt", "Ecal_gt")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pool_metrics(rows, key):
    by = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            by[r["prompt_id"]].append(r)
    acc, tau, sel_f, sel_e, reg, n = [], [], [], [], [], 0
    for pid, rr in by.items():
        if len(rr) < 2:
            continue
        n += 1; s = np.array([r[key] for r in rr]); h = np.array([r["Hbar"] for r in rr]); a, b = np.triu_indices(len(rr), 1); dh = h[a] - h[b]; ok = dh != 0
        if ok.any():
            ds = s[a] - s[b]; acc.append(float(np.where(ds[ok] == 0, 0.5, (np.sign(ds[ok]) == np.sign(dh[ok])).astype(float)).mean()))
        if np.ptp(s) > 0 and np.ptp(h) > 0:
            tau.append(float(kendalltau(s, h).statistic))
        top = np.flatnonzero(s == s.max()); sel_f.append(float(h[top[0]])); sel_e.append(float(h[top].mean())); reg.append(float(h.max() - h[top[0]]))
    allr = [r for v in by.values() for r in v]; xs = np.array([r[key] for r in allr]); ys = np.array([r["Hbar"] for r in allr])
    return {"n_prompts": n, "pairwise_acc": np.mean(acc) if acc else None, "kendall_tau_b": np.mean(tau) if tau else None, "selected_Hbar_first": np.mean(sel_f) if sel_f else None,
            "selected_Hbar_expected": np.mean(sel_e) if sel_e else None, "regret_first": np.mean(reg) if reg else None,
            "pooled_srcc": float(spearmanr(xs, ys).statistic) if len(xs) > 2 else None, "pooled_plcc": float(pearsonr(xs, ys).statistic) if len(xs) > 2 else None}


def standard_size(rows, key, rng_seed=2718):
    by = defaultdict(list)
    for r in rows:
        by[r["prompt_id"]].append(r)
    n9, n3, n1 = [], [], []
    for pid in sorted(by):
        rr = sorted(by[pid], key=lambda r: r["image_id"])
        if len(rr) < 9 or any(r.get(key) is None for r in rr):
            continue
        sub = random.Random(f"{rng_seed}-{pid}").sample(rr, 9); h = np.array([r["Hbar"] for r in sub]); s = np.array([r[key] for r in sub])
        n9.append(h[np.flatnonzero(s == s.max())[0]]); n1.append(h.mean())
        n3.append(np.mean([h[k * 3:(k + 1) * 3][np.flatnonzero(s[k * 3:(k + 1) * 3] == s[k * 3:(k + 1) * 3].max())[0]] for k in range(3)]))
    return {"n_prompts_ge9": len(n9), "M1_random": np.mean(n1) if n1 else None, "M3": np.mean(n3) if n3 else None, "M9": np.mean(n9) if n9 else None}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--readouts", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = Path(a.readouts); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = {r["image_id"]: r for r in jl(DATA / "manifest_images.jsonl")}; fit = [r for r in man.values() if r["role"] == "FIT"]
    ec_cut = np.quantile([r.get("element_count_main") or 0 for r in fit], [1 / 3, 2 / 3]); pl_cut = np.quantile([len(r["prompt"].split()) for r in fit], [1 / 3, 2 / 3])
    units = sorted(p.name for p in R.iterdir() if p.is_dir()); wp, ss, tail, link = [], [], [], 0
    with open(out / "candidate_linkage.jsonl", "w") as LF:
        for u in units:
            arm, seed = u.split("-")[2], int(u.split("-s")[-1])
            for role in ("CAL_SELECT", "CAL_PROB", "DEV"):
                rows = jl(R / u / f"{role}_readouts.jsonl")
                for r in rows:
                    m = man[r["image_id"]]; r.update({"arm": arm, "seed": seed, "total_votes": len(m.get("overall_votes") or []), "generator_id": m["generator_id"]})
                    LF.write(json.dumps(r) + "\n"); link += 1
                for key in READOUTS:
                    wp.append({"role": role, "unit": u, "arm": arm, "seed": seed, "readout": key, **pool_metrics(rows, key)})
                    ss.append({"role": role, "unit": u, "arm": arm, "seed": seed, "readout": key, **standard_size(rows, key)})
                el = jl(Path("/projects/EEG-foundation-model/yinghao/outputs_v7/units") / u / "pred" / f"elements_{role}.jsonl"); bri = defaultdict(list)
                for e in el:
                    bri[e["image_id"]].append(e["y"] * (1 - e["p_prior"]) ** 2 + (1 - e["y"]) * e["p_prior"] ** 2)
                by = defaultdict(list)
                for r in rows:
                    if r["E_gt"] is not None:
                        by[r["prompt_id"]].append(r)
                for pid, rr in by.items():
                    if len(rr) < 2:
                        continue
                    s = np.array([r["E_gt"] for r in rr]); order = np.argsort(-s); h = np.array([r["Hbar"] for r in rr]); top = rr[order[0]]
                    m0 = man[top["image_id"]]; ec = m0.get("element_count_main") or 0; pl = len(m0["prompt"].split())
                    tail.append({"role": role, "unit": u, "arm": arm, "seed": seed, "prompt_id": pid, "n": len(rr),
                                 "mean_element_brier": float(np.mean([np.mean(bri[r["image_id"]]) for r in rr if bri.get(r["image_id"])])),
                                 "selected_element_brier": float(np.mean(bri[top["image_id"]])) if bri.get(top["image_id"]) else None,
                                 "margin_top1_top2": float(s[order[0]] - s[order[1]]), "regret": float(h.max() - h[order[0]]),
                                 "max_positive_error": float(max(r["E_gt"] - r["U_ann"] for r in rr if r["U_ann"] is not None)),
                                 "element_stratum": int(np.searchsorted(ec_cut, ec, side="right")), "length_stratum": int(np.searchsorted(pl_cut, pl, side="right"))})
    def write(name, rows):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write("within_prompt_metrics.csv", wp); write("standard_size_selection.csv", ss); write("tail_error_per_prompt.csv", tail)
    agg = defaultdict(list)
    for t in tail:
        agg[(t["role"], t["arm"], t["element_stratum"], t["length_stratum"])].append(t)
    write("tail_error.csv", [{"role": k[0], "arm": k[1], "element_stratum": k[2], "length_stratum": k[3], "n_prompt_unit_rows": len(v),
                              **{m: float(np.mean([x[m] for x in v if x[m] is not None])) for m in ("mean_element_brier", "selected_element_brier", "margin_top1_top2", "regret", "max_positive_error")}}
                             for k, v in sorted(agg.items())])
    cache = [r for r in jl("/home/infres/yinwang/CS_QMI/outputs/V7_CLOSEOUT/auto/evalmuse_cal_dev_parser_cache.jsonl") if "prompt" in r]; cov = []
    for role in ("CAL_SELECT", "CAL_PROB", "DEV"):
        cr = [r for r in cache if r["source"] == f"evalmuse_{role}"]; st = Counter(r["status"] for r in cr)
        rows = jl(R / units[0] / f"{role}_readouts.jsonl")
        cov.append({"role": role, "n_prompts": len(cr), **{f"status_{k}": v for k, v in st.items()}, "json_ok": sum(1 for r in cr if r["json_ok"]),
                    "prompts_with_unresolved": sum(1 for r in cr if r.get("unresolved")), "images": len(rows), "images_E_auto_defined": sum(1 for r in rows if r["E_auto"] is not None),
                    "mean_auto_elements": float(np.mean([r["n_auto_elements"] or 0 for r in rows])), "mean_human_elements": float(np.mean([r["n_gt_elements"] for r in rows]))})
    write("parser_coverage.csv", cov)
    noise = []
    for role in ("CAL_SELECT", "CAL_PROB", "DEV"):
        imgs = [m for m in man.values() if m["role"] == role]
        for nv in (3, 6):
            by = defaultdict(list)
            for m in imgs:
                v = list(m.get("overall_votes") or [])
                if len(v) == nv:
                    random.Random(f"2718-{m['image_id']}").shuffle(v); k = nv // 2; by[m["prompt_id"]].append((np.mean(v[:k]), np.mean(v[k:])))
            ag = []
            for pid, pairs in by.items():
                if len(pairs) < 2:
                    continue
                x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs]); i, j = np.triu_indices(len(pairs), 1); dx, dy = x[i] - x[j], y[i] - y[j]; ok = (dx != 0) & (dy != 0)
                if ok.any():
                    ag.append(float((np.sign(dx[ok]) == np.sign(dy[ok])).mean()))
            noise.append({"role": role, "votes_per_image": nv, "split": f"{nv // 2} vs {nv - nv // 2}", "n_prompts": len(ag), "within_prompt_pair_agreement": float(np.mean(ag)) if ag else None})
    write("label_noise_sensitivity.csv", noise)
    print(json.dumps({"linkage_rows": link, "units": len(units), "within_prompt_rows": len(wp), "tail_rows": len(tail)}, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
