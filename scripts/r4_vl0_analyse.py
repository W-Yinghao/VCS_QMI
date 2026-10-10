"""R4-VL0 diagnostics (reports/R4/R4_VL0_PROTOCOL_20261011.md §Diagnostics 1–7) from the saved score matrices of scripts/r4_vl0_scores.py.
Exploratory and descriptive; items 3 and 4 use ground truth (oracle-assisted) and are not deployable procedures.  Every subset, K and permutation is
shared by all methods.  Writes reports/R4/R4_VL0_results.json and a markdown table.
    python scripts/r4_vl0_analyse.py [--datasets refcocog refcocoplus] [--backbone siglip2_b16]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_22_tableC import iou_xyxy  # noqa: E402

D = Path("/home/infres/yinwang/CS_QMI/outputs/R4_VL0"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R4"
OBJS, SEEDS, KS, ND = ("vcs", "js", "softmax"), (0, 1, 2), (4, 8, 16, 32, None), (1, 3, 7, 15, 31, None)


def inside_frac(b, t):
    ix = max(0.0, min(b[2], t[2]) - max(b[0], t[0])); iy = max(0.0, min(b[3], t[3]) - max(b[1], t[1]))
    return ix * iy / max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))


def geometry(meta):
    """Per image, per expression: IoU of every proposal with the target; per proposal the error-type stratum relative to that target."""
    G = []
    for m in meta:
        P = m["prop_boxes"]; tg = m["target"]; refs = m["ref_boxes"]; others = m["other_boxes"]
        allann = np.concatenate([refs, others]) if len(others) else refs; allcat = np.concatenate([m["ref_cat"], m["other_cat"]]) if len(others) else m["ref_cat"]
        if len(P) == 0:
            G.append({"iou": np.zeros((len(tg), 0)), "etype": [[] for _ in tg]}); continue
        iou_ann = np.stack([iou_xyxy(a, P) for a in allann])                      # [n_ann, n_prop]
        iou = np.stack([iou_ann[t] for t in tg])                                    # targets are referred objects: index into allann directly
        et = []
        for j, t in enumerate(tg):
            row = []
            for k in range(len(P)):
                if iou[j, k] >= 0.5:
                    row.append("valid"); continue
                other = [a for a in range(len(allann)) if a != t and iou_ann[a, k] >= 0.5]
                if other:
                    a = max(other, key=lambda a: iou_ann[a, k]); row.append("other_same_cat" if allcat[a] == m["ref_cat"][t] else "other_diff_cat")
                elif iou[j, k] >= 0.1 or inside_frac(P[k], refs[t]) >= 0.8:
                    row.append("part_or_duplicate")
                else:
                    row.append("no_matching_annotation")
            et.append(row)
        G.append({"iou": iou, "etype": et})
    return G


def per_image_metrics(meta, G, sc):
    """Per image dict of per-query arrays for every diagnostic; image-macro is taken later."""
    out = []
    for m, g, s in zip(meta, G, sc):
        cp = s["cos_prop"].numpy(); cr = s["cos_ref"].numpy(); iou = g["iou"]; nq = len(m["target"]); n_p = cp.shape[0]
        rec = {"all": np.zeros(nq), "covered": np.zeros(nq, bool), "topk": {}, "covk": {}, "orc": {}, "c2x2": {k: np.full(nq, np.nan) for k in ("A+C", "B+C", "A+D", "B+D")},
               "margin": np.full(nq, np.nan), "etype": [None] * nq}
        valid = iou >= 0.5 if n_p else np.zeros((nq, 0), bool)
        rec["covered"] = valid.any(1) if n_p else np.zeros(nq, bool)
        for K in KS:
            k = n_p if K is None else min(K, n_p)
            if k == 0:
                rec["topk"][str(K)] = np.zeros(nq); rec["covk"][str(K)] = np.zeros(nq); continue
            pred = cp[:k].argmax(0); rec["topk"][str(K)] = valid[np.arange(nq), pred].astype(float); rec["covk"][str(K)] = valid[:, :k].any(1).astype(float)
        rec["all"] = rec["topk"]["None"]
        rng = np.random.default_rng(int(m["image_id"]))
        perm = rng.permutation(n_p)
        for nd in ND:
            rec["orc"][str(nd)] = np.full(nq, np.nan)
        for j in range(nq):
            t = m["target"][j]; others_ref = [i for i in range(len(m["ref_boxes"])) if i != t]
            if n_p and rec["covered"][j]:
                kv = int(np.argmax(np.where(valid[j], iou[j], -1)))                # highest-IoU valid proposal (B)
                neg = [k for k in perm if not valid[j, k]]                          # IoU < 0.5, fixed per-image random order
                for nd in ND:
                    sel = neg if nd is None else neg[:nd]
                    rec["orc"][str(nd)][j] = 1.0 if not sel else float(cp[kv, j] > cp[sel, j].max())
                negD = [k for k in range(n_p) if not valid[j, k]][:len(others_ref)]  # detector-score order, as many as C
                cA, cB = cr[t, j], cp[kv, j]; cC = cr[others_ref, j] if others_ref else np.array([-np.inf]); cD = cp[negD, j] if negD else np.array([-np.inf])
                rec["c2x2"]["A+C"][j] = float(cA > cC.max()); rec["c2x2"]["B+C"][j] = float(cB > cC.max())
                rec["c2x2"]["A+D"][j] = float(cA > cD.max()); rec["c2x2"]["B+D"][j] = float(cB > cD.max())
                inval = ~valid[j]
                if inval.any():
                    rec["margin"][j] = float(cp[valid[j], j].max() - cp[inval, j].max())
            if n_p:
                p = int(cp[:, j].argmax())
                rec["etype"][j] = None if valid[j, p] else g["etype"][j][p]
        out.append(rec)
    return out


def macro(per, getter, cond=None):
    vals = []
    for r in per:
        x = getter(r)
        if cond is not None:
            x = x[cond(r)]
        x = x[~np.isnan(x)] if x.dtype.kind == "f" else x
        if len(x):
            vals.append(float(np.mean(x)))
    return float(np.mean(vals)) if vals else None


def quantities(per):
    q = {"acc_all_image_macro": macro(per, lambda r: r["all"]), "acc_all_query": float(np.mean(np.concatenate([r["all"] for r in per]))),
         "coverage_image_macro": macro(per, lambda r: r["covered"].astype(float)),
         "acc_covered_image_macro": macro(per, lambda r: r["all"], cond=lambda r: r["covered"])}
    for K in KS:
        q[f"topK_{K}_acc"] = macro(per, lambda r, K=K: r["topk"][str(K)]); q[f"topK_{K}_coverage"] = macro(per, lambda r, K=K: r["covk"][str(K)])
    for nd in ND:
        q[f"oracle_nd_{nd}_acc"] = macro(per, lambda r, nd=nd: r["orc"][str(nd)])
    for c in ("A+C", "B+C", "A+D", "B+D"):
        q[f"2x2_{c}"] = macro(per, lambda r, c=c: r["c2x2"][c])
    q["margin_mean"] = macro(per, lambda r: r["margin"])
    errs = [e for r in per for e in r["etype"] if e is not None]; n = len([e for r in per for e in r["etype"]])
    for t in ("other_same_cat", "other_diff_cat", "part_or_duplicate", "no_matching_annotation"):
        q[f"err_{t}_share_of_queries"] = float(sum(e == t for e in errs) / n) if n else None
    return q


def per_image_vector(per, key):
    """Per-image scalar for the image bootstrap (same definitions as `quantities`)."""
    f = {"acc_all": lambda r: np.mean(r["all"]), "B+D": lambda r: np.nanmean(r["c2x2"]["B+D"]) if np.isfinite(r["c2x2"]["B+D"]).any() else np.nan,
         "A+D": lambda r: np.nanmean(r["c2x2"]["A+D"]) if np.isfinite(r["c2x2"]["A+D"]).any() else np.nan,
         "B+C": lambda r: np.nanmean(r["c2x2"]["B+C"]) if np.isfinite(r["c2x2"]["B+C"]).any() else np.nan,
         "A+C": lambda r: np.nanmean(r["c2x2"]["A+C"]) if np.isfinite(r["c2x2"]["A+C"]).any() else np.nan,
         "oracle_all": lambda r: np.nanmean(r["orc"]["None"]) if np.isfinite(r["orc"]["None"]).any() else np.nan,
         "margin": lambda r: np.nanmean(r["margin"]) if np.isfinite(r["margin"]).any() else np.nan}[key]
    return np.array([f(r) for r in per], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["refcocog", "refcocoplus"]); ap.add_argument("--backbone", default="siglip2_b16"); a = ap.parse_args()
    res = {"protocol": "reports/R4/R4_VL0_PROTOCOL_20261011.md", "backbone": a.backbone, "datasets": {}}
    for ds in a.datasets:
        meta = torch.load(D / f"meta_{ds}.pt", weights_only=False); G = geometry(meta); R = {"n_images": len(meta), "n_queries": int(sum(len(m["target"]) for m in meta)),
                                                                                         "gates": {}, "runs": {}, "contrasts": {}}
        per = {}
        for o in OBJS:
            for s in SEEDS:
                f = D / "scores" / f"{ds}_{a.backbone}_{o}_s{s}.pt"
                if not f.exists():
                    continue
                x = torch.load(f, weights_only=False); R["gates"][x["tag"]] = x["gates"]
                if not (x["gates"]["G1"][2] and x["gates"]["G2"][2]):
                    continue
                per[(o, s)] = per_image_metrics(meta, G, x["scores"]); R["runs"][f"{o}_s{s}"] = quantities(per[(o, s)])
        rng = np.random.default_rng(0)
        for other in ("softmax", "js"):
            seeds = [s for s in SEEDS if ("vcs", s) in per and (other, s) in per]
            if not seeds:
                continue
            for key in ("acc_all", "A+C", "B+C", "A+D", "B+D", "oracle_all", "margin"):
                diffs = np.stack([per_image_vector(per[("vcs", s)], key) - per_image_vector(per[(other, s)], key) for s in seeds])   # [seeds, images]
                seed_means = np.nanmean(diffs, axis=1); img = np.nanmean(diffs, axis=0); img = img[np.isfinite(img)]
                boots = np.array([img[rng.integers(0, len(img), len(img))].mean() for _ in range(1000)])
                R["contrasts"][f"vcs_minus_{other}/{key}"] = {"mean": float(img.mean()), "image_boot_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                              "per_seed": [float(v) for v in seed_means], "n_images": int(len(img))}
        res["datasets"][ds] = R
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT / "R4_VL0_results.json", "w"), indent=1)
    L = ["# R4-VL0 — candidate diagnostics (exploratory; items 3–4 oracle-assisted)", ""]
    for ds, R in res["datasets"].items():
        bad = {k: v for k, v in R["gates"].items() if not (v["G1"][2] and v["G2"][2])}
        L += [f"## {ds} ({R['n_images']} images, {R['n_queries']} expressions); gates pass {len(R['gates']) - len(bad)} / {len(R['gates'])}", ""]
        keys = ["acc_all_image_macro", "coverage_image_macro", "acc_covered_image_macro"] + [f"topK_{K}_acc" for K in KS] + [f"oracle_nd_{n}_acc" for n in ND] + \
               [f"2x2_{c}" for c in ("A+C", "B+C", "A+D", "B+D")] + ["margin_mean"] + [f"err_{t}_share_of_queries" for t in ("other_same_cat", "other_diff_cat", "part_or_duplicate", "no_matching_annotation")]
        L += ["| quantity | " + " | ".join(OBJS) + " |", "|---|---|---|---|"]
        for k in keys:
            row = []
            for o in OBJS:
                v = [R["runs"][f"{o}_s{s}"][k] for s in SEEDS if f"{o}_s{s}" in R["runs"] and R["runs"][f"{o}_s{s}"][k] is not None]
                row.append(f"{100 * np.mean(v):.2f} ± {100 * np.std(v, ddof=1):.2f}" if len(v) > 1 else ("—" if not v else f"{100 * v[0]:.2f}"))
            L.append(f"| {k} | " + " | ".join(row) + " |")
        L += ["", "| contrast | mean (×100) | image bootstrap 95 % | per seed |", "|---|---|---|---|"]
        for k, v in R["contrasts"].items():
            L.append(f"| {k} | {100 * v['mean']:+.2f} | [{100 * v['image_boot_95'][0]:+.2f}, {100 * v['image_boot_95'][1]:+.2f}] | " + ", ".join(f"{100 * x:+.2f}" for x in v["per_seed"]) + " |")
        L.append("")
    (OUT / "R4_VL0_tables.md").write_text("\n".join(L)); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
