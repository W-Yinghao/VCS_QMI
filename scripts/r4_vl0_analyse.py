"""R4-VL0 diagnostics (reports/R4/R4_VL0_PROTOCOL_20261011.md §Diagnostics 1–7) from the saved score matrices of scripts/r4_vl0_scores.py.
Exploratory and descriptive; items 3 and 4 use ground truth (oracle-assisted) and are not deployable procedures.  Every subset, K and permutation is
shared by all methods.  One definition per quantity: per-query arrays -> per-image means -> image-macro (tables) and paired image-level contrasts
(VCS − softmax, VCS − JS: per-seed differences of the per-image means, seed-averaged, 1 000-resample image bootstrap with a per-quantity seed) for
EVERY quantity.  The oracle distractor curve is reported as frozen (queries with fewer than n_d distractors are scored against all they have —
`oracle_nd_<n>_acc`) together with the fraction of covered queries that have >= n_d distractors (`..._frac_full`) and the curve restricted to those
queries (`..._acc_strict`).  Results are merged into the existing reports/R4/R4_VL0_results.json (datasets not processed are kept); missing score
files are listed, never silently skipped.
    python scripts/r4_vl0_analyse.py [--datasets refcocog refcocoplus] [--backbone siglip2_b16]
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vl1_22_tableC import iou_xyxy  # noqa: E402
from r4_vl0_scores import OUT as SCORES_ROOT, score_path  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "R4"
OBJS, SEEDS, KS, ND = ("vcs", "js", "softmax"), (0, 1, 2), (4, 8, 16, 32, None), (1, 3, 7, 15, 31, None)
ETYPES = ("other_same_cat", "other_diff_cat", "part_or_duplicate", "no_matching_annotation")


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
        iou_ann = np.stack([iou_xyxy(a, P) for a in allann]); iou = np.stack([iou_ann[t] for t in tg]); et = []
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


def check_alignment(meta, sc):
    if len(sc) != len(meta):
        raise ValueError(f"score list ({len(sc)}) and meta ({len(meta)}) differ in length")
    for i, (m, s) in enumerate(zip(meta, sc)):
        if tuple(s["cos_prop"].shape) != (len(m["prop_boxes"]), len(m["target"])) or s["cos_ref"].shape != (len(m["ref_boxes"]), len(m["target"])):
            raise ValueError(f"image {i} ({m['image_id']}): score shapes {tuple(s['cos_prop'].shape)} / {tuple(s['cos_ref'].shape)} do not match meta "
                             f"({len(m['prop_boxes'])} proposals, {len(m['ref_boxes'])} referred, {len(m['target'])} queries)")


def per_query(meta, G, sc):
    """Per image: dict quantity -> per-query array (nan where undefined)."""
    out = []
    for m, g, s in zip(meta, G, sc):
        cp = s["cos_prop"].numpy(); cr = s["cos_ref"].numpy(); iou = g["iou"]; nq = len(m["target"]); n_p = cp.shape[0]; nan = np.full(nq, np.nan)
        valid = iou >= 0.5 if n_p else np.zeros((nq, 0), bool); covered = valid.any(1) if n_p else np.zeros(nq, bool)
        q = {"coverage": covered.astype(float)}
        for K in KS:
            k = n_p if K is None else min(K, n_p)
            if k == 0:
                q[f"topK_{K}_acc"] = np.zeros(nq); q[f"topK_{K}_coverage"] = np.zeros(nq)
            else:
                pred = cp[:k].argmax(0); q[f"topK_{K}_acc"] = valid[np.arange(nq), pred].astype(float); q[f"topK_{K}_coverage"] = valid[:, :k].any(1).astype(float)
        q["acc_all"] = q["topK_None_acc"]; q["acc_covered"] = np.where(covered, q["acc_all"], np.nan)
        for nd in ND:
            q[f"oracle_nd_{nd}_acc"] = nan.copy(); q[f"oracle_nd_{nd}_acc_strict"] = nan.copy(); q[f"oracle_nd_{nd}_frac_full"] = nan.copy()
        for c in ("A+C", "B+C", "A+D", "B+D"):
            q[f"2x2_{c}"] = nan.copy()
        q["margin"] = nan.copy()
        for t in ETYPES:
            q[f"err_{t}"] = np.zeros(nq)
        perm = np.random.default_rng(int(m["image_id"])).permutation(n_p)
        for j in range(nq):
            t = m["target"][j]; others_ref = [i for i in range(len(m["ref_boxes"])) if i != t]
            if n_p and covered[j]:
                kv = int(np.argmax(np.where(valid[j], iou[j], -1)))                     # highest-IoU valid proposal (B)
                neg = [k for k in perm if not valid[j, k]]                              # IoU < 0.5 to the target, fixed per-image random order
                for nd in ND:
                    sel = neg if nd is None else neg[:nd]; v = 1.0 if not sel else float(cp[kv, j] > cp[sel, j].max())
                    q[f"oracle_nd_{nd}_acc"][j] = v; full = nd is None or len(neg) >= nd
                    q[f"oracle_nd_{nd}_frac_full"][j] = float(full); q[f"oracle_nd_{nd}_acc_strict"][j] = v if full else np.nan
                negD = [k for k in range(n_p) if not valid[j, k]][:len(others_ref)]      # detector-score order, as many as C
                cA, cB = cr[t, j], cp[kv, j]; cC = cr[others_ref, j] if others_ref else np.array([-np.inf]); cD = cp[negD, j] if negD else np.array([-np.inf])
                q["2x2_A+C"][j] = float(cA > cC.max()); q["2x2_B+C"][j] = float(cB > cC.max()); q["2x2_A+D"][j] = float(cA > cD.max()); q["2x2_B+D"][j] = float(cB > cD.max())
                inval = ~valid[j]
                if inval.any():
                    q["margin"][j] = float(cp[valid[j], j].max() - cp[inval, j].max())
            if n_p:
                p = int(cp[:, j].argmax())
                if not valid[j, p]:
                    q[f"err_{g['etype'][j][p]}"][j] = 1.0
        out.append(q)
    return out


def per_image(per, key):
    """Per-image mean of a per-query quantity (nan when no query defines it)."""
    return np.array([np.nanmean(r[key]) if np.isfinite(r[key]).any() else np.nan for r in per], dtype=float)


def macro(per, key):
    x = per_image(per, key); x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def quantities(per):
    keys = ["acc_all", "coverage", "acc_covered"] + [f"topK_{K}_{s}" for K in KS for s in ("acc", "coverage")] + \
           [f"oracle_nd_{nd}_{s}" for nd in ND for s in ("acc", "acc_strict", "frac_full")] + [f"2x2_{c}" for c in ("A+C", "B+C", "A+D", "B+D")] + ["margin"] + [f"err_{t}" for t in ETYPES]
    out = {k + ("_image_macro" if k in ("acc_all", "coverage", "acc_covered") else ""): macro(per, k) for k in keys}
    out["acc_all_query"] = float(np.mean(np.concatenate([r["acc_all"] for r in per]))); out["_keys"] = keys
    return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["refcocog", "refcocoplus"]); ap.add_argument("--backbone", default="siglip2_b16"); a = ap.parse_args()
    rp = OUT / "R4_VL0_results.json"; res = json.load(open(rp)) if rp.exists() else {"datasets": {}}
    res.update({"protocol": "reports/R4/R4_VL0_PROTOCOL_20261011.md", "backbone": a.backbone})
    for ds in a.datasets:
        meta = torch.load(SCORES_ROOT / f"meta_{ds}.pt", weights_only=False); G = geometry(meta)
        R = {"n_images": len(meta), "n_queries": int(sum(len(m["target"]) for m in meta)), "gates": {}, "missing": [], "runs": {}, "n_seeds": {}, "contrasts": {}}
        per = {}
        for o in OBJS:
            for s in SEEDS:
                f = score_path(ds, a.backbone, o, s)
                if not f.exists():
                    R["missing"].append(f.name); print("MISSING", f.name, flush=True); continue
                x = torch.load(f, weights_only=False); R["gates"][x["tag"]] = x["gates"]
                if not (x["gates"]["G1"][2] and x["gates"]["G2"][2]):
                    print("GATE FAIL (excluded)", x["tag"], flush=True); continue
                check_alignment(meta, x["scores"]); per[(o, s)] = per_query(meta, G, x["scores"]); R["runs"][f"{o}_s{s}"] = quantities(per[(o, s)])
            R["n_seeds"][o] = sum(1 for s in SEEDS if (o, s) in per)
        keys = next(iter(R["runs"].values()))["_keys"] if R["runs"] else []
        for other in ("softmax", "js"):
            seeds = [s for s in SEEDS if ("vcs", s) in per and (other, s) in per]
            if not seeds:
                continue
            for key in keys:
                diffs = np.stack([per_image(per[("vcs", s)], key) - per_image(per[(other, s)], key) for s in seeds])         # [seeds, images]
                img = np.nanmean(diffs, axis=0); img = img[np.isfinite(img)]
                if len(img) == 0:
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{ds}/{other}/{key}".encode()))
                boots = np.array([img[rng.integers(0, len(img), len(img))].mean() for _ in range(1000)])
                R["contrasts"][f"vcs_minus_{other}/{key}"] = {"mean": float(img.mean()), "image_boot_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                              "per_seed": [float(np.nanmean(d)) for d in diffs], "n_images": int(len(img)), "n_seeds": len(seeds)}
        for v in R["runs"].values():
            v.pop("_keys", None)
        res["datasets"][ds] = R
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(rp, "w"), indent=1)
    L = ["# R4-VL0 — candidate diagnostics (exploratory; items 3–4 oracle-assisted)", "",
         "`oracle_nd_<n>_acc` is the frozen reading (queries with fewer than n distractors are scored against all they have); `_frac_full` = fraction of covered queries with >= n "
         "distractors; `_acc_strict` = restricted to those queries.  Contrasts: VCS − other, per-seed differences of per-image means, image bootstrap 95 %.", ""]
    for ds, R in res["datasets"].items():
        bad = {k: v for k, v in R["gates"].items() if not (v["G1"][2] and v["G2"][2])}
        L += [f"## {ds} ({R['n_images']} images, {R['n_queries']} expressions); gates pass {len(R['gates']) - len(bad)} / {len(R['gates'])}; missing score files: {R['missing'] or 'none'}", ""]
        hdr = [f"{o} (n={R['n_seeds'].get(o, 0)})" for o in OBJS]; L += ["| quantity | " + " | ".join(hdr) + " |", "|---|---|---|---|"]
        first = next(iter(R["runs"].values()), {})
        for k in first:
            row = []
            for o in OBJS:
                v = [R["runs"][f"{o}_s{s}"][k] for s in SEEDS if f"{o}_s{s}" in R["runs"] and R["runs"][f"{o}_s{s}"].get(k) is not None]
                row.append(f"{100 * np.mean(v):.2f} ± {100 * np.std(v, ddof=1):.2f}" if len(v) > 1 else ("—" if not v else f"{100 * v[0]:.2f}"))
            L.append(f"| {k} | " + " | ".join(row) + " |")
        L += ["", "| contrast | mean (×100) | image bootstrap 95 % | per seed |", "|---|---|---|---|"]
        for k, v in R["contrasts"].items():
            L.append(f"| {k} | {100 * v['mean']:+.2f} | [{100 * v['image_boot_95'][0]:+.2f}, {100 * v['image_boot_95'][1]:+.2f}] | " + ", ".join(f"{100 * x:+.2f}" for x in v["per_seed"]) + " |")
        L.append("")
    (OUT / "R4_VL0_tables.md").write_text("\n".join(L)); print("\n".join(L[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
