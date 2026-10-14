"""V7X-40 secondary system table on EvalMuse DEV (descriptive, unadjusted; V7X_30): every system's image score against the human overall score
(mean of the 1-5 votes): average-rank Spearman (SRCC), Pearson (PLCC), within-prompt pairwise accuracy (pairs with equal human means excluded; tied
scores 0.5); prompt-cluster bootstrap 5,000 (seed 2718) for SRCC and pairwise accuracy.  Element level (systems with element outputs) on the
COMMON element set (element ids every listed system scores): natural Brier of the element probability (prompt -> image -> element macro),
prompt-macro AUROC, majority accuracy.  B0 (released FGA-BLIP2) was trained on all of EvalMuse TRAIN, which contains DEV: its rows are exposed.
DEV labels are read here only.  Score files that do not exist yet are skipped and listed.
    python -I scripts/v7x/x40_dev_external.py --out F"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v7"))
import v7_common as VC  # noqa: E402
import v7_score as VS  # noqa: E402

B = VC.BIG_OUT; X = B / "x30_dev"; REPS, SEED = 5000, 2718


def jl(p):
    return [r for r in VC.read_jsonl(p) if "header" not in r]


def pairwise(pids, s, h):
    g = defaultdict(list)
    for q, a, b in zip(pids, s, h):
        g[q].append((a, b))
    per = {}
    for q, rows in g.items():
        num = den = 0.0
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if rows[i][1] == rows[j][1]:
                    continue
                d = (rows[i][0] - rows[j][0]) * (rows[i][1] - rows[j][1]); num += 1.0 if d > 0 else 0.5 if d == 0 else 0.0; den += 1
        if den:
            per[q] = (num, den)
    return per


def overall_metrics(name, score, imgs):
    ids = [i for i in imgs if i in score]; pids = [imgs[i]["prompt_id"] for i in ids]; s = np.array([score[i] for i in ids], float); h = np.array([imgs[i]["overall_mean"] for i in ids], float)
    per = pairwise(pids, s, h); P = sorted(per); rng = np.random.default_rng(SEED); byq = defaultdict(list)
    for k, q in enumerate(pids):
        byq[q].append(k)
    qs = sorted(byq); boot_s, boot_p = [], []
    for _ in range(REPS):
        pick = rng.integers(0, len(qs), len(qs)); idx = np.concatenate([byq[qs[j]] for j in pick]); boot_s.append(VC.avg_rank_spearman(s[idx], h[idx]))
        num = sum(per[qs[j]][0] for j in pick if qs[j] in per); den = sum(per[qs[j]][1] for j in pick if qs[j] in per); boot_p.append(num / den if den else np.nan)
    tot = (sum(per[q][0] for q in P) / sum(per[q][1] for q in P)) if P else None
    return {"system": name, "n_images": len(ids), "n_prompts": len(set(pids)), "srcc": VC.avg_rank_spearman(s, h), "srcc_ci95": [float(np.nanquantile(boot_s, 0.025)), float(np.nanquantile(boot_s, 0.975))],
            "plcc": float(np.corrcoef(s, h)[0, 1]), "pairwise_acc": tot, "pairwise_ci95": [float(np.nanquantile(boot_p, 0.025)), float(np.nanquantile(boot_p, 0.975))]}


def element_metrics(name, prob, els, common):
    ids = sorted(common); p = np.array([prob[e] for e in ids]); y = np.array([els[e]["y"] for e in ids]); pids = [els[e]["prompt_id"] for e in ids]; iids = [els[e]["image_id"] for e in ids]
    br = VC.expected_brier(p, y); maj = y != 0.5
    return {"system": name, "n_elements": len(ids), "natural_brier_macro": VC.hierarchical_mean(pids, iids, br)[0], "auroc_prompt_macro": VS.auroc_macro(pids, p, y)[0],
            "accuracy_majority": float(((p[maj] >= 0.5) == (y[maj] > 0.5)).mean()) if maj.any() else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists: never overwritten")
    imgs = {i["image_id"]: i for i in VC.read_jsonl(VC.DATA_OUT / "manifest_images.jsonl") if i["role"] == "DEV"}
    els = {e["element_id"]: {**e, "prompt_id": imgs[e["image_id"]]["prompt_id"]} for e in VC.read_jsonl(VC.DATA_OUT / "manifest_elements.jsonl")
           if e["image_id"] in imgs and e["scope"] == "main" and e.get("y") is not None}
    overall, element, missing = {}, {}, []
    def native(tag, path):
        if not Path(path).exists():
            missing.append(path); return
        d = json.load(open(path)); overall[tag] = {r["img_path"]: float(r["score_result"]) for r in d}
        element[tag] = {f"{r['img_path']}::{k}": float(v) for r in d for k, v in r["element_result"].items()}
    native("B0_FGA_released_O (exposed)", B / "native_eval/B0_released/native_eval_DEV.json")
    for s in (0, 1, 2):
        native(f"N0_s{s}_O", B / f"units/V7-FIT-N0_native_l1-s{s}/native_eval_DEV.json")
    for u in sorted(glob.glob(str(B / "units/V7-FIT-M*"))):
        n = Path(u).name.replace("V7-FIT-", ""); ims = VC.read_jsonl(Path(u) / "pred/images_DEV.jsonl"); overall[n + "_O"] = {r["image_id"]: r["O"] for r in ims}
        ev = VC.read_jsonl(Path(u) / "pred/elements_DEV.jsonl"); per = defaultdict(list)
        for e in ev:
            per[e["image_id"]].append(e["p_prior"])
        overall[n + "_E"] = {k: float(np.mean(v)) for k, v in per.items()}; element[n] = {e["element_id"]: e["p_prior"] for e in ev}
    for tag, f in (("B1_VQAScore", "b1_vqascore_clipflant5xxl.jsonl"), ("B2_VQAScore_Qwen25VL7B", "b2_vqascore_qwen25vl7b.jsonl"), ("B5_HPSv3", "b5_hpsv3.jsonl")):
        if (X / f).exists():
            overall[tag] = {r["image_id"]: r["score"] for r in jl(X / f)}
        else:
            missing.append(str(X / f))
    for tag, pat in (("B4T_TIFA", "tifa_scores_*.jsonl"), ("B4D_DSG", "dsg_scores_*.jsonl"), ("B3_PNVQA_own", "pnvqa_scores_own_*.jsonl"), ("B3_PNVQA_evalmuse_elements", "pnvqa_scores_evalmuse_*.jsonl")):
        fs = sorted(glob.glob(str(X / pat)))
        if not fs:
            missing.append(str(X / pat)); continue
        rows = [r for f in fs for r in jl(f)]; overall[tag] = {r["image_id"]: r["score"] for r in rows}
        if tag == "B3_PNVQA_evalmuse_elements":
            element[tag] = {f"{r['image_id']}::{e['element']}": e["S_e"] for r in rows for e in r.get("elements", [])}
    common = set(els)
    for tag, m in element.items():
        common &= set(m)
    res = {"overall": [overall_metrics(k, v, imgs) for k, v in overall.items()], "element_common_set": [element_metrics(k, v, els, common) for k, v in element.items()] if common else [],
           "n_common_elements": len(common), "missing": missing, "note": "descriptive, unadjusted; B0 exposed (trained on all EvalMuse TRAIN)"}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r in res["overall"]:
        print(f"{r['system']:40s} SRCC {r['srcc']:.4f} PLCC {r['plcc']:.4f} pairwise {r['pairwise_acc']:.4f} (n {r['n_images']})")
    print("missing:", missing)


if __name__ == "__main__":
    main()
