"""r7 R7-20 D2 — factual-support fixtures (r7 package directions/D2_factual_support).
  pope : host LLaVA-1.5-7B on the fixed POPE-adversarial subset (50 images × 6 questions, shared with D1), with the image and as a text-only
         control (same prompt, no image token); official yes/no parsing (llava/eval/eval_pope.py); P(yes) AUROC; COCO-image overlap with
         M-HalDetect.  GPU, env /projects/EEG-foundation-model/yinghao/envs/sparsevlm.
  text : M-HalDetect (hendryx-scale/mhal-detect 6ef75f5, CC BY-NC 4.0) segment loader grouped by source COCO image; schema audit; pure-text
         shortcut check inside the official train split only (5 image-grouped folds): hashed n-gram logistic, position-only and length-only
         classifiers for INACCURATE vs ACCURATE (ANALYSIS / UNSURE kept out of the binary task).  CPU, main env.
    python scripts/r7_fixtures_d2.py pope --out OUT
    python scripts/r7_fixtures_d2.py text --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

MHAL = Path("/projects/EEG-foundation-model/yinghao/data/mhal_detect/repo")
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------------------------------------------------------------------------------ M-HalDetect
def coco_id(name: str) -> int:
    return int(re.search(r"(\d{6,})", name).group(1))


def load_mhal(split: str):
    """Segments of the official split with their source COCO-val2014 image (the grouping unit)."""
    entries = json.loads((MHAL / f"{split}_raw.json").read_text()); segs = []
    for ei, e in enumerate(entries):
        img, resp = coco_id(e["image"]), e["response"]
        for si, s in enumerate(e["annotations"]):
            segs.append({"split": split, "entry": ei, "seg": si, "image": img, "label": s["label"], "text": s["text"], "start": s["start"],
                         "end": s["end"], "rel_pos": s["start"] / max(len(resp), 1), "n_segs": len(e["annotations"]), "resp_len": len(resp),
                         "span_matches_text": resp[s["start"]:s["end"]] == s["text"]})
    return entries, segs


def fold_of(group, k=5):
    return int(hashlib.sha256(f"r7-20/mhal-fold/{group}".encode()).hexdigest()[:8], 16) % k


def grouped_folds(groups, k=5):
    """Image-grouped folds; asserts no image crosses a FIT / CAL boundary."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r7" / "package_r7_20261011" / "reference")); import vcs_vl_reference as REF  # noqa: E402
    f = np.array([fold_of(g, k) for g in groups])
    for i in range(k):
        REF.assert_group_disjoint({"FIT": [str(g) for g, x in zip(groups, f) if x != i], "CAL": [str(g) for g, x in zip(groups, f) if x == i]})
    return f


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int); order = np.argsort(score, kind="mergesort"); r = np.empty(len(score)); s = score[order]
    i = 0
    while i < len(s):                                                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def ngram_matrix(texts, buckets=2 ** 18):
    from scipy.sparse import csr_matrix
    rows, cols = [], []
    for i, t in enumerate(texts):
        w = re.findall(r"[a-z0-9']+", t.lower()); feats = set(w) | {a + " " + b for a, b in zip(w, w[1:])}
        for f in feats:
            rows.append(i); cols.append(zlib.crc32(f.encode()) % buckets)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), buckets))


def logreg(X, y, l2=1e-4):
    """Class-balanced L2 logistic regression (L-BFGS); returns a scorer."""
    from scipy.optimize import minimize
    y = np.asarray(y, float); wts = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9)) / len(y)
    d = X.shape[1]

    def fg(w):
        z = X @ w[:d] + w[d]; p = 1 / (1 + np.exp(-z)); loss = np.sum(wts * (np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)))
        g = wts * (p - y); return loss + 0.5 * l2 * w[:d] @ w[:d], np.concatenate([X.T @ g + l2 * w[:d], [g.sum()]])
    w = minimize(fg, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return (lambda Z: Z @ w[:d] + w[d]), w


def shortcut_cv(segs, positive, negative, k=5):
    keep = [s for s in segs if s["label"] in positive + negative]; y = np.array([s["label"] in positive for s in keep], int)
    folds = grouped_folds([s["image"] for s in keep], k)
    feats = {"ngram_logistic": ngram_matrix([s["text"] for s in keep]),
             "position_only": np.array([[s["rel_pos"], s["seg"], s["n_segs"], np.log1p(s["resp_len"]), np.log1p(len(s["text"]))] for s in keep]),
             "length_only": np.array([[np.log1p(len(s["text"]))] for s in keep])}
    res = {"n": len(keep), "positive_rate": float(y.mean()), "images": len({s["image"] for s in keep}), "folds": k}
    for name, X in feats.items():
        if not hasattr(X, "tocsr"):
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        oof = np.zeros(len(y)); fold_auc = []
        for i in range(k):
            tr, te = folds != i, folds == i
            f, _ = logreg(X[tr], y[tr]); oof[te] = f(X[te]); fold_auc.append(auroc(oof[te], y[te]))
        pred = (oof > 0).astype(int)
        res[name] = {"auroc_oof": auroc(oof, y), "auroc_fold_mean": float(np.mean(fold_auc)), "auroc_fold_sd": float(np.std(fold_auc)),
                     "balanced_accuracy": float(0.5 * (np.mean(pred[y == 1] == 1) + np.mean(pred[y == 0] == 0)))}
    f, w = logreg(feats["ngram_logistic"], y)                                          # cue inspection only (fit on all train items)
    vocab = {}
    for s in keep:
        ws = re.findall(r"[a-z0-9']+", s["text"].lower())
        for g in set(ws) | {a + " " + b for a, b in zip(ws, ws[1:])}:
            vocab.setdefault(zlib.crc32(g.encode()) % 2 ** 18, g)
    top = np.argsort(-w[:-1])[:15]; bot = np.argsort(w[:-1])[:15]
    res["top_cues_positive"] = [vocab.get(int(i), "?") for i in top]; res["top_cues_negative"] = [vocab.get(int(i), "?") for i in bot]
    return res


def cmd_text(a):
    out = Path(a.out) / "d2"; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tr_e, tr = load_mhal("train"); va_e, va = load_mhal("val")
    def audit(entries, segs):
        labs = {}
        for s in segs:
            labs[s["label"]] = labs.get(s["label"], 0) + 1
        per_img = {}
        for e in entries:
            per_img[coco_id(e["image"])] = per_img.get(coco_id(e["image"]), 0) + 1
        return {"entries": len(entries), "segments": len(segs), "images": len(per_img), "labels": labs,
                "responses_per_image_max": max(per_img.values()), "responses_per_image_mean": float(np.mean(list(per_img.values()))),
                "distinct_questions": len({e["question"] for e in entries}), "span_text_consistent": float(np.mean([s["span_matches_text"] for s in segs])),
                "span_coverage_of_response": float(np.mean([sum(x["end"] - x["start"] for x in e["annotations"]) / max(len(e["response"]), 1) for e in entries]))}
    tri, vai = {s["image"] for s in tr}, {s["image"] for s in va}
    res = {"source": "hendryx-scale/mhal-detect 6ef75f5 (annotations only; images are COCO val2014, not downloaded here)",
           "schema": {"train": audit(tr_e, tr), "val": audit(va_e, va)},
           "official_split_image_overlap": {"train_images": len(tri), "val_images": len(vai), "shared": len(tri & vai)},
           "shortcut_train_only": {"inaccurate_vs_accurate": shortcut_cv(tr, ["INACCURATE"], ["ACCURATE"]),
                                   "analysis_vs_factual": shortcut_cv(tr, ["ANALYSIS"], ["ACCURATE", "INACCURATE"])},
           "seconds": None}
    pope = Path(a.out) / "fixture" / "pope.jsonl"
    if pope.exists():
        pi = {coco_id(json.loads(x)["image_id"]) for x in pope.read_text().splitlines() if x.strip()}
        res["pope_fixture_overlap"] = {"pope_fixture_images": len(pi), "in_mhal_train": len(pi & tri), "in_mhal_val": len(pi & vai)}
    res["seconds"] = time.time() - t0
    (out / "mhal_summary.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ POPE
def cmd_pope(a, host_factory=None):
    import r7_vl_common as V
    root = Path(a.out); out = root / "d2"; out.mkdir(parents=True, exist_ok=True); V.write_fixtures(root / "fixture", ("pope",))
    items = V.load_fixture(root / "fixture", "pope"); pred_path, run_path = out / "pope_preds.jsonl", out / "pope_run.json"
    done = {(r["condition"], r["qid"]): r for r in V.read_jsonl(pred_path)}               # resume: finished answers are kept
    if pred_path.exists():
        pred_path.write_text("".join(V.jdump(r) + "\n" for r in done.values()))
    info = json.loads(run_path.read_text()) if run_path.exists() else {"items": len(items), "category": items[0]["category"]}
    todo = {c: [it for it in items if (c, it["qid"]) not in done] for c in ("image", "text_only")}
    if any(todo.values()):
        host = (host_factory or (lambda: V.Host(root, sparse=False)))(); torch = host.torch
        info.update({"load_seconds": host.load_seconds, "stage_notes": getattr(host, "stage_notes", []), "env": V.env_info()})
        for it in items[:3]:
            host.answer(it["prompt"], it["image_file"])
        V.cuda_peak_reset(torch)
        with open(pred_path, "a") as fh:
            for cond in ("image", "text_only"):
                if not todo[cond]:                                                     # finished in an earlier session: keep its timing
                    continue
                t0 = time.time()
                for it in todo[cond]:
                    r = host.answer(it["prompt"], it["image_file"] if cond == "image" else None)
                    row = {"condition": cond, "qid": it["qid"], "image_id": it["image_id"], "label": it["label"], **r}
                    fh.write(V.jdump(row) + "\n"); fh.flush(); done[(cond, it["qid"])] = row
                info[f"wall_seconds_{cond}"] = time.time() - t0; info[f"items_timed_{cond}"] = len(todo[cond])
                run_path.write_text(V.jdump(info, indent=1))
        info["peak_mem_gb"] = V.cuda_peak_gb(torch); run_path.write_text(V.jdump(info, indent=1))
    rows = [done[(c, it["qid"])] for c in ("image", "text_only") for it in items]
    res = {"run": info, "conditions": {}}
    for cond in ("image", "text_only"):
        rr = [r for r in rows if r["condition"] == cond]; y = [r["label"] for r in rr]
        m = V.pope_metrics([V.pope_yes(r["text"]) for r in rr], y)
        py = [r["p_yes"] / max(r["p_yes"] + r["p_no"], 1e-12) for r in rr]
        acc = np.array([float(V.pope_yes(r["text"]) == r["label"]) for r in rr])
        res["conditions"][cond] = {**{k: (100 * v if isinstance(v, float) else v) for k, v in m.items()}, "auroc_p_yes": auroc(py, y),
                                   "accuracy_ci95_image_bootstrap": [100 * x for x in V.cluster_bootstrap(acc, [r["image_id"] for r in rr])],
                                   "timing_ms_mean": {k: float(np.mean([r[k] for r in rr])) for k in ("vision_ms", "prefill_ms", "decode_ms", "wall_ms")}}
    d1 = root / "d1" / "pred_full.jsonl"
    if (root / "d1" / "run_full.json").exists():                                        # same host, same items, separate process (completed run only)
        prev = {r["qid"]: r["text"] for r in map(json.loads, d1.read_text().splitlines()) if r["fixture"] == "pope"}
        img = [r for r in rows if r["condition"] == "image" and r["qid"] in prev]
        res["identical_to_d1_full_run"] = {"n": len(img), "fraction": float(np.mean([prev[r["qid"]] == r["text"] for r in img])) if img else None}
    mh = {coco_id(e["image"]) for split in ("train", "val") for e in json.loads((MHAL / f"{split}_raw.json").read_text())}
    pi = {coco_id(it["image_id"]) for it in items}
    res["mhal_image_overlap"] = {"pope_fixture_images": len(pi), "also_in_mhal_detect": len(pi & mh)}
    (out / "pope_summary.json").write_text(V.jdump(res, indent=1)); print(V.jdump(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pope", "text"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"pope": cmd_pope, "text": cmd_text}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
