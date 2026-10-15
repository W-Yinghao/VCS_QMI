"""V7C: the frozen V7-03 secondary readouts, fixed before any GenAI-Rank label is read (V7-03 readout.secondary; closeout package docs/04 §1:
alpha, parser and calibrator are not changed after terminal results).

Per formal M unit (M0 / M1 / M2 x seeds 0-2):
  H_alpha = alpha (O - 1) / 4 + (1 - alpha) E, alpha in {0, .25, .5, .75, 1} chosen on CAL_SELECT by prompt-macro within-prompt pairwise accuracy
            against the overall human score over multi-image prompts (E = the human elements' mean prior-corrected probability on EvalMuse);
            ties -> the lower alpha (closer to the primary E).  Deployment (external prompts): E over the parser's elements; E undefined
            (decomposition_failed) -> the frozen O fallback, H = (O - 1) / 4.
  E_cal   = mean over the elements of sigmoid(a z + b), Platt (a, b) fitted on CAL_PROB elements by soft log loss with the natural
            prompt -> image -> element weights; E undefined -> the same O fallback ((O - 1) / 4 keeps the O order).
Outputs: <out>/<unit>/{alpha.json, genai_rank_H.jsonl, genai_rank_Ecal.jsonl, <role>_readouts.jsonl}.  Reads CAL_SELECT / CAL_PROB / DEV labels
(training-side roles) only; no GenAI label.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BIG = Path("/projects/EEG-foundation-model/yinghao/outputs_v7")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
UNITS = [f"V7-FIT-{a}-s{s}" for a in ("M0_semantic_l1", "M1_semantic_logistic", "M2_semantic_vcs") for s in (0, 1, 2)]


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pairwise_acc_macro(score: dict, human: dict, prompt_of: dict) -> float:
    by = defaultdict(list)
    for i in score:
        by[prompt_of[i]].append(i)
    accs = []
    for ids in by.values():
        if len(ids) < 2:
            continue
        s = np.array([score[i] for i in ids]); h = np.array([human[i] for i in ids]); a, b = np.triu_indices(len(ids), 1); dh = h[a] - h[b]; ok = dh != 0
        if ok.any():
            ds = s[a] - s[b]; accs.append(float(np.where(ds[ok] == 0, 0.5, (np.sign(ds[ok]) == np.sign(dh[ok])).astype(float)).mean()))
    return float(np.mean(accs))


def platt(z, y, w, iters=200):
    """Weighted soft-label logistic fit p = sigmoid(a z + b) by Newton steps."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b))); g = np.array([np.sum(w * (p - y) * z), np.sum(w * (p - y))]); s = w * p * (1 - p)
        H = np.array([[np.sum(s * z * z), np.sum(s * z)], [np.sum(s * z), np.sum(s)]]) + 1e-12 * np.eye(2); step = np.linalg.solve(H, g); a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def natural_weights(els):
    per_p = defaultdict(set); per_i = defaultdict(int)
    for e in els:
        per_p[e["prompt_id"]].add(e["image_id"]); per_i[e["image_id"]] += 1
    n_p = len(per_p); return np.array([1.0 / (n_p * len(per_p[e["prompt_id"]]) * per_i[e["image_id"]]) for e in els])


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); out = Path(a.out); summary = {}
    for u in UNITS:
        U = BIG / "units" / u / "pred"; d = out / u; d.mkdir(parents=True, exist_ok=True)
        sel_el, sel_im = jl(U / "elements_CAL_SELECT.jsonl"), jl(U / "images_CAL_SELECT.jsonl")
        egt = defaultdict(list)
        for e in sel_el:
            egt[e["image_id"]].append(e["p_prior"])
        O = {r["image_id"]: r["O"] for r in sel_im}; Hh = {r["image_id"]: r["overall"] for r in sel_im}; P = {r["image_id"]: r["prompt_id"] for r in sel_im}
        ids = [i for i in O if egt.get(i)]; E = {i: float(np.mean(egt[i])) for i in ids}
        accs = {al: pairwise_acc_macro({i: al * (O[i] - 1) / 4 + (1 - al) * E[i] for i in ids}, Hh, P) for al in ALPHAS}
        best = max(accs.values()); alpha = min(al for al, v in accs.items() if v == best)
        cp = jl(U / "elements_CAL_PROB.jsonl"); z = np.array([e["z"] for e in cp]); y = np.array([e["y"] for e in cp]); w = natural_weights(cp); pa, pb = platt(z, y, w)
        rec = {"unit": u, "alpha": alpha, "alpha_selection": {str(k): v for k, v in accs.items()}, "alpha_rule": "max CAL_SELECT prompt-macro within-prompt pairwise accuracy; ties -> lower alpha",
               "n_cal_select_images_with_elements": len(ids), "platt": {"a": pa, "b": pb, "fit": "CAL_PROB elements, soft log loss, natural prompt -> image -> element weights"}}
        (d / "alpha.json").write_text(json.dumps(rec, indent=1)); summary[u] = {"alpha": alpha, "platt": [pa, pb]}
        for name, fn in (("H", lambda r: alpha * (r["O"] - 1) / 4 + (1 - alpha) * r["E"]),
                         ("Ecal", lambda r: float(np.mean(1 / (1 + np.exp(-(pa * np.array(r["element_z"]) + pb))))))):
            with open(d / f"genai_rank_{name}.jsonl", "w") as fh:
                for r in jl(U / "genai_rank.jsonl"):
                    v = (r["O"] - 1) / 4 if (r["E"] is None or r.get("decomposition_failed")) else fn(r)
                    fh.write(json.dumps({"prompt_id": r["prompt_id"], "generator_id": r["generator_id"], "image_id": r["image_id"], "candidate_index": r["candidate_index"],
                                         "score": float(v), "O_fallback": bool(r["E"] is None or r.get("decomposition_failed"))}) + "\n")
        for role in ("CAL_SELECT", "CAL_PROB", "DEV"):                                     # per-image E_gt / E_auto / O / H / E_cal for V7C-01 / 02
            el = jl(U / f"elements_{role}.jsonl"); im = jl(U / f"images_{role}.jsonl"); au = {r["image_id"]: r for r in jl(U / f"evalmuse_auto_{role}.jsonl")}
            g = defaultdict(list); gz = defaultdict(list); gy = defaultdict(list)
            for e in el:
                g[e["image_id"]].append(e["p_prior"]); gz[e["image_id"]].append(e["z"]); gy[e["image_id"]].append(e["y"])
            with open(d / f"{role}_readouts.jsonl", "w") as fh:
                for r in im:
                    i = r["image_id"]; e_gt = float(np.mean(g[i])) if g.get(i) else None; ar = au.get(i, {})
                    fh.write(json.dumps({"image_id": i, "prompt_id": r["prompt_id"], "role": role, "unit": u, "Hbar": r["overall"], "O": r["O"],
                                         "E_gt": e_gt, "E_auto": ar.get("E"), "parser_status": ar.get("parser_status"), "n_auto_elements": ar.get("n_elements"),
                                         "n_gt_elements": len(g.get(i, [])), "U_ann": float(np.mean(gy[i])) if gy.get(i) else None,
                                         "H_gt": (alpha * (r["O"] - 1) / 4 + (1 - alpha) * e_gt) if e_gt is not None else None,
                                         "Ecal_gt": float(np.mean(1 / (1 + np.exp(-(pa * np.array(gz[i]) + pb))))) if gz.get(i) else None}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
