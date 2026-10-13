"""V7 evaluators, fixed before any DEV or external label is read (V7 package docs/05).
  table-a --units U [U ...] --out F       EvalMuse DEV (public-train prompt-disjoint holdout) probability table per M unit: natural element Brier
            (prompt -> image -> element macro; micro for reference) of (i) the prior-corrected output p_pi (primary), (ii) the raw balanced output
            sigmoid(z), (iii) a Platt fit on CAL_PROB only (a, b on z); soft NLL (eps 1e-7, saturation share reported); majority accuracy
            (y = 0.5 excluded); per-prompt-macro AUROC over elements with y in {0, 1} (NA for a single class); ECE (10 fixed bins, auxiliary).
            Pre-registered primary: M2 - M1 natural Brier (p_pi) on DEV, per seed (same seed pairs), prompt-cluster paired bootstrap (5,000, seed 2718).
Unit dirs carry pred/elements_{CAL_PROB,DEV}.jsonl from v7_train.py export.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as VC  # noqa: E402

BOOT, BOOT_SEED, ECE_BINS = 5000, 2718, 10


def platt_fit(z, y, iters=200):
    """Logistic calibration p = sigmoid(a z + b) fitted by Newton steps on soft labels (CAL_PROB only)."""
    z, y = np.asarray(z, float), np.asarray(y, float); a, b = 1.0, 0.0
    for _ in range(iters):
        s = a * z + b; p = 1 / (1 + np.exp(-s)); g = np.array([np.sum((p - y) * z), np.sum(p - y)]); w = p * (1 - p) + 1e-12
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-9 * np.eye(2); step = np.linalg.solve(H, g); a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def auroc_macro(pids, p, y):
    """Per-prompt AUROC over binary-labelled elements (ties 0.5), averaged over prompts that have both classes; NA otherwise."""
    g = defaultdict(list)
    for q, pp, yy in zip(pids, p, y):
        if yy in (0.0, 1.0):
            g[q].append((pp, yy))
    vals = []
    for rows in g.values():
        pos = [pp for pp, yy in rows if yy == 1.0]; neg = [pp for pp, yy in rows if yy == 0.0]
        if pos and neg:
            vals.append(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))
    return (float(np.mean(vals)) if vals else None), len(vals)


def ece(p, y, bins=ECE_BINS):
    p, y = np.asarray(p, float), np.asarray(y, float); edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for k in range(bins):
        m = (p >= edges[k]) & ((p < edges[k + 1]) if k < bins - 1 else (p <= 1))
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def per_prompt(pids, iids, v):
    g = defaultdict(lambda: defaultdict(list))
    for q, i, x in zip(pids, iids, v):
        g[q][i].append(float(x))
    return {q: float(np.mean([np.mean(x) for x in imgs.values()])) for q, imgs in g.items()}


def unit_table(unit: Path):
    log = json.loads((unit / "log.json").read_text()); pi = log["prior_fit_spanned"]
    cal = VC.read_jsonl(unit / "pred" / "elements_CAL_PROB.jsonl"); dev = VC.read_jsonl(unit / "pred" / "elements_DEV.jsonl")
    a, b = platt_fit([e["z"] for e in cal], [e["y"] for e in cal]); z = np.array([e["z"] for e in dev]); y = np.array([e["y"] for e in dev])
    pids, iids = [e["prompt_id"] for e in dev], [e["image_id"] for e in dev]
    outs = {"prior_corrected": VC.natural_probability(z, pi), "raw_balanced": 1 / (1 + np.exp(-z)), "cal_prob_platt": 1 / (1 + np.exp(-(a * z + b)))}
    res = {"unit": unit.name, "arm": log["arm"], "seed": log["seed"], "lr": log["lr"], "prior": pi, "platt": [a, b], "n_elements": len(dev)}
    for k, p in outs.items():
        br = VC.expected_brier(p, y); nll = VC.soft_nll(p, y); maj = (y != 0.5)
        res[k] = {"natural_brier_macro": VC.hierarchical_mean(pids, iids, br)[0], "natural_brier_micro": float(br.mean()),
                  "nll_macro": VC.hierarchical_mean(pids, iids, nll)[0], "saturated_share": float(((p < 1e-7) | (p > 1 - 1e-7)).mean()),
                  "accuracy_majority": float(((p[maj] >= 0.5) == (y[maj] > 0.5)).mean()) if maj.any() else None, "excluded_y_half": int((~maj).sum()),
                  "auroc_prompt_macro": auroc_macro(pids, p, y)[0], "ece_10": ece(p, y)}
    res["per_prompt_brier"] = per_prompt(pids, iids, VC.expected_brier(outs["prior_corrected"], y)); return res


def paired_boot(a: dict, b: dict, reps=BOOT, seed=BOOT_SEED):
    keys = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in keys]); rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return {"delta": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "n_prompts": len(keys), "replicates": reps}


def cmd_table_a(a):
    tabs = [unit_table(Path(u)) for u in a.units]; by = defaultdict(dict)
    for t in tabs:
        by[t["arm"]][t["seed"]] = t
    primary = {}
    for s in sorted(set(by.get("M2_semantic_vcs", {})) & set(by.get("M1_semantic_logistic", {}))):
        primary[str(s)] = paired_boot(by["M2_semantic_vcs"][s]["per_prompt_brier"], by["M1_semantic_logistic"][s]["per_prompt_brier"])
    out = {"units": [{k: v for k, v in t.items() if k != "per_prompt_brier"} for t in tabs], "primary_M2_minus_M1_natural_brier_dev": primary,
           "rule": "negative = M2 lower risk; per seed, same seed pairs; prompt-cluster paired bootstrap"}
    Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(primary, indent=1)); return 0


REF_DIR = Path("/home/infres/yinwang/CS_QMI/outputs/V7/package_ref/reference")
GENS = ("DALLE_3", "SDXL_Base")


def _ref():
    """The package's reference ranking functions (ranking_pool / holm), imported from its read-only copy."""
    if str(REF_DIR) not in sys.path:
        sys.path.insert(0, str(REF_DIR))
    import v7_metrics as RM
    return RM


def pools(pred_rows, labels):
    """Join a prediction export with the sealed labels by explicit IDs -> {(prompt, gen): (scores[9], human[9])} in candidate order."""
    lab = {(str(r["prompt_id"]), r["generator_id"], r["image_id"]): r for r in labels}; pred = {}
    for r in pred_rows:
        k = (str(r["prompt_id"]), r["generator_id"], r["image_id"])
        if k in pred:
            raise ValueError(f"duplicate prediction {k}")
        if not math.isfinite(float(r["score"])):
            raise ValueError(f"nonfinite prediction {k}")
        pred[k] = float(r["score"])
    if set(pred) != set(lab):
        raise ValueError("prediction / label universes differ: no silent dropping")
    g = defaultdict(list)
    for k, r in lab.items():
        g[k[:2]].append((r["candidate_index"], pred[k], float(r["human_score"])))
    out = {}
    for k, v in g.items():
        v.sort()
        if [c for c, _, _ in v] != list(range(9)):
            raise ValueError(f"pool {k} must hold candidates 0..8 once")
        out[k] = (np.array([x for _, x, _ in v]), np.array([h for _, _, h in v]))
    return out


def rank_method(P, protocol="official", tie_rule="first"):
    RM = _ref(); res = {}
    for gen in GENS:
        rows = {k[0]: {n: RM.ranking_pool(s, h, n, protocol, tie_rule) for n in (1, 3, 9)} for k, (s, h) in P.items() if k[1] == gen}
        res[gen] = {"per_prompt_selected_N9": {p: r[9]["selected"] for p, r in rows.items()},
                    **{f"N{n}": {x: float(np.mean([r[n][x] for r in rows.values()])) for x in ("selected", "oracle", "regret")} for n in (1, 3, 9)},
                    "n_prompts": len(rows)}
    return res


def cmd_rank(a):
    """Methods: {name: [prediction files, one per seed]}; seeds of a method are paired with the same seed index of an M comparator, a seedless
    baseline (one file) is reused for every seed.  Primary: M2-E vs M1-E / B0 / B1 on N9 per generator (6), Holm."""
    if not a.allow_scoring:
        raise SystemExit("refusing to read sealed labels without --allow-scoring (after the prediction / protocol freeze and the owner's approval)")
    labels = VC.read_jsonl(a.labels); methods = json.loads(Path(a.methods).read_text()); RM = _ref(); res = {"methods": {}, "sensitivity": {}}
    per = {}
    for name, files in methods.items():
        runs = [rank_method(pools(VC.read_jsonl(f), labels)) for f in files]; per[name] = runs
        res["methods"][name] = {gen: {f"N{n}": {x: [r[gen][f"N{n}"][x] for r in runs] for x in ("selected", "regret")} for n in (1, 3, 9)} for gen in GENS}
        res["sensitivity"][name] = {proto_tie: [rank_method(pools(VC.read_jsonl(f), labels), *proto_tie.split("/"))[g]["N3" if proto_tie.startswith("all") else "N9"]["selected"]
                                                for f in files for g in GENS] for proto_tie in ("all_subsets/first", "official/mean")}
    rng = np.random.default_rng(BOOT_SEED); prompts = sorted({p for (p, g) in pools(VC.read_jsonl(next(iter(methods.values()))[0]), labels)})
    idx = [rng.integers(0, len(prompts), len(prompts)) for _ in range(BOOT)]; prim = {}; pv = []
    for comp in a.comparators:
        for gen in GENS:
            ours, theirs = per[a.ours], per[comp]; S = len(ours)
            d = np.array([np.mean([ours[s][gen]["per_prompt_selected_N9"][p] - theirs[s if len(theirs) == S else 0][gen]["per_prompt_selected_N9"][p] for s in range(S)]) for p in prompts])
            means = np.array([d[i].mean() for i in idx]); p2 = min(1.0, 2 * min((1 + (means <= 0).sum()) / (BOOT + 1), (1 + (means >= 0).sum()) / (BOOT + 1)))
            prim[f"{a.ours} - {comp} @ {gen} N9"] = {"delta": float(d.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "p_approx": p2,
                                                     "per_seed_delta": [float(np.mean([ours[s][gen]["per_prompt_selected_N9"][p] - theirs[s if len(theirs) == S else 0][gen]["per_prompt_selected_N9"][p] for p in prompts])) for s in range(S)]}
            pv.append(p2)
    for k, h in zip(prim, RM.holm(pv)):
        prim[k]["p_holm"] = float(h)
    res["primary"] = prim; res["rule"] = "prompt-cluster bootstrap shared across generators (5,000, seed 2718); seed-paired per-prompt N9 differences averaged over seeds; Holm over the family"
    out = Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite a scored result; use a new versioned path")
    out.write_text(json.dumps(res, indent=1)); print(json.dumps(prim, indent=1)); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table-a"); t.add_argument("--units", nargs="+", required=True); t.add_argument("--out", required=True)
    r = sub.add_parser("rank"); r.add_argument("--methods", required=True); r.add_argument("--labels", required=True); r.add_argument("--out", required=True)
    r.add_argument("--ours", default="M2_E"); r.add_argument("--comparators", nargs="+", default=["M1_E", "B0_FGA_released_O", "B1_VQAScore"]); r.add_argument("--allow-scoring", action="store_true")
    a = ap.parse_args(argv)
    return cmd_table_a(a) if a.cmd == "table-a" else cmd_rank(a)


if __name__ == "__main__":
    sys.exit(main())
