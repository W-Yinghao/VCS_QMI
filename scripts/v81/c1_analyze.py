#!/usr/bin/env python
"""V8.1 C1 — CPU analysis of the 9 C1 units against the original full-adaptation V8 units (V81_C1_PROTOCOL_FROZEN_20261015.yaml).

Primary (pre-registered), DEV only, per (arm, seed) C1 minus the original V8-FIT unit of the same arm and seed, then the seed mean:
  (a) pooled within-prompt pair accuracy: all within-prompt pairs whose exact human means differ (vote sums compared by integer cross-multiplication),
      model ties count 0.5, pooled over pairs;
  (b) selected-image human mean: per prompt the top-1 of raw mu (ties -> lowest candidate index; candidates in lexical image-id order, as V81_02),
      the human mean of that image, averaged over prompts.
  Uncertainty: prompt_group cluster bootstrap (V7 near-duplicate component), 2,000 replicates, seed 2718, percentile 95 % interval of the seed-mean
  difference; one resample per replicate shared by every arm / seed / variant.  "improves" = interval lower bound > 0 for >= 2 of the 3 arms on (a)
  or on (b).
Secondary (descriptive): pooled Spearman, prompt-macro mean-rating MSE and its decomposition (prompt bias^2 + within-prompt centred error), RPS / CE,
CAL_SELECT curves from the logs, FIT-AUDIT best vs resume, cost.  Labels: EvalMuse development roles only (DEV, CAL_SELECT via the logs, FIT-AUDIT).

    python -I scripts/v81/c1_analyze.py [--out reports/V8_1_DIAGNOSTICS/results/c1_results.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
MANIFEST = Path("/home/infres/yinwang/CS_QMI/outputs/V7/data/manifest_images.jsonl")
V8 = Path("/projects/EEG-foundation-model/yinghao/outputs_v8/units")
C1 = Path("/projects/EEG-foundation-model/yinghao/outputs_v81/c1/units")
RES = REPO / "reports" / "V8_1_DIAGNOSTICS" / "results"
ARMS, SEEDS, REPS, BOOT_SEED = ["U-CS", "U-RPS", "U-MSE"], [0, 1, 2], 2000, 2718
LEVELS = np.arange(1, 6)


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev_items():
    """DEV overall-valid images: prompt, cluster, candidate order (lexical image id), vote sum / count (exact mean), histogram."""
    rows = [r for r in read_jsonl(MANIFEST) if r.get("role") == "DEV" and r.get("overall_votes")]
    per = defaultdict(list)
    for r in rows:
        per[str(r["prompt_id"])].append(r)
    prompts = sorted(per); items = {}
    for p in prompts:
        groups = {str(r["prompt_group"]) for r in per[p]}
        if len(groups) != 1:
            raise SystemExit(f"prompt {p} spans several prompt_groups")
        g = next(iter(groups))
        for k, r in enumerate(sorted(per[p], key=lambda r: str(r["image_id"]))):
            v = [int(x) for x in r["overall_votes"]]; h = np.bincount(v, minlength=6)[1:] / len(v)
            items[str(r["image_id"])] = {"prompt": p, "cluster": g, "cand": k, "s": sum(v), "n": len(v), "hist": h}
    return prompts, items


def per_prompt_stats(prompts, items, mu: dict):
    """Per prompt: concordance sum and informative pair count (a), selected human mean (b), plus image-level vectors for the secondary readouts."""
    byp = defaultdict(list)
    for i, it in items.items():
        byp[it["prompt"]].append(i)
    conc, npair, sel = np.zeros(len(prompts)), np.zeros(len(prompts)), np.zeros(len(prompts))
    for k, p in enumerate(prompts):
        ids = sorted(byp[p], key=lambda i: items[i]["cand"])
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ia, ib = items[ids[a]], items[ids[b]]; d = ia["s"] * ib["n"] - ib["s"] * ia["n"]
                if d == 0:
                    continue
                npair[k] += 1; dm = mu[ids[a]] - mu[ids[b]]
                conc[k] += 0.5 if dm == 0 else float((dm > 0) == (d > 0))
        best = max(range(len(ids)), key=lambda t: (mu[ids[t]], -t))                   # first maximum in candidate order
        sel[k] = items[ids[best]]["s"] / items[ids[best]]["n"]
    return conc, npair, sel


def secondary(prompts, items, pred: dict) -> dict:
    ids = sorted(items); mu = np.array([pred[i]["mu"] for i in ids]); h = np.array([items[i]["s"] / items[i]["n"] for i in ids])
    q = np.array([pred[i]["q"] for i in ids], float); u = np.array([pred[i]["u"] for i in ids], float); H = np.array([items[i]["hist"] for i in ids])
    y = np.flip(np.cumsum(np.flip(H, 1), 1), 1)[:, 1:]
    rps = float(np.mean(np.mean(y * (1 - u) ** 2 + (1 - y) * u ** 2, 1))); ce = float(np.mean(-(H * np.log(np.clip(q, 1e-12, 1))).sum(1)))
    byp = defaultdict(list)
    for k, i in enumerate(ids):
        byp[items[i]["prompt"]].append(k)
    mse, bias2, cen = [], [], []
    for p in prompts:
        e = mu[byp[p]] - h[byp[p]]; b = e.mean(); mse.append(float((e ** 2).mean())); bias2.append(float(b ** 2)); cen.append(float(((e - b) ** 2).mean()))
    return {"spearman_pooled": float(np.corrcoef(rankdata(mu), rankdata(h))[0, 1]), "mse_prompt_macro": float(np.mean(mse)), "mse_bias2_prompt_macro": float(np.mean(bias2)),
            "mse_centred_prompt_macro": float(np.mean(cen)), "rps_micro": rps, "ce_micro": ce, "mu_mean": float(mu.mean()), "mu_sd": float(mu.std())}


def boot_weights(prompts, items, reps=REPS, seed=BOOT_SEED):
    clus = sorted({it["cluster"] for it in items.values()}); cidx = {c: k for k, c in enumerate(clus)}
    p_c = np.array([cidx[next(it["cluster"] for it in items.values() if it["prompt"] == p)] for p in prompts])
    rng = np.random.default_rng(seed); W = np.zeros((reps, len(prompts)))
    for r in range(reps):
        cnt = np.bincount(rng.integers(0, len(clus), len(clus)), minlength=len(clus)); W[r] = cnt[p_c]
    return W, len(clus)


def spearman_weighted(mu, h, w):
    m = np.repeat(mu, w); hh = np.repeat(h, w)
    return float(np.corrcoef(rankdata(m), rankdata(hh))[0, 1])


def log_curves(unit: Path) -> dict:
    L = json.loads((unit / "log.json").read_text())
    return {"selected_epoch": L.get("selected_epoch"), "gpu": L.get("gpu"), "lr": L.get("lr"), "seconds": L.get("seconds"),
            "cal_mse": [e["cal_select"]["mean_mse_prompt_macro"] for e in L["epochs_log"]], "cal_pairacc": [e["cal_select"]["within_prompt_pairwise_acc_macro"] for e in L["epochs_log"]],
            "cal_rps": [e["cal_select"]["rps_micro"] for e in L["epochs_log"]], "train_loss": [e["train_means"]["loss"] for e in L["epochs_log"]],
            "train_s": [e["train_s"] for e in L["epochs_log"]], "eval_s": [e["eval_s"] for e in L["epochs_log"]], "steps": [e["steps"] for e in L["epochs_log"]],
            "peak_mem_gb": max((e.get("peak_mem_gb") or 0) for e in L["epochs_log"]), "n_trainable": (L.get("model") or {}).get("n_trainable_v8")}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(RES / "c1_results.json"))
    ap.add_argument("--c1-from-v8", action="store_true", help="code check only: use the V8 units on the C1 side (every difference must be exactly 0)")
    a = ap.parse_args(argv); t0 = time.time()
    prompts, items = dev_items(); W, n_clusters = boot_weights(prompts, items); ids = sorted(items)
    h_img = np.array([items[i]["s"] / items[i]["n"] for i in ids]); pidx = {p: k for k, p in enumerate(prompts)}; w_img_prompt = np.array([pidx[items[i]["prompt"]] for i in ids])
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "manifest_sha256": sha(MANIFEST), "n_dev_images": len(items), "n_dev_prompts": len(prompts),
           "n_clusters": n_clusters, "bootstrap": {"reps": REPS, "seed": BOOT_SEED, "unit": "prompt_group"}, "units": {}, "primary": {}, "secondary": {}}
    stats = {}
    for arm in ARMS:
        for seed in SEEDS:
            for variant, unit in (("C1", (V8 / f"V8-FIT-{arm}-s{seed}") if a.c1_from_v8 else (C1 / f"V81-C1-{arm}-s{seed}")), ("V8", V8 / f"V8-FIT-{arm}-s{seed}")):
                f = unit / "pred" / "images_DEV.jsonl"; pred = {r["image_id"]: r for r in read_jsonl(f)}
                if set(pred) != set(items):
                    raise SystemExit(f"{unit.name}: DEV ids differ from the manifest ({len(set(pred) ^ set(items))})")
                mu = {i: float(pred[i]["mu"]) for i in items}; conc, npair, sel = per_prompt_stats(prompts, items, mu)
                stats[(arm, seed, variant)] = {"conc": conc, "npair": npair, "sel": sel, "mu_img": np.array([mu[i] for i in ids])}
                out["units"][f"{variant}|{arm}|s{seed}"] = {"unit": unit.name, "dev_pred_sha256": sha(f), "pair_acc_pooled": float(conc.sum() / npair.sum()),
                                                            "selected_human_mean": float(sel.mean()), "n_informative_pairs": int(npair.sum()), **secondary(prompts, items, pred),
                                                            "log": log_curves(unit)}
                if variant == "C1" and (unit / "pred" / "fit_audit.json").exists():
                    out["units"][f"{variant}|{arm}|s{seed}"]["fit_audit"] = json.loads((unit / "pred" / "fit_audit.json").read_text())
    # primary: seed-mean differences with the shared cluster bootstrap
    for metric in ("pair_acc_pooled", "selected_human_mean"):
        res = {}
        for arm in ARMS:
            d_point = []; d_boot = np.zeros(REPS)
            for seed in SEEDS:
                c, v = stats[(arm, seed, "C1")], stats[(arm, seed, "V8")]
                if metric == "pair_acc_pooled":
                    d_point.append(c["conc"].sum() / c["npair"].sum() - v["conc"].sum() / v["npair"].sum())
                    d_boot += (W @ c["conc"]) / (W @ c["npair"]) - (W @ v["conc"]) / (W @ v["npair"])
                else:
                    d_point.append(c["sel"].mean() - v["sel"].mean())
                    d_boot += (W @ c["sel"]) / W.sum(1) - (W @ v["sel"]) / W.sum(1)
            d_boot /= len(SEEDS); lo, hi = np.percentile(d_boot, [2.5, 97.5])
            res[arm] = {"per_seed_diff": [float(x) for x in d_point], "seed_mean_diff": float(np.mean(d_point)), "ci95": [float(lo), float(hi)], "ci_above_0": bool(lo > 0),
                        "ci_below_0": bool(hi < 0)}
        res["n_arms_ci_above_0"] = sum(res[a]["ci_above_0"] for a in ARMS); out["primary"][metric] = res
    out["primary"]["improves"] = any(out["primary"][m]["n_arms_ci_above_0"] >= 2 for m in ("pair_acc_pooled", "selected_human_mean"))
    out["primary"]["rule"] = "improves = seed-mean 95 % interval lower bound > 0 for >= 2 of the 3 arms on (a) pair_acc_pooled or on (b) selected_human_mean"
    # secondary: pooled Spearman differences with the same resamples (image weights = prompt weights)
    sp = {}
    for arm in ARMS:
        d_point = []; d_boot = np.zeros(REPS)
        for seed in SEEDS:
            c, v = stats[(arm, seed, "C1")], stats[(arm, seed, "V8")]
            d_point.append(out["units"][f"C1|{arm}|s{seed}"]["spearman_pooled"] - out["units"][f"V8|{arm}|s{seed}"]["spearman_pooled"])
            for r in range(REPS):
                w = W[r][w_img_prompt].astype(int); d_boot[r] += spearman_weighted(c["mu_img"], h_img, w) - spearman_weighted(v["mu_img"], h_img, w)
        d_boot /= len(SEEDS); lo, hi = np.percentile(d_boot, [2.5, 97.5])
        sp[arm] = {"per_seed_diff": [float(x) for x in d_point], "seed_mean_diff": float(np.mean(d_point)), "ci95": [float(lo), float(hi)]}
    out["secondary"]["spearman_pooled_diff"] = sp
    keys = ("mse_prompt_macro", "mse_bias2_prompt_macro", "mse_centred_prompt_macro", "rps_micro", "ce_micro", "spearman_pooled", "pair_acc_pooled", "selected_human_mean")
    out["secondary"]["seed_means"] = {f"{variant}|{arm}": {k: float(np.mean([out["units"][f"{variant}|{arm}|s{s}"][k] for s in SEEDS])) for k in keys}
                                      for arm in ARMS for variant in ("C1", "V8")}
    out["seconds"] = time.time() - t0; Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({"primary": out["primary"], "spearman": sp}, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
