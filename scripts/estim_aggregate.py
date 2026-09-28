"""E line / R1 aggregate: per (setting, kind, variant, N, lr) over seeds — target-free staircase separability (brief appendix B), tail
statistics, learned/oracle ratios; R1 shifts in resolution units.

    python scripts/estim_aggregate.py --inputs <dir with cell JSONs> --out <prefix>

Separability between adjacent levels k, k+1: |mean_{k+1} − mean_k| / pooled SD, where mean_k and SD_k pool the last-1000-step evaluation
estimates of all seeds (frozen definition); IQR version uses medians and pooled IQR.  Divergence/spike counts are summed over seeds and levels.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(d):
    cells = []
    for f in sorted(Path(d).glob("*.json")):
        if f.name.startswith(("aggregate", "oracle_table")): continue
        try: cells.append(json.loads(f.read_text()))
        except Exception as e: print("skip", f.name, e)
    return cells


def last_vals(level, n_last=10):
    ev = np.array([r[2] for r in level["log"]], dtype=float); return ev[-n_last:]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    cells = load(a.inputs); groups = defaultdict(list)
    for c in cells:
        key = (c["setting"], c["kind"], c["variant"], c["N"], c["lr"], json.dumps(c.get("contamination")) if c.get("contamination") else "")
        groups[key].append(c)
    rows = []
    for key, cs in sorted(groups.items()):
        levels = sorted({l["level"] for c in cs for l in c["levels"]}); per = {}
        for lv in levels:
            vals = np.concatenate([last_vals(l) for c in cs for l in c["levels"] if l["level"] == lv]); vals = vals[np.isfinite(vals)]
            ratios = [l["learned_over_oracle"] for c in cs for l in c["levels"] if l["level"] == lv and np.isfinite(l["learned_over_oracle"])]
            per[lv] = {"mean": float(vals.mean()) if len(vals) else float("nan"), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                       "median": float(np.median(vals)) if len(vals) else float("nan"), "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) else float("nan"),
                       "truth": next(l["truth"][c["target"]] for c in cs for l in c["levels"] if l["level"] == lv), "mi": next(l["mi_nats"] for c in cs for l in c["levels"] if l["level"] == lv),
                       "learned_over_oracle": float(np.mean(ratios)) if ratios else float("nan"), "q01": float(np.min([l["q01"] for c in cs for l in c["levels"] if l["level"] == lv])),
                       "q99": float(np.max([l["q99"] for c in cs for l in c["levels"] if l["level"] == lv])), "max_jump": float(np.max([l["max_jump"] for c in cs for l in c["levels"] if l["level"] == lv])),
                       "divergences": int(sum(l["divergences"] for c in cs for l in c["levels"] if l["level"] == lv)), "n_vals": int(len(vals))}
        sep = {}
        for k0, k1 in zip(levels[:-1], levels[1:]):
            p0, p1 = per[k0], per[k1]; psd = np.sqrt((p0["sd"] ** 2 + p1["sd"] ** 2) / 2); piqr = (p0["iqr"] + p1["iqr"]) / 2
            sep[f"{k0}-{k1}"] = {"sd": float(abs(p1["mean"] - p0["mean"]) / psd) if psd > 0 else float("nan"), "iqr": float(abs(p1["median"] - p0["median"]) / piqr) if piqr > 0 else float("nan")}
        rows.append({"setting": key[0], "kind": key[1], "variant": key[2], "N": key[3], "lr": key[4], "contamination": key[5], "seeds": len(cs), "levels": per, "separability": sep,
                     "divergences": sum(p["divergences"] for p in per.values()), "max_jump": max(p["max_jump"] for p in per.values()), "spikes_gt5": int(sum(1 for c in cs for l in c["levels"] if l["max_jump"] > 5.0))})
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    L = ["# Estimator benchmark — aggregate", "", "## Per cell group (seeds pooled): level means, separability, tails", "",
         "| setting | estimator | negatives | N | lr | contam | seeds | level means (truth) | separability sd [iqr] per step | divergences | max jump | spikes>5 | learned/oracle per level |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lm = "; ".join(f"L{k}: {p['mean']:.3f}±{p['sd']:.3f} ({p['truth']:.3f})" for k, p in r["levels"].items())
        sp = "; ".join(f"{k}: {v['sd']:.2f} [{v['iqr']:.2f}]" for k, v in r["separability"].items())
        lo = "; ".join(f"{p['learned_over_oracle']:.3f}" for p in r["levels"].values())
        L.append(f"| {r['setting']} | {r['kind']} | {r['variant']} | {r['N']} | {r['lr']:g} | {r['contamination'] or '—'} | {r['seeds']} | {lm} | {sp} | {r['divergences']} | {r['max_jump']:.2f} | {r['spikes_gt5']} | {lo} |")
    # R1: shift of the (4-nat) level mean vs eps = 0 in resolution units (adjacent-step gap of the clean staircase for the same estimator/variant/N/lr)
    clean = {(r["setting"], r["kind"], r["variant"], r["N"], r["lr"]): r for r in rows if not r["contamination"]}
    cont = [r for r in rows if r["contamination"]]
    if cont:
        L += ["", "## R1 — shift in resolution units (level 1 = 4 nats; unit = |mean_L2 − mean_L1| of the clean staircase for the same estimator; if the clean run lacks L2 the SD of L1 is used)", "", "| estimator | negatives | contamination | eps | mean | shift / unit |", "|---|---|---|---|---|---|"]
        base = {}
        for r in cont:
            ck = json.loads(r["contamination"]); key = (r["setting"], r["kind"], r["variant"], r["N"], r["lr"])
            base.setdefault((key, ck["kind"]), r if ck["eps"] == 0 else None)
        for r in sorted(cont, key=lambda r: (r["kind"], r["variant"], r["contamination"])):
            ck = json.loads(r["contamination"]); key = (r["setting"], r["kind"], r["variant"], r["N"], r["lr"]); b = base.get((key, ck["kind"]))
            lv = r["levels"].get(1) or next(iter(r["levels"].values()))
            unit = float("nan")
            if key in clean and 1 in clean[key]["levels"] and 2 in clean[key]["levels"]: unit = abs(clean[key]["levels"][2]["mean"] - clean[key]["levels"][1]["mean"])
            elif b is not None: unit = (b["levels"].get(1) or next(iter(b["levels"].values())))["sd"]
            shift = (lv["mean"] - (b["levels"].get(1) or next(iter(b["levels"].values())))["mean"]) if b is not None else float("nan")
            L.append(f"| {r['kind']} | {r['variant']} | {ck['kind']} | {ck['eps']:g} | {lv['mean']:.4f} | {shift / unit if unit else float('nan'):+.2f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); print("->", a.out + ".md", len(rows), "groups"); return 0


if __name__ == "__main__":
    import sys; sys.exit(main())
