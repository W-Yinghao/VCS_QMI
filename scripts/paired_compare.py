"""Paired per-seed comparison of 800-epoch runs (P114 / P120 / P126 / P127 reading rules).
    python scripts/paired_compare.py --a P107_AP3_views4_800ep_seed{s} --b P114_JSAP3_views4_800ep_seed{s} --seeds 0 1 2 [--close 0.3] [--out f.json]
Delta = A - B per seed (same seed); 95 % t interval over seeds.  Labels (P114 freeze decision 1): close |mean| < close; clear |mean| >= close and the
interval excludes 0; otherwise inconclusive.  Reads outputs/<run>/evaluations/evaluation_epoch_800.json; missing runs are listed, not imputed.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
METRICS = {"linear": "linear_val_top1_pct", "knn": "knn_val_top1_pct"}

def score(run):
    f = OUT / run / "evaluations" / "evaluation_epoch_800.json"
    if not f.is_file(): return None
    d = json.loads(f.read_text()); return {k: float(d[v]) for k, v in METRICS.items()}

def label(m, lo, hi, close):
    if abs(m) < close: return "close"
    return "clear" if (lo > 0 or hi < 0) else "inconclusive"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--close", type=float, default=0.3); ap.add_argument("--out")
    a = ap.parse_args(); res = {"a": a.a, "b": a.b, "close": a.close, "per_seed": {}, "missing": []}
    for s in a.seeds:
        ra, rb = a.a.format(s=s), a.b.format(s=s); A, B = score(ra), score(rb)
        if A is None or B is None: res["missing"] += [r for r, x in ((ra, A), (rb, B)) if x is None]; continue
        res["per_seed"][s] = {"A": A, "B": B}
    for k in METRICS:
        d = np.array([v["A"][k] - v["B"][k] for v in res["per_seed"].values()]); n = len(d)
        if n == 0: continue
        m = float(d.mean()); h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        res[k] = {"n": n, "A_mean": float(np.mean([v["A"][k] for v in res["per_seed"].values()])),
                  "B_mean": float(np.mean([v["B"][k] for v in res["per_seed"].values()])), "delta": d.tolist(), "mean": m,
                  "ci95": [m - h, m + h], "label": label(m, m - h, m + h, a.close) if n > 1 else "single seed"}
        print(f"{k:6s} n={n} A {res[k]['A_mean']:.2f} B {res[k]['B_mean']:.2f}  A-B {m:+.2f} [{m-h:+.2f}, {m+h:+.2f}]  per-seed {' '.join(f'{x:+.2f}' for x in d)}  -> {res[k]['label']}")
    if res["missing"]: print("missing:", ", ".join(res["missing"]))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
