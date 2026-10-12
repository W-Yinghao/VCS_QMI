"""R6 REG-P3 — the two secondary readouts the aggregate does not summarise (frozen protocol, metrics.secondary):
  1. window-macro success (REG-P2's summary, descriptive): per DEV window, success rate over its 10 perturbations; mean over windows;
     mean ± sd over seeds for the seeded rows.  Thresholds in the result files' units (err_full = 4 × native px): < 42 / < 9 = < 10.5 / < 2.25 native px.
  2. exact F profiles along θ and x for the first DEV window of 5 categories ('#profile' entries of each run's DEV records): per method and
     window, the offset of the profile's maximum from the truth and the width of the region where the min–max-normalised profile is >= 0.5
     (contiguous around the maximum); a PNG of the normalised profiles (seed 0 for the seeded rows).
Descriptive only; nothing here is a contrast or a selection.  Run on a CPU node:  python -I scripts/r6_reg_p3_secondary.py --results R --out-prefix P
"""
import argparse, json, statistics as st
from pathlib import Path

ORDER = ["vcs", "logistic", "infonce", "comir", "mi", "nmi", "kernel"]


def window_macro(dev: dict, thr: float) -> float:
    rates = [sum(r["err_full"] < thr for r in recs) / len(recs) for k, recs in dev.items() if "#" not in k and recs]
    return sum(rates) / len(rates)


def peak(offsets, F):
    i = max(range(len(F)), key=lambda j: F[j]); lo, hi = min(F), max(F)
    if hi - lo <= 0: return offsets[i], float("nan")
    z = [(f - lo) / (hi - lo) for f in F]; a = b = i
    while a > 0 and z[a - 1] >= 0.5: a -= 1
    while b < len(z) - 1 and z[b + 1] >= 0.5: b += 1
    return offsets[i], offsets[b] - offsets[a]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out-prefix", required=True); a = ap.parse_args()
    runs = {}
    for f in sorted(Path(a.results).glob("*.json")):
        r = json.load(open(f)); runs.setdefault(r["method"], []).append(r)
    out = {"window_macro": {}, "profiles": {}}
    lines = ["# R6 REG-P3 — secondary readouts (descriptive)", "",
             "## Window-macro DEV success (%; per window over its 10 perturbations, mean over windows; mean ± sd over seeds)", "",
             "| method | seeds | n windows | < 10.5 px | < 2.25 px |", "|---|---|---|---|---|"]
    for m in ORDER:
        if m not in runs: continue
        rs = sorted(runs[m], key=lambda r: (r.get("seed") is None, r.get("seed")))
        s42 = [100 * window_macro(r["dev"], 42) for r in rs]; s9 = [100 * window_macro(r["dev"], 9) for r in rs]
        nw = len([k for k in rs[0]["dev"] if "#" not in k])
        fmt = lambda v: f"{st.mean(v):.1f} ± {st.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
        out["window_macro"][m] = {"seeds": [r.get("seed") for r in rs], "n_windows": nw, "success42": s42, "success9": s9}
        lines.append(f"| {m} | {[r.get('seed') for r in rs]} | {nw} | {fmt(s42)} | {fmt(s9)} |")
    lines += ["", "## Energy profiles around the truth (first DEV window of 5 categories)", "",
              "Offset of the profile maximum from the truth and the width of the >= 0.5 region of the min–max-normalised profile "
              "(θ in degrees; x in the profile's own 512-px-window pixels).  Seeded rows: all seeds.", "",
              "| method | windows × seeds | θ: max offset 0 (count) | θ: median width (°) | x: max offset 0 (count) | x: median width (px) |",
              "|---|---|---|---|---|---|"]
    prof_windows = None
    for m in ORDER:
        if m not in runs: continue
        th0 = x0 = n = 0; wth, wx = [], []
        for r in runs[m]:
            for k, v in r["dev"].items():
                if not k.endswith("#profile"): continue
                ot, wt = peak(v["theta_offsets_deg"], v["F_theta"]); ox, wxx = peak(v["x_offsets_px512"], v["F_x"])
                n += 1; th0 += abs(ot) < 1e-9; x0 += abs(ox) < 1e-9; wth.append(wt); wx.append(wxx)
                out["profiles"].setdefault(m, []).append({"seed": r.get("seed"), "window": k.split("#")[0], "theta_max_offset": ot,
                                                         "theta_width": wt, "x_max_offset": ox, "x_width": wxx})
        if n:
            lines.append(f"| {m} | {n} | {th0} / {n} | {st.median(wth):.1f} | {x0} / {n} | {st.median(wx):.1f} |")
    Path(a.out_prefix + ".md").write_text("\n".join(lines) + "\n"); Path(a.out_prefix + ".json").write_text(json.dumps(out, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        wins = sorted({p["window"] for ps in out["profiles"].values() for p in ps})
        fig, ax = plt.subplots(len(wins), 2, figsize=(9, 2.1 * len(wins)), sharex="col")
        for m in ORDER:
            r0 = next((r for r in runs.get(m, []) if r.get("seed") in (0, None)), None)
            if r0 is None: continue
            for i, w in enumerate(wins):
                v = r0["dev"].get(w + "#profile")
                if not v: continue
                for j, (o, F) in enumerate(((v["theta_offsets_deg"], v["F_theta"]), (v["x_offsets_px512"], v["F_x"]))):
                    lo, hi = min(F), max(F); ax[i][j].plot(o, [(f - lo) / (hi - lo) if hi > lo else 0 for f in F], lw=1.5, label=m)
        for i, w in enumerate(wins):
            ax[i][0].set_ylabel(w, fontsize=8); ax[i][0].axvline(0, c="0.6", lw=0.8); ax[i][1].axvline(0, c="0.6", lw=0.8)
        ax[-1][0].set_xlabel("θ offset from truth (°)"); ax[-1][1].set_xlabel("x offset from truth (512-px window px)")
        ax[0][1].legend(fontsize=7, ncol=2, frameon=False); fig.suptitle("REG-P3 DEV energy profiles (min–max normalised; seed 0)", fontsize=10)
        fig.tight_layout(); fig.savefig(a.out_prefix + "_profiles.png", dpi=130)
    except Exception as e:      # the tables above are the readout; the figure is optional
        print("figure skipped:", e)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
