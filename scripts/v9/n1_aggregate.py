"""V9-N1 aggregate (package §04.5–04.6, §08.1): tables per source × rho (RMSE / MAE / bias of rho, interval coverage / width / empty / full rates,
posterior Brier / log loss / AUROC, ESS) and the RMSE averaged over the four rho values per source (the pre-registered main comparison, no
favourable-rho selection); the three figures as dependency-free SVG.  Markdown and JSON are generated from the cell files only.

    python scripts/v9/n1_aggregate.py --exact <dir> --image <dir> --out <prefix>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v9 import svgplot  # noqa: E402
from v9.quant import METHODS  # noqa: E402

MAIN = ("raw_moment", "vcs_moment", "bce_moment", "bbse_hard", "bbse_soft", "em_uncal", "em_bcts", "em_isotonic", "kde_moment", "kde_em", "rulsif_moment", "rulsif_em", "nystrom_moment", "nystrom_em", "kernel_mpe")


def load_cells(d: Path, kind: str):
    cells = []
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".partial.json") or f.name.endswith("_panel.json"):
            continue
        r = json.loads(f.read_text()); r["_file"] = f.name; r["_kind"] = kind; cells.append(r)
    return cells


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def table_rho(cells, group_key):
    """Per (group, rho, n): RMSE of every method (rows = methods, columns = cells)."""
    keys = sorted({(c[group_key], c["rho"], c["n"]) for c in cells}); lines = ["| method | " + " | ".join(f"{g} ρ={r:g} n={n}" for g, r, n in keys) + " |", "|---|" + "---|" * len(keys)]
    for m in MAIN:
        row = []
        for g, r, n in keys:
            cs = [c for c in cells if (c[group_key], c["rho"], c["n"]) == (g, r, n)]
            vals = [c["summary"]["methods"][m]["rmse"] for c in cs if c["summary"]["methods"][m]["rmse"] is not None]
            row.append(fmt(float(np.mean(vals))) if vals else "—")
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_avg(cells, group_key, metric="rmse"):
    """Main comparison: metric averaged over the rho grid per group and n (mean over seeds where present)."""
    groups = sorted({c[group_key] for c in cells}); ns = sorted({c["n"] for c in cells})
    lines = ["| method | " + " | ".join(f"{g} n={n}" for g in groups for n in ns) + " |", "|---|" + "---|" * (len(groups) * len(ns))]
    out = {}
    for m in MAIN:
        row = []
        for g in groups:
            for n in ns:
                per_rho = []
                for rho in sorted({c["rho"] for c in cells}):
                    cs = [c for c in cells if (c[group_key], c["rho"], c["n"]) == (g, rho, n)]; vals = [c["summary"]["methods"][m][metric] for c in cs if c["summary"]["methods"][m][metric] is not None]
                    if vals:
                        per_rho.append(float(np.mean(vals)))
                v = float(np.mean(per_rho)) if per_rho else None; out[(m, g, n)] = v; row.append(fmt(v))
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines), out


def table_intervals(cells, group_key, key_fn):
    keys = sorted({(c[group_key], c["rho"], c["n"]) for c in cells}); lines = ["| cell | coverage | mean width (bounded) | bounded | full | empty |", "|---|---|---|---|---|---|"]
    for g, r, n in keys:
        cs = [c for c in cells if (c[group_key], c["rho"], c["n"]) == (g, r, n)]; ivs = [key_fn(c) for c in cs if key_fn(c)]
        if not ivs:
            continue
        cov = np.mean([i["coverage"] for i in ivs]); w = [i["mean_width_bounded"] for i in ivs if i["mean_width_bounded"] is not None]
        lines.append(f"| {g} ρ={r:g} n={n} | {cov:.3f} | {fmt(float(np.mean(w)) if w else None)} | {np.mean([i['rate_bounded'] for i in ivs]):.2f} | {np.mean([i['rate_full'] for i in ivs]):.2f} | {np.mean([i['rate_empty'] for i in ivs]):.2f} |")
    return "\n".join(lines)


def table_posterior(cells, group_key):
    groups = sorted({c[group_key] for c in cells}); lines = ["| method | " + " | ".join(f"{g}: Brier / logloss / AUROC / ESS" for g in groups) + " |", "|---|" + "---|" * len(groups)]
    for m in MAIN:
        row = []
        for g in groups:
            cs = [c for c in cells if c[group_key] == g and c["rho"] > 0]
            def mean_of(k):
                v = [c["summary"]["methods"][m][k]["mean"] for c in cs if c["summary"]["methods"][m][k]["mean"] is not None]; return float(np.mean(v)) if v else None
            row.append(f"{fmt(mean_of('brier'))} / {fmt(mean_of('logloss'))} / {fmt(mean_of('auroc'))} / {fmt(mean_of('ess_fraction'), 2)}")
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def figures(cells, group_key, out_dir: Path, tag: str):
    paths = {}
    # 1. rho true vs estimate (mean over repeats, per method) at the largest n per group
    for g in sorted({c[group_key] for c in cells}):
        n_max = max(c["n"] for c in cells if c[group_key] == g); series = []
        for m in ("raw_moment", "vcs_moment", "bce_moment", "bbse_hard", "em_bcts", "kde_moment", "rulsif_moment", "nystrom_moment", "kernel_mpe"):
            xs, ys = [], []
            for rho in sorted({c["rho"] for c in cells if c[group_key] == g}):
                cs = [c for c in cells if (c[group_key], c["rho"], c["n"]) == (g, rho, n_max)]
                est = [r["methods"][m]["rho_clipped"] for c in cs for r in c["records"] if r["methods"][m]["rho_clipped"] is not None]
                if est:
                    xs.append(float(np.mean([r.get("rho_realized", rho) for c in cs for r in c["records"]]))); ys.append(float(np.mean(est)))
            series.append({"name": m, "x": xs, "y": ys})
        paths[f"rho_{g}"] = svgplot.plot(series, out_dir / f"{tag}_rho_estimate_{g}.svg", f"{tag} {g}: ρ estimate vs realised ρ (n = {n_max}, mean over repeats)", "realised ρ", "estimate", identity=True, xlim=(0, 0.6), ylim=(0, 0.7))
    # 2. interval width vs n at matched coverage (oracle / VCS / BCE fixed scores)
    series = []
    for g in sorted({c[group_key] for c in cells}):
        for key, lab in (("vcs", "VCS critic"), ("bce", "BCE critic")):
            xs, ys = [], []
            for n in sorted({c["n"] for c in cells if c[group_key] == g}):
                cs = [c for c in cells if c[group_key] == g and c["n"] == n]; w = []
                for c in cs:
                    iv = c["summary"].get(f"interval_{key}") or c["summary"]["methods"].get(f"{key}_moment", {}).get("interval")
                    if iv and iv["mean_width_bounded"] is not None:
                        w.append(iv["mean_width_bounded"])
                if w:
                    xs.append(n); ys.append(float(np.mean(w)))
            if xs:
                series.append({"name": f"{g} {lab}", "x": xs, "y": ys})
    if series:
        paths["width"] = svgplot.plot(series, out_dir / f"{tag}_interval_width.svg", f"{tag}: finite-sample set width vs batch size (α = 0.05)", "batch size n", "mean width (bounded sets)", logx=True)
    # 3. Brier vs ESS fraction per method (rho > 0 cells)
    series = []
    for m in ("vcs_moment", "bce_moment", "em_bcts", "em_isotonic", "kde_moment", "rulsif_moment", "nystrom_moment", "kernel_mpe"):
        xs, ys = [], []
        for c in cells:
            if c["rho"] > 0:
                b, e = c["summary"]["methods"][m]["brier"]["mean"], c["summary"]["methods"][m]["ess_fraction"]["mean"]
                if b is not None and e is not None:
                    xs.append(e); ys.append(b)
        if xs:
            series.append({"name": m, "x": xs, "y": ys, "line": False})
    if series:
        paths["brier_ess"] = svgplot.plot(series, out_dir / f"{tag}_brier_vs_ess.svg", f"{tag}: posterior Brier vs ESS fraction (cells with ρ > 0)", "ESS / n", "Brier of w")
    return {k: str(v) for k, v in paths.items()}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--exact", default=None); ap.add_argument("--image", default=None); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); fig_dir = out.parent / "figs"; md = ["# V9-N1 aggregate", ""]; J = {}
    for kind, d, gk in (("exact", a.exact, "design"), ("image", a.image, "seed")):
        if not d:
            continue
        cells = load_cells(Path(d), kind)
        if not cells:
            md.append(f"## {kind}: no cells"); continue
        avg_md, avg = table_avg(cells, gk); md += [f"## N1-{kind.capitalize()} ({len(cells)} cells; files in `{d}`)", "", "### Main comparison: RMSE of ρ̂ averaged over the ρ grid (per {gk}, per n)".replace("{gk}", gk), "", avg_md, "",
                                                 f"### RMSE per cell", "", table_rho(cells, gk), "", "### Finite-sample sets (fixed VCS critic)", "",
                                                 table_intervals(cells, gk, lambda c: c["summary"].get("interval_vcs") or c["summary"]["methods"]["vcs_moment"].get("interval")), "",
                                                 "### Posterior weights (cells with ρ > 0; mean over repeats and cells)", "", table_posterior(cells, gk), ""]
        if kind == "exact":
            md += ["### Fixed-score controls (oracle η; affine / anti identical by invariance; constant unidentified; noise)", ""]
            lines = ["| cell | oracle MAE | affine MAE | anti MAE | constant identified | noise MAE | oracle interval coverage |", "|---|---|---|---|---|---|---|"]
            for c in cells:
                cc = c["summary"]["controls"]
                lines.append(f"| {c['design']} ρ={c['rho']:g} n={c['n']} | {fmt(cc['oracle_eta']['mae'])} | {fmt(cc['affine_eta']['mae'])} | {fmt(cc['anti_eta']['mae'])} | {cc['constant']['n_identified']} / {c['repeats']} | {fmt(cc['noise']['mae'])} | {fmt(cc['oracle_eta'].get('interval', {}).get('coverage'))} |")
            md += lines + [""]
        if kind == "image":
            md += ["Note: realised ρ (the replaced fraction) is the target; repeats share the U pool and the fitted panel (dependent).", ""]
        figs = figures(cells, gk, fig_dir, kind); md += ["Figures: " + ", ".join(f"`{v}`" for v in figs.values()), ""]
        J[kind] = {"n_cells": len(cells), "files": [c["_file"] for c in cells], "avg_rmse": {f"{m}|{g}|{n}": v for (m, g, n), v in avg.items()}, "figures": figs,
                   "cells": [{k: c[k] for k in ("rho", "n", "repeats", "summary") if k in c} | {gk: c[gk]} for c in cells]}
    Path(str(out) + ".md").write_text("\n".join(md) + "\n"); Path(str(out) + ".json").write_text(json.dumps(J, indent=1)); print(f"-> {out}.md / .json"); return 0


if __name__ == "__main__":
    sys.exit(main())
