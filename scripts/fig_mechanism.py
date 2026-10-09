"""Mechanism figure (T2e; EXPERIMENT_ORDER B5) — assembly of existing frozen results only, no new measurement.
  (a) weight on contaminated positives: gradient-norm ratio of replaced to kept positives (w.r.t. projector outputs, second half of training),
      P145 seed 0, CIFAR-10 / CIFAR-100 (reports/P145_results.json)
  (b) where the information sits: CIFAR-100 fine-label linear readout (recipe_raw) at layer3 / h / r / z, mean ± sd over seeds
      (P141; A-P3 and JS-AP3 5 seeds, SimCLR 3; reports/P141_v7_layer_readout_results.json)
  (c) dependence the critic can fit at h vs z: common J of the VCS-MLP critic at t 0 for VCS and SimCLR encoders, seeds 1-2 (P149 + add. 1);
      fit-limited lower readings (W8 / W11: not an S or DPI comparison across sites)
Palette: reference categorical slots 1-3 (validated: CVD / normal-vision pass; aqua below 3:1 contrast → legend + direct labels).
    python scripts/fig_mechanism.py   (vl_baselines venv: matplotlib)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "reports"; OUT = R / "figures"
COL = {"VCS": "#2a78d6", "JS": "#eb6834", "SimCLR": "#1baf7a"}; NAME = {"VCS": "VCS (A-P3)", "JS": "matched JS", "SimCLR": "SimCLR"}; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def main() -> int:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "text.color": INK, "axes.labelcolor": INK})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.2, 3.9), gridspec_kw={"width_ratios": [1, 1.15, 1.25]})
    # (a) contaminated-positive weight
    p = json.load(open(R / "P145_results.json")); x = np.arange(2); w = 0.26
    for j, (m, key) in enumerate((("VCS", "vcs"), ("JS", "js"), ("SimCLR", "simclr"))):
        v = [p[f"{key}/{ds}"]["gnorm_repl_over_kept"] for ds in ("c10", "c100")]
        bars = a.bar(x + (j - 1) * w, v, w * 0.92, color=COL[m], label=NAME[m])
        for bx, bv in zip(bars, v):
            a.text(bx.get_x() + bx.get_width() / 2, bv + 0.012, f"{bv:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    a.set_xticks(x); a.set_xticklabels(["CIFAR-10", "CIFAR-100"]); a.set_ylim(0, 0.9); style(a)
    a.set_ylabel("‖g(replaced positives)‖ / ‖g(kept positives)‖")
    a.set_title("(a) weight on wrong positives (ε 0.10, seed 0)", loc="left"); a.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=3)
    # (b) layer profile, CIFAR-100 fine
    t = json.load(open(R / "P141_v7_layer_readout_results.json"))["table"]; sites = ["layer3", "h", "r", "z"]
    for m, key, mk in (("VCS", "A-P3", "o"), ("JS", "JS-AP3", "s"), ("SimCLR", "SimCLR", "^")):
        rows = {r_["site"]: r_["fine:recipe_raw"] for r_ in t if r_["method"] == key}
        mu = [rows[s][0] for s in sites]; sd = [rows[s][1] for s in sites]; n = rows["h"][2]
        b.errorbar(range(4), mu, yerr=sd, color=COL[m], marker=mk, ms=6, lw=2, capsize=2, label=f"{NAME[m]} (n {n})")
        b.annotate(f"{mu[3] - mu[1]:+.1f}", (3, mu[3]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    b.set_xticks(range(4)); b.set_xticklabels(["layer3", "h (backbone)", "r", "z (projector)"]); style(b)
    b.set_ylabel("CIFAR-100 fine linear top-1 (%)"); b.set_xlim(-0.3, 3.55)
    b.set_title("(b) fine-label information by site (z − h noted)", loc="left"); b.legend(frameon=False, fontsize=7.5, loc="lower left")
    # (c) J at h vs z
    h1 = json.load(open(R / "P149_results.json"))["runs"]; h2 = json.load(open(R / "P149A1_results_h_seed2.json"))["runs"]
    z1 = json.load(open(R / "P149A1_results_z_seed1.json"))["runs"]; z2 = json.load(open(R / "P149A1_results_z_seed2.json"))["runs"]
    J = lambda d, k: d[k]["estimators"]["vcs_mlp"]["per_setting"]["identity_t0"]["J_mean"]
    enc = [("C10\nVCS enc.", "P107_AP3_views4_800ep_seed{s}", "VCS"), ("C10\nSimCLR enc.", "P41_simclr_views4_800ep_seed{s}", "SimCLR"),
           ("C100\nVCS enc.", "P107_AP3_c100_views4_800ep_seed{s}", "VCS"), ("C100\nSimCLR enc.", "P91_c100_simclr_views4_800ep_seed{s}", "SimCLR")]
    for i, (lab, run, fam) in enumerate(enc):
        for s, dh, dz, dx in ((1, h1, z1, -0.08), (2, h2, z2, 0.08)):
            k = run.format(s=s); jh, jz = J(dh, k), J(dz, k)
            c.plot([i + dx - 0.12, i + dx + 0.12], [jh, jz], color=COL[fam], lw=1.2, alpha=0.7)
            c.scatter([i + dx - 0.12], [jh], s=36, facecolors="white", edgecolors=COL[fam], linewidths=1.6, zorder=3, label="h (backbone)" if i == 0 and s == 1 else None)
            c.scatter([i + dx + 0.12], [jz], s=42, color=COL[fam], edgecolors="white", linewidths=0.8, zorder=3, label="z (projector)" if i == 0 and s == 1 else None)
    c.set_xticks(range(4)); c.set_xticklabels([e[0] for e in enc], fontsize=8); c.set_ylim(0.7, 1.0); style(c)
    c.set_ylabel("common J of the fitted VCS critic (t 0)")
    c.set_title("(c) critic-fittable dependence: h vs z (seeds 1–2)", loc="left")
    leg = c.legend(frameon=False, fontsize=7.5, loc="lower right")
    for hnd in leg.legend_handles:
        hnd.set_edgecolor(INK2); hnd.set_facecolor("white" if hnd.get_label() == "h (backbone)" else INK2)
    c.text(0.02, 0.03, "fit-limited lower readings; not a\ncross-site DPI statement (W11)", transform=c.transAxes, va="bottom", fontsize=7, color=INK2)
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_mechanism.png", dpi=170); fig.savefig(OUT / "fig_mechanism.svg"); print("wrote", OUT / "fig_mechanism.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
