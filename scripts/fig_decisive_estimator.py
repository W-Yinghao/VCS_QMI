"""The decisive estimator figure (O1 §17.5): estimator precision / stability / gradient variance vs dependence strength, from the P151 panels
(reports/P151_E0/*.json: P85 / P86 records re-aggregated) and the P152 recorded fits (reports/P152_results.json).  Writes PNG + SVG under
reports/figures/ (never PDF).  Same-target methods are compared on posterior MSE for S; MI estimators on their own targets (|error| to the
analytic MI, with the in-batch InfoNCE bound log 1024 shown).  Analysis only.
    python scripts/fig_decisive_estimator.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]; P151 = REPO / "reports" / "P151_E0"; OUT = REPO / "reports" / "figures"
prec = json.load(open(P151 / "e_precision.json")); stab = json.load(open(P151 / "e_stability.json")); reso = json.load(open(P151 / "e_resolution.json"))
p152 = json.load(open(REPO / "reports" / "P152_results.json"))["table"]
LEVELS = ["2", "4", "6", "8", "10"]
SAME = {"neural:vcs:inbatch": ("VCS (bounded quadratic)", "C3", "-", "o"), "neural:js:inbatch": ("matched JS (same posterior)", "C0", "--", "s"),
        "s_kernel:rff": ("RFF ridge-tanh (kernel)", "C2", ":", "^"), "rls:tanh": ("rLS-tanh (kernel)", "C8", ":", "v"),
        "s_kde:common_risk": ("S-KDE", "C7", ":", "x"), "s_kernel:nystrom:c512": ("Nyström", "C5", ":", "d")}
NATIVE = {"neural:infonce:inbatch": ("InfoNCE", "C1", "-", "o"), "neural:nwj:inbatch": ("NWJ", "C4", "-", "s"),
          "neural:dv:inbatch": ("DV / MINE", "C6", "-", "^"), "neural:smile:inbatch": ("SMILE (corr.)", "C9", "-", "d")}
P152K = {"vcs": ("VCS", "C3"), "js": ("matched JS", "C0"), "infonce": ("InfoNCE", "C1"), "nwj": ("NWJ", "C4"), "dv": ("DV / MINE", "C6"), "smile": ("SMILE", "C9")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2)); g = prec["gaussian"]; S = [g[l]["S"] for l in LEVELS]; I = [int(l) for l in LEVELS]
    # (a) same-target precision
    a = ax[0, 0]
    for k, (lab, c, ls, mk) in SAME.items():
        m = [g[l]["same_target"][k]["posterior_mse"]["mean"] for l in LEVELS]; sd = [g[l]["same_target"][k]["posterior_mse"]["sd"] for l in LEVELS]
        a.errorbar(S, m, yerr=sd, label=lab, color=c, ls=ls, marker=mk, capsize=2)
    a.set_yscale("log"); a.set_xlabel("dependence S (Gaussian ladder, MI 2…10 nats)"); a.set_ylabel("posterior MSE  E_M (T − η)²  (3 seeds)")
    a.set_title("(a) same target S: precision"); a.legend(fontsize=7)
    # (b) native targets
    b = ax[0, 1]
    for k, (lab, c, ls, mk) in NATIVE.items():
        v = [g[l]["native_target"][k]["value"]["mean"] for l in LEVELS]; sd = [g[l]["native_target"][k]["value"]["sd"] for l in LEVELS]
        b.errorbar(I, v, yerr=sd, label=lab, color=c, ls=ls, marker=mk, capsize=2)
    b.plot(I, I, color="k", lw=0.8, ls="--", label="true MI"); b.axhline(np.log(1024), color="C1", lw=0.6, ls=":", label="InfoNCE bound log 1024")
    b.set_xlabel("true MI (nats)"); b.set_ylabel("estimate (nats), in-batch negatives"); b.set_title("(b) MI estimators on their own targets"); b.legend(fontsize=7)
    # (c) seed spread of the value (relative)
    c = ax[0, 2]; st = stab["gaussian"]
    for k, (lab, col, ls, mk) in {**SAME, **NATIVE}.items():
        if k not in st[LEVELS[0]]:
            continue
        rel = []
        for l in LEVELS:
            e = st[l][k]; val = g[l]["same_target"][k]["J_eval"]["mean"] if k in g[l]["same_target"] else g[l]["native_target"][k]["value"]["mean"]
            rel.append(e["seed_sd_value"] / abs(val) if val else np.nan)
        c.plot(I, rel, label=lab, color=col, ls=ls, marker=mk)
    c.set_yscale("log"); c.set_xlabel("true MI (nats)"); c.set_ylabel("seed sd / |value|"); c.set_title("(c) value spread over fit seeds"); c.legend(fontsize=6)
    # (d) gradient-norm CV (P152)
    d = ax[1, 0]; I3 = [2, 6, 10]
    for k, (lab, col) in P152K.items():
        m = [p152[f"{k}/I{i}"]["gnorm_last_half_cv"]["mean"] for i in I3]; sd = [p152[f"{k}/I{i}"]["gnorm_last_half_cv"]["sd"] or 0 for i in I3]
        d.errorbar(I3, m, yerr=sd, label=lab, color=col, marker="o", capsize=2)
    d.set_xlabel("true MI (nats)"); d.set_ylabel("CV of critic gradient norm (last 1000 updates)"); d.set_title("(d) gradient-norm variability (P152, 3 seeds)"); d.legend(fontsize=7)
    # (e) evaluation variance conditional on the fit (relative) + saturation
    e = ax[1, 1]
    for k, (lab, col) in P152K.items():
        m = [p152[f"{k}/I{i}"]["checkpoints"]["select_best"]["eval_block_rel_sd"]["mean"] for i in I3]
        e.plot(I3, m, label=lab, color=col, marker="o")
    e.set_yscale("log"); e.set_xlabel("true MI (nats)"); e.set_ylabel("sd over 64 independent EVAL blocks / |value|"); e.set_title("(e) evaluation variance given the fit"); e.legend(fontsize=7)
    # (f) resolution of adjacent levels (standardised gap), Gaussian and xor
    f = ax[1, 2]; steps = ["2->4", "4->6", "6->8", "8->10"]; x = np.arange(len(steps)); w = 0.13
    keys = ["neural:vcs:inbatch", "neural:js:inbatch", "neural:infonce:inbatch", "neural:nwj:inbatch", "neural:dv:inbatch", "neural:smile:inbatch"]
    for j, k in enumerate(keys):
        lab, col = ({**SAME, **NATIVE}[k][0], {**SAME, **NATIVE}[k][1])
        gz = [reso["gaussian"][s]["methods"][k]["standardised"] for s in steps]; xz = [reso["xor_mixture"][s]["methods"][k]["standardised"] if k in reso["xor_mixture"][s]["methods"] else np.nan for s in steps]
        f.bar(x + (j - 2.5) * w, gz, w, color=col, label=lab)
        f.scatter(x + (j - 2.5) * w, xz, color="k", marker="x", s=22, zorder=3, label="xor ladder (×)" if j == 0 else None)
    f.set_yscale("log"); f.axhline(2, color="k", lw=0.6, ls=":"); f.set_xticks(x); f.set_xticklabels([s.replace("->", "→") for s in steps])
    f.set_xlabel("adjacent MI levels (nats)"); f.set_ylabel("|ΔJ| / pooled sd  (bars Gaussian, × xor)"); f.set_title("(f) resolution of adjacent levels"); f.legend(fontsize=6)
    fig.suptitle("Bounded quadratic estimator vs matched JS, kernel routes and MI estimators — synthetic ladders with oracles (P85/P86 → P151; P152)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_decisive_estimator.png", dpi=170); fig.savefig(OUT / "fig_decisive_estimator.svg"); print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
