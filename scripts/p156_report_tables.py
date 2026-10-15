"""P156 report tables, generated from the committed aggregate JSON (primary merged grid; the five-point grid as sensitivity) and, for block E,
from the merged cell files (selected rows only).  Writes one markdown file; every number in the report comes from here."""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def fmt(t: dict) -> str:
    t = t.get("stat", t)
    return f"{t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]"


def sig(t: dict) -> str:
    lo, hi = t.get("stat", t)["ci95"]
    return "VCS lower" if hi < 0 else "JS lower" if lo > 0 else "n.s."


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--agg", required=True); ap.add_argument("--grid5", required=True)
    ap.add_argument("--cells", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    A, G = json.loads(Path(a.agg).read_text()), json.loads(Path(a.grid5).read_text()); L = []
    L += ["## T1. Primary family on both grids (Holm over 3)", "", "| test | merged grid (primary) | p Holm | five-point grid (sensitivity) | p Holm |", "|---|---|---|---|---|"]
    for k in A["primary"]:
        x, y = A["primary"][k], G["primary"][k]
        L.append(f"| {k} | {fmt(x)} | {x['p_holm']:.4f} | {fmt(y)} | {y['p_holm']:.4f} |")
    p2s, p2s5 = A["secondary"]["P2_sat_pooled"], G["secondary"]["P2_sat_pooled"]
    L += [f"| P2-sat (secondary) | {fmt(p2s)} | p {p2s['stat']['p']:.4f} | {fmt(p2s5)} | p {p2s5['stat']['p']:.4f} |", ""]
    sm = A["descriptive"]["saturation_model"]
    L += [f"Saturation model (descriptive, {sm['n_cells']} cells, {sm['model']}): " + ", ".join(f"{k} {v:+.4f}" for k, v in sm["coef"].items()), ""]

    sat = A["descriptive"]["saturation"]; L += ["## T2. Block A clean cells (MLP-256): Δe = e_VCS − e_JS (absolute), saturation", "",
                                                "| d | I | S | sat95 | Δe | reading |", "|---|---|---|---|---|---|"]
    for r in sat:
        if r["block"] == "A" and r["ctype"] == "clean":
            L.append(f"| {r['d']} | {r['I']:.0f} | {r['S']:.3f} | {r['sat95']:.3f} | {fmt(r['delta'])} | {sig(r['delta'])} |")
    L.append("")

    gd = A["descriptive"]["gamma_dose"]; L += ["## T3. γ dose: arm with the lowest mean e per cell (counts of cells)", "", "| group | js (γ 0) | b05 (γ ½) | vcs (γ 1) | b2 (γ 2) | cells |", "|---|---|---|---|---|---|"]
    groups = defaultdict(lambda: defaultdict(int))
    for r in gd:
        if set(r["mean_e"]) != {"js", "b05", "vcs", "b2"}:
            continue
        if r["block"] == "A":
            g = "A clean" if r["ctype"] == "clean" else f"A contaminated ε {r['eps']:g}"
        elif r["block"] in ("B", "B2"):
            g = f"{r['block']} matched clean"
        elif r["block"] == "E":
            g = f"E {r['cls']} " + ("clean" if r["eps"] == 0 else f"ε {r['eps']:g}")
        else:
            continue
        groups[g][min(r["mean_e"], key=r["mean_e"].get)] += 1
    for g in sorted(groups):
        c = groups[g]; L.append(f"| {g} | {c['js']} | {c['b05']} | {c['vcs']} | {c['b2']} | {sum(c.values())} |")
    L.append("")

    L += ["## T3b. γ dose in the block A clean cells (MLP-256): mean e by arm", "", "| d | I | js (γ 0) | b05 (γ ½) | vcs (γ 1) | b2 (γ 2) | monotone in γ |", "|---|---|---|---|---|---|---|"]
    for r in gd:
        if r["block"] == "A" and r["ctype"] == "clean":
            m = r["mean_e"]; v = [m["js"], m["b05"], m["vcs"], m["b2"]]
            L.append(f"| {r['d']} | {r['I']:.0f} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {v[3]:.4f} | {'rising' if v == sorted(v) else 'falling' if v == sorted(v, reverse=True) else 'no'} |")
    L.append("")
    # block E from the merged cells
    R = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(Path(a.cells) / "P156_E_*.json")):
        D = json.loads(Path(f).read_text()); c = D["cell"]
        for r in D["rows"]:
            if r["selected"]:
                R[(c["cls"], c["d"], c["I"], c["eps"])][r["arm"]].append((r["readout"]["posterior_mse"], r["lr"], r["lr_edge"], bool(r["select"].get("not_converged"))))
    L += ["## T4. Block E (restricted classes, independent-pair contamination): mean e over 8 seeds; seeds at the top lr / not converged (VCS and JS)", "",
          "| class | d | I | ε | e_JS | e_γ½ | e_VCS | e_γ2 | (e_VCS − e_JS) / clean e_JS | JS top-lr / n.c. | VCS top-lr / n.c. |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(R):
        m = {arm: st.mean(x[0] for x in v) for arm, v in R[key].items()}; clean = R[(key[0], key[1], key[2], 0.0)]
        e0 = st.mean(x[0] for x in clean["js"])
        cnt = lambda arm: f"{sum(x[2] == 'high' for x in R[key][arm])} / {sum(x[3] for x in R[key][arm])}"
        L.append(f"| {key[0]} | {key[1]} | {key[2]:.0f} | {key[3]:g} | {m['js']:.4f} | {m['b05']:.4f} | {m['vcs']:.4f} | {m['b2']:.4f} | {(m['vcs'] - m['js']) / e0:+.3f} | {cnt('js')} | {cnt('vcs')} |")
    edge = defaultdict(lambda: [0, 0])
    for key, arms in R.items():
        if key[0] in ("cos2", "quad3"):
            for arm, v in arms.items():
                edge[(key[0], arm)][0] += sum(x[2] == "high" for x in v); edge[(key[0], arm)][1] += len(v)
    L += ["", "Top-of-grid share after the extension (lr 0.3 selected; addendum 1: no second extension): " +
          "; ".join(f"{c} {arm} {n} / {t}" for (c, arm), (n, t) in sorted(edge.items())), ""]

    L += ["## T5. Blocks C / D / D′ (VCS − JS, absolute Δe)", "", "| block | type | ε | d | I | N / updates | Δe | reading |", "|---|---|---|---|---|---|---|---|"]
    for r in A["descriptive"]["blocks_C_D_Dp"]:
        L.append(f"| {r['block']} | {r['ctype']} ({r['variant']}) | {r['eps']:g} | {r['d']} | {r['I']:.0f} | {r['N']} / {r['updates']} | {fmt(r['delta'])} | {sig(r['delta'])} |")
    L.append("")
    ct = A["descriptive"]["clean_target_vs_contaminated"]
    more = sum(1 for r in ct if r["delta_e_clean_target"]["mean"] < r["delta_e"]["mean"])
    sig_vcs = sum(1 for r in ct if r["delta_e"]["ci95"][1] < 0); sig_vcs_ct = sum(1 for r in ct if r["delta_e_clean_target"]["ci95"][1] < 0)
    sig_js = sum(1 for r in ct if r["delta_e"]["ci95"][0] > 0); sig_js_ct = sum(1 for r in ct if r["delta_e_clean_target"]["ci95"][0] > 0)
    L += [f"## T6. Clean-target diagnostic ({len(ct)} contaminated cells)", "",
          f"- Δe on the clean target more favourable to VCS than Δe on the contaminated target in {more} / {len(ct)} cells.",
          f"- Unadjusted 95 % intervals below 0 (VCS lower): {sig_vcs} cells on e, {sig_vcs_ct} on the clean target; above 0 (JS lower): {sig_js} on e, {sig_js_ct} on the clean target.", ""]
    ss = A["descriptive"]["selection_rule_sensitivity"]; flip = 0; n = 0
    for r in ss:
        a1, b1 = r["delta_native_selection"], r["delta_commonJ_selection"]
        if a1 and b1:
            n += 1; flip += sig(a1) != sig(b1)
    L += [f"## T7. Selection-rule sensitivity: the unadjusted reading (VCS lower / n.s. / JS lower) changes between native-risk and common-J selection in {flip} / {n} cells.", ""]
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
