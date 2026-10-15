"""REV9 / REV9-FULL report tables from the committed results JSON (rev9_score.py output) and, as descriptive context, the uniform-grid rows of
the frozen D1-30 results on the same item sets.  Every number in the REV9 reports comes from here."""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

BUDGETS = (72, 144, 288, 432)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--d1-30", required=True)
    ap.add_argument("--title", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = json.loads(Path(a.res).read_text()); D = json.loads(Path(a.d1_30).read_text())["benches"]; L = [f"# {a.title}", ""]
    L += ["## Guard (I0, seed 0 of each objective)", "", "| objective | estimator max abs diff | gate max abs diff | passed |", "|---|---|---|---|"]
    for t, g in R["guard"].items():
        if isinstance(g, dict):
            L.append(f"| {t} | {g.get('max_abs_diff_estimator_vs_frozen')} | {g.get('max_abs_diff_gate_vs_frozen')} | "
                     f"{g.get('passed', g.get('max_abs_diff_estimator_vs_frozen') == 0 and (g.get('max_abs_diff_gate_vs_frozen') or 0) <= 1e-3)} |")
    issues = sum(v.get("dropped_lines", 0) + v.get("blank_lines", 0) + v.get("duplicate_qids", 0) for v in R["integrity"].values())
    L += ["", f"Prediction-file integrity: {len(R['integrity'])} files, {issues} dropped / blank / duplicate lines.", ""]
    summary = []
    for t, rows in R["primary"].items():
        L += [f"## {t}: accuracy (%), mean over seeds; differences are seed means of per-item paired differences (image-grouped bootstrap, Holm over 24)", "",
              "| bench | budget | frozen | I1 − frozen | p Holm | I2 − frozen | p Holm | I2 − I1 | uniform grid | frozen − grid | I1 − grid | I2 − grid |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        by = {(r["bench"], r["cell"].split("@")[1], r["contrast"]): r for r in rows}; n_sig = defaultdict(lambda: [0, 0])
        for b in R["benches"]:
            for k in BUDGETS:
                c = R["benches"][b].get(f"{t}@{k}")
                if not c:
                    continue
                fr, i1, i2 = st.mean(c["frozen"]), by[(b, str(k), "I1-frozen")], by[(b, str(k), "I2-frozen")]
                g = D[b]["fixed_budget"].get(f"static:grid:{k}", {}).get("quality"); i2m = st.mean(c["I2"]); i1m = st.mean(c["I1"])
                d21 = c.get("I2-I1", {}); d21s = f"{d21['delta']:+.2f} [{d21['ci95'][0]:+.2f}, {d21['ci95'][1]:+.2f}]" if d21 else "—"
                for iv, r in (("I1", i1), ("I2", i2)):
                    if r["p_holm"] < 0.05:
                        n_sig[iv][0 if r["delta"] > 0 else 1] += 1
                L.append(f"| {b} | {k} | {fr:.2f} | {i1['delta']:+.2f} [{i1['ci95'][0]:+.2f}, {i1['ci95'][1]:+.2f}] | {i1['p_holm']:.3f} | "
                         f"{i2['delta']:+.2f} [{i2['ci95'][0]:+.2f}, {i2['ci95'][1]:+.2f}] | {i2['p_holm']:.3f} | {d21s} | "
                         f"{g:.2f} | {fr - g:+.2f} | {i1m - g:+.2f} | {i2m - g:+.2f} |" if g is not None else f"| {b} | {k} | {fr:.2f} | ... | | | | | — | | |")
        L.append(""); summary.append((t, {iv: tuple(v) for iv, v in n_sig.items()}))
    L += ["## Holm-significant contrasts (p Holm < 0.05): (positive, negative) of 12 per intervention", ""]
    for t, s in summary:
        L.append(f"- {t}: I1 {s.get('I1', (0, 0))}, I2 {s.get('I2', (0, 0))}")
    L += ["", "## Soft-hard gap of the gate (mean over the units that have it; soft − hard selection loss, logged at the end of phase B)", "",
          "| objective | arm | " + " | ".join(f"budget {k}" for k in BUDGETS) + " |", "|---|---|" + "---|" * len(BUDGETS)]
    acc = defaultdict(lambda: defaultdict(list))
    for unit, arms in R["gaps"].items():
        t = unit.rsplit("_s", 1)[0]
        for arm, per in arms.items():
            for k in BUDGETS:
                if str(k) in per and per[str(k)] is not None:
                    acc[(t, arm)][k].append(per[str(k)]["gap"])
    for (t, arm), per in sorted(acc.items()):
        L.append(f"| {t} | {arm} | " + " | ".join(f"{st.mean(per[k]):+.4f} (n {len(per[k])})" if per[k] else "—" for k in BUDGETS) + " |")
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
