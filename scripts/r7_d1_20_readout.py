"""r7 D1-20 CAL-only readout: the selector units' CAL records (DEV sealed until D1-30).  Reads outputs/R7_D1/selectors/*.json and lr_choice.json,
writes a markdown table set + JSON.  Descriptive; nothing here selects anything (selection already happened inside each unit by the frozen rule).
  python -I scripts/r7_d1_20_readout.py --selectors DIR --out-prefix reports/R7/R7_D1_20_CAL_READOUT
"""
import argparse, json, statistics as st
from pathlib import Path


def unit(path):
    d = json.loads(path.read_text()); n = d.get("native_selected") or {}
    dyn = {k: round(v.get("mean_kept_fraction", float("nan")), 3) for k, v in (d.get("dynamic") or {}).items()}
    gap = d.get("soft_hard_gap") or {}
    return {"tag": path.stem, "objective": d.get("objective"), "selected_step": d.get("selected_step"),
            "cal_spearman": d.get("selected_cal_spearman"), "native_criterion": n.get("criterion"), "native_step": n.get("step"),
            "native_cal_spearman": n.get("cal_spearman_r_h"), "soft_hard_gap": gap, "dynamic_kept": dyn,
            "best_eval_spearman": max(e["cal_spearman_r_h"] for e in d["evals"]), "hours": round(d.get("seconds", 0) / 3600, 2),
            "gpu": (d.get("env") or {}).get("gpu") or (d.get("env") or {}).get("device")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selectors", required=True); ap.add_argument("--out-prefix", required=True); a = ap.parse_args()
    S = Path(a.selectors); choice = json.loads((S / "lr_choice.json").read_text())
    U = {p.stem: unit(p) for p in sorted(S.glob("*.json")) if not p.stem.startswith("lr_choice") and not p.name.endswith(".done.json")}
    fmt = lambda x: "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    L = ["# r7 D1-20 — CAL-only readout (DEV sealed until D1-30)", "",
         "Every number is from CAL (2,000 images; 95,868 question × mask pairs); the frozen selection rule (max CAL Spearman between the",
         "selector's risk score r(K) and the realised teacher h) already chose each unit's checkpoint.  The native criterion is a pre-declared",
         "secondary, never used for selection.", "", "## 1. Learning-rate search (seed 0) and the frozen choice", "",
         "| objective | lr | selected step | CAL Spearman(r, h) | native-selected step | its Spearman |", "|---|---|---|---|---|---|"]
    for obj in ("vcs", "logistic", "direct"):
        for lr in ("0.0001", "0.0003"):
            u = U.get(f"{obj}_lr{lr}_s0")
            if u: L.append(f"| {obj} | {lr} | {u['selected_step']} | {fmt(u['cal_spearman'])} | {u['native_step']} | {fmt(u['native_cal_spearman'])} |")
    L += ["", "Frozen choice (lr_choice.json; the frozen planner's rule): " + "; ".join(f"{o} {choice[o]['chosen_lr']:g} (CAL Spearman "
          + ", ".join(f"{k}: {v:.4f}" for k, v in choice[o]['cal_spearman'].items()) + ")" for o in ("vcs", "logistic", "direct")), "",
          "## 2. Main units at the chosen rates (seeds 0–2)", "",
          "| unit | selected step | CAL Spearman(r, h) | native-selected Spearman | best eval Spearman | dynamic mean kept (0.25 / 0.5 / 0.75) | soft − hard gap at budgets 72 / 144 / 288 / 432 | h |",
          "|---|---|---|---|---|---|---|---|"]
    groups = {}
    def row(t):
        u = U[t]; g = u["soft_hard_gap"]; gs = " / ".join(f"{float(v['gap']):+.3f}" if isinstance(v, dict) and 'gap' in v else '—' for k, v in sorted(g.items(), key=lambda kv: float(kv[0]))[:4]) if g else "—"
        L.append(f"| {t} | {u['selected_step']} | {fmt(u['cal_spearman'])} | {fmt(u['native_cal_spearman'])} | {fmt(u['best_eval_spearman'])} | "
                 f"{' / '.join(str(u['dynamic_kept'].get(k, '—')) for k in ('0.25', '0.5', '0.75'))} | {gs} | {u['hours']} |")
    for obj in ("vcs", "logistic", "direct"):
        lr = choice[obj]["chosen_lr"]
        tags = [f"{obj}_lr{lr:.4f}_s{s}" for s in (0, 1, 2) if f"{obj}_lr{lr:.4f}_s{s}" in U]
        for t in tags: row(t)
        groups[obj] = [U[t]["cal_spearman"] for t in tags]
    L += ["", "## 3. Ablations (seeds 0–2)", "", "| unit | selected step | CAL Spearman(r, h) | native-selected Spearman | best eval Spearman | dynamic mean kept | soft − hard gap at budgets 72 / 144 / 288 / 432 | h |",
          "|---|---|---|---|---|---|---|---|"]
    for suf, name in (("_noq", "A1 no question (vcs)"), ("_dtarget", "A2 target d (direct)")):
        tags = sorted(t for t in U if t.endswith(suf))
        for t in tags: row(t)
        groups[name] = [U[t]["cal_spearman"] for t in tags]
    L += ["", "## 4. Summary (CAL Spearman of the selected checkpoint, mean ± sd over seeds)", "", "| arm | n | mean | sd | values |", "|---|---|---|---|---|"]
    for k, v in groups.items():
        if v: L.append(f"| {k} | {len(v)} | {st.mean(v):.4f} | {st.stdev(v) if len(v) > 1 else 0:.4f} | {', '.join(f'{x:.4f}' for x in v)} |")
    Path(a.out_prefix + ".md").write_text("\n".join(L) + "\n")
    Path(a.out_prefix + ".json").write_text(json.dumps({"choice": choice, "units": U, "groups": groups}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
