"""P109 aggregation (results only): X1 measurement table, I1 level / power / prediction-effect tables, I2 pilot table + the frozen 3-criterion rule.

    python scripts/p109_aggregate.py [--out reports/P109_v4_X_I_results]

Inputs (read-only): outputs/P109_X1/<run>.json, reports/P109_i1_<run>_<family>_{power,level,effects}.json, reports/P109_i2_pilot.json.
Readings follow reports/P109_V4_X_I_PREREG_FROZEN_20261001.md; nothing here ranks methods by J.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
FAMILY = {"P35_vcs_a5": "recipe VCS (P35)", "P41_simclr": "tuned SimCLR (P41)", "P104_G2": "G2", "P104_U2": "U2"}
CANDS = ("zero", "cosine", "mlp", "prod_ridge", "rff_ridge")
LAYERS = ("layer3", "h", "z", "logits")
STATS = ("vcs_closed", "js_exact")


def fam_of(run: str) -> str:
    return next(v for k, v in FAMILY.items() if run.startswith(k))


def ms(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return "—"
    return f"{np.mean(x):.4f} ± {np.std(x, ddof=1):.4f}" if len(x) > 1 else f"{x[0]:.4f}"


def x1(anom):
    rows = []
    for f in sorted(glob.glob(str(OUT_ROOT / "P109_X1" / "*.json"))):
        d = json.load(open(f)); s = d["summary"]
        if s.get("status") != "COMPLETED":
            anom.append(f"X1 {d['run']}: status {s.get('status')}")
        r = {"run": d["run"], "family": fam_of(d["run"]), "linear": s.get("linear_val_top1_pct"), "knn": s.get("knn_val_top1_pct"),
             "train_heldout_J": s.get("train_critic_heldout_J")}
        for rep in ("z", "h"):
            R = d["representations"][rep]
            for c in CANDS:
                v = R["candidates"].get(c, {}).get("eval_J")
                r[f"{rep}_{c}"] = v
                if v is not None and not np.isfinite(v):
                    anom.append(f"X1 {d['run']} {rep} {c}: non-finite eval J")
            r[f"{rep}_picked"] = R["picked"]; r[f"{rep}_picked_eval_J"] = R["picked_eval_J"]
            ev = [R["candidates"][c]["eval_J"] for c in CANDS if c != "zero" and c in R["candidates"]]
            r[f"{rep}_spread"] = (min(ev), max(ev))
            r[f"{rep}_train_critic_eval_J"] = R.get("train_critic_eval_J")
        rows.append(r)
    return rows


def i1(anom):
    level, power, effects = [], [], []
    for f in sorted(glob.glob(str(REPO / "reports" / "P109_i1_*.json"))):
        if f.endswith(".partial.json"):
            continue
        m = re.match(r".*P109_i1_(.+)_(colour|blur)_(power|level|effects)\.json$", f)
        if not m:
            anom.append(f"I1 unparsed file {f}"); continue
        run, fam, kind = m.groups(); d = json.load(open(f))
        for cname, c in (d.get("cells") or {}).items():
            row = {"run": run, "family": fam, "cell": cname, "mode": c["mode"], "version": c.get("version"), "n": c["n"], "R": c["repeats"]}
            for L in LAYERS:
                for st in STATS:
                    sm = c["summary"].get(f"{L}/{st}", {})
                    row[f"{L}/{st}"] = sm.get("power"); row[f"{L}/{st}/maxT"] = sm.get("power_maxT")
            for st in STATS:
                row[f"any/{st}"] = c["summary"].get(f"any_layer_maxT/{st}", {}).get("power")
            (level if c["mode"] != "planted" else power).append(row)
        for ver, e in (d.get("prediction_effects") or {}).items():
            o = e["overall"]
            effects.append({"run": run, "family_encoder": fam_of(run), "nuisance": fam, "version": ver, "n_base": e.get("n_eval_base_images"), **o})
    for r in level:
        for k, v in r.items():
            if "/" in k and not k.endswith("maxT") and not k.startswith("any") and v is not None and r["R"] >= 200 and v > 0.09:
                r.setdefault("flags", []).append(k)
    return level, power, effects


def i2(anom):
    d = json.load(open(REPO / "reports" / "P109_i2_pilot.json"))
    rows, verdict = [], {}
    cells = d["cells"]
    for cname, c in cells.items():
        for key, sm in c["summary"].items():
            obj, var, ro = key.split("/")
            rows.append({"cell": cname, "objective": obj, "variant": var, "readout": ro,
                         **{f: sm.get(f, {}).get("mean") for f in ("delta_J", "D_T", "r_BA", "R_orth", "nesting_violations")},
                         **{f + "_sd": sm.get(f, {}).get("sd_refit") for f in ("delta_J", "R_orth")}})
    null = [k for k in cells if k.endswith("/s0")]; ns = [k for k in cells if k not in null]
    get = lambda cell, obj, var, ro, f: cells[cell]["summary"][f"{obj}/{var}/{ro}"][f]["mean"]
    for obj in ("vcs", "js"):
        for var in ("indep_sampled", "indep_exact", "nested_sampled", "nested_exact"):
            for ro in ("readout_sampled", "readout_exact"):
                if f"{obj}/{var}/{ro}" not in cells[ns[0]]["summary"]:
                    continue
                c1 = all(abs(get(c, obj, var, ro, "R_orth")) <= 0.5 * abs(get(c, obj, "indep_sampled", "readout_sampled", "R_orth")) for c in ns)
                c2 = all(get(c, obj, var, ro, "nesting_violations") < get(c, obj, "indep_sampled", "readout_sampled", "nesting_violations") for c in ns)
                c3 = all(abs(get(c, obj, var, ro, "delta_J")) <= abs(get(c, obj, "indep_sampled", "readout_sampled", "delta_J")) for c in null)
                verdict[f"{obj}/{var}/{ro}"] = {"R_orth_halved": c1, "fewer_nesting_violations": c2, "null_floor_not_larger": c3,
                                                "verdict": "error reduced" if (c1 and c2 and c3) else "no clear reduction"}
    return rows, verdict, ns, null


def fmt(v, p=3):
    return "—" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/P109_v4_X_I_results"); a = ap.parse_args()
    anom: list[str] = []
    X = x1(anom); LV, PW, EF = i1(anom); I2, V, ns, null = i2(anom)
    L = ["# P109 results (v4 X1 / I1 / I2) — aggregated by scripts/p109_aggregate.py (results only)", ""]
    L += ["## X1 — EVAL J of each measurement candidate (fitted on FIT, picked on TUNE; unclipped), training critic on the same pairs, accuracy", "",
          "| run | rep | zero | cosine | mlp | prod_ridge | rff_ridge | picked (EVAL J) | training critic EVAL J | training held-out J | linear | kNN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in X:
        for rep in ("z", "h"):
            L.append(f"| {r['run']} | {rep} | " + " | ".join(fmt(r[f'{rep}_{c}'], 4) for c in CANDS)
                     + f" | {r[f'{rep}_picked']} ({r[f'{rep}_picked_eval_J']:.4f}) | {fmt(r.get(f'{rep}_train_critic_eval_J'), 4)} | "
                     + (f"{fmt(r['train_heldout_J'], 4)} | {r['linear']:.2f} | {r['knn']:.2f} |" if rep == "z" else "| | |"))
    L += ["", "Family mean ± sd over seeds:", "", "| family | n | picked J (z) | picked J (L2 h) | training critic J (z) | candidate spread z (min–max, non-zero) | linear | kNN |",
          "|---|---|---|---|---|---|---|---|"]
    for fam in FAMILY.values():
        rr = [r for r in X if r["family"] == fam]
        if not rr:
            continue
        sp = f"{min(r['z_spread'][0] for r in rr):.4f}–{max(r['z_spread'][1] for r in rr):.4f}"
        L.append(f"| {fam} | {len(rr)} | {ms([r['z_picked_eval_J'] for r in rr])} | {ms([r['h_picked_eval_J'] for r in rr])} | "
                 f"{ms([r.get('z_train_critic_eval_J') for r in rr])} | {sp} | {ms([r['linear'] for r in rr])} | {ms([r['knn'] for r in rr])} |")
    L += ["", "## I1 — level (null cells; flag > 0.09 at R = 200)", "", "| run | family | cell | n | R | " + " | ".join(f"{l}/{s}" for l in LAYERS for s in STATS) + " | any(maxT) vcs / js | flags |",
          "|---|---|---|---|---|" + "---|" * (len(LAYERS) * len(STATS)) + "---|---|"]
    for r in LV:
        L.append(f"| {r['run']} | {r['family']} | {r['cell']} | {r['n']} | {r['R']} | " + " | ".join(fmt(r[f'{l}/{s}'], 3) for l in LAYERS for s in STATS)
                 + f" | {fmt(r['any/vcs_closed'])} / {fmt(r['any/js_exact'])} | {', '.join(r.get('flags', [])) or '—'} |")
    L += ["", "## I1 — power (raw p / max-T adjusted across the 4 layers), R = 100", "",
          "| run | version | n | " + " | ".join(f"{l} vcs / js (raw; maxT)" for l in LAYERS) + " | any layer (maxT) vcs / js |", "|---|---|---|" + "---|" * len(LAYERS) + "---|"]
    for r in PW:
        L.append(f"| {r['run']} | {r['version']} | {r['n']} | " + " | ".join(
            f"{fmt(r[f'{l}/vcs_closed'],2)} / {fmt(r[f'{l}/js_exact'],2)}; {fmt(r[f'{l}/vcs_closed/maxT'],2)} / {fmt(r[f'{l}/js_exact/maxT'],2)}" for l in LAYERS)
            + f" | {fmt(r['any/vcs_closed'],2)} / {fmt(r['any/js_exact'],2)} |")
    L += ["", "## I1 — prediction effects (clean vs planted version of the same EVAL base image; 95 % bootstrap CI over base images)", "",
          "| run | version | Δ true-class prob [CI] | Δ margin [CI] | acc clean → planted | Δ acc [CI] | flip rate |", "|---|---|---|---|---|---|---|"]
    for e in EF:
        L.append(f"| {e['run']} | {e['version']} | {e['d_prob_true']:+.4f} [{e['d_prob_true_ci'][0]:+.4f}, {e['d_prob_true_ci'][1]:+.4f}] | "
                 f"{e['d_margin']:+.3f} [{e['d_margin_ci'][0]:+.3f}, {e['d_margin_ci'][1]:+.3f}] | {e['acc_clean']:.4f} → {e['acc_planted']:.4f} | "
                 f"{e['d_acc']:+.4f} [{e['d_acc_ci'][0]:+.4f}, {e['d_acc_ci'][1]:+.4f}] | {e['prediction_flip_rate']:.4f} |")
    L += ["", "## I2 — pilot (mean over 5 repeats; refit sd in parentheses)", "",
          "| cell | objective | variant | readout | ΔJ (sd) | D_T | r_BA | R_orth (sd) | nesting violations |", "|---|---|---|---|---|---|---|---|---|"]
    for r in I2:
        L.append(f"| {r['cell']} | {r['objective']} | {r['variant']} | {r['readout']} | {r['delta_J']:+.4f} ({r['delta_J_sd']:.4f}) | {r['D_T']:.4f} | "
                 f"{r['r_BA']:+.4f} | {r['R_orth']:+.4f} ({r['R_orth_sd']:.4f}) | {r['nesting_violations']:.1f} |")
    L += ["", f"Frozen rule vs `indep_sampled` + sampled readout (non-saturated cells {ns}; null cell {null}):", "",
          "| objective / variant / readout | |R_orth| ≥ 50 % smaller (both cells) | fewer nesting violations (both) | null |ΔJ| not larger | verdict |", "|---|---|---|---|---|"]
    for k, v in V.items():
        L.append(f"| {k} | {v['R_orth_halved']} | {v['fewer_nesting_violations']} | {v['null_floor_not_larger']} | **{v['verdict']}** |")
    L += ["", "## Anomalies", ""] + ([f"- {x}" for x in anom] or ["- none"])
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    Path(a.out + ".json").write_text(json.dumps({"X1": X, "I1_level": LV, "I1_power": PW, "I1_effects": EF, "I2": I2, "I2_rule": V, "anomalies": anom}, indent=1, default=str))
    print(f"-> {a.out}.md / .json  (X1 {len(X)} encoders, I1 level {len(LV)} / power {len(PW)} / effects {len(EF)}, I2 {len(I2)} rows, anomalies {len(anom)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
