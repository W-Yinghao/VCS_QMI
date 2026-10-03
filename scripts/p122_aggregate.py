"""P122 aggregate: one table per dataset × measurement law — EVAL J per level (train / s / Z / H) for each loss, the selected candidates, and the
paired increments J_s − J_train, J_Z − J_s, J_H − J_Z with 95 % bootstrap intervals over pair units; plus cost and the density diagnostics.
Wording: "additional score recovered in these function classes and budgets", never true information terms.

    python scripts/p122_aggregate.py --in outputs/P122_evidence --out reports/P122_v6_evidence_results
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_ssl.utils import atomic_write_json  # noqa: E402


def fmt_inc(d):
    return f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else "—"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    recs = [json.load(open(f)) for f in sorted(glob.glob(str(Path(a.inp) / "*.json"))) if not f.endswith(".partial.json")]
    L = ["# P122 — v6 V6-EVIDENCE results (frozen encoders, common measurement augmentation; seed-0 encoders)", "",
         "Increments are additional score recovered in these function classes and budgets (1000 updates, 3 lrs × 3 inits, TUNE selection), not true terms.",
         "Unit = pair index (one P and one Q per anchor); 95 % bootstrap intervals over EVAL units.", ""]
    rows = []
    for (ds, law) in sorted({(r["dataset"], r["measurement_law"]) for r in recs}):
        L += [f"## {ds} — measurement law: common-{law}", "",
              "| encoder (train aug) | loss | J train | J s | J Z | J H | s − train | Z − s | H − Z | selected s / Z residual / H residual | min |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in recs if x["dataset"] == ds and x["measurement_law"] == law]:
            crop = r["train_augmentation"]["crop_scale"][0]
            for loss, d in r["losses"].items():
                e, inc = d["eval"], d["increments"]
                zsel, hsel = d["levels"]["Z"]["selected"], d["levels"]["H"]["selected"]
                row = {"dataset": ds, "law": law, "run": r["run"], "train_crop_min": crop, "loss": loss,
                       **{f"J_{k}": v["J"] for k, v in e.items()}, **{k: v for k, v in inc.items()},
                       "selected_s": d["levels"]["s"]["selected"], "Z_residual": zsel.get("residual"), "H_residual": hsel.get("residual"),
                       "minutes": r["seconds"]["total"] / 60, "density_diag": d.get("density_diag", {})}
                rows.append(row)
                L.append(f"| {r['run']} (crop≥{crop}) | {loss} | {e['train']['J']:.4f}" if "train" in e else f"| {r['run']} (crop≥{crop}) | {loss} | —")
                L[-1] += (f" | {e['s']['J']:.4f} | {e['Z']['J']:.4f} | {e['H']['J']:.4f} | {fmt_inc(inc.get('s_minus_train'))} | {fmt_inc(inc['Z_minus_s'])} | "
                          f"{fmt_inc(inc['H_minus_Z'])} | {d['levels']['s']['selected']} / {zsel.get('residual')} / {hsel.get('residual')} | {r['seconds']['total'] / 60:.1f} |")
        L.append("")
    L += ["## Density-ratio consistency (necessary-condition diagnostic only; f = 0 also gives 0)", "",
          "| dataset | law | encoder | loss | level | global log E_Q e^{2f} | per-anchor mean (sd) | per-anchor ESS |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        for lv, dd in row["density_diag"].items():
            L.append(f"| {row['dataset']} | {row['law']} | {row['run']} | {row['loss']} | {lv} | {dd['global_log_mean_exp_2f_Q']:+.3f} | "
                     f"{dd['per_anchor_mean']:+.3f} ({dd['per_anchor_sd']:.3f}) | {dd['per_anchor_ess_mean']:.1f} |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n"); atomic_write_json(Path(a.out + ".json"), {"rows": rows, "n_files": len(recs)})
    print(f"-> {a.out}.md ({len(recs)} files, {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
