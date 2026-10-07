"""P143 (MV6-I2) aggregation: Table-4-axis rows (fixed VCS measurement, vary encoder) for every audited encoder seed, CIFAR-10 and CIFAR-100.

Rows: the four existing CIFAR-10 VCS / SimCLR encoders (P118 / P109 records) and the P143 encoders (VICReg, selected logistic on CIFAR-10;
A-P3, SimCLR, VICReg, selected logistic on CIFAR-100).  Per row: H / O rejection (VCS statistic = the Table 4 column; JS statistic in its own
columns), max-T power, both nulls with flags (> 0.09 at R = 200, any layer / statistic), paired effects with intervals, detection n and response n.
    python scripts/p143_aggregate.py --out reports/P143_results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FLAG = 0.09
STRENGTH = {"colour": 0.1, "blur": 0.25}
ROWS = [  # (dataset, display name, run, record prefix)
    *[("CIFAR-10", "VCS", f"P107_AP3_views4_800ep_seed{s}", "P118") for s in (1, 2)],
    *[("CIFAR-10", "SimCLR", f"P41_simclr_views4_800ep_seed{s}", "P118") for s in (1, 2)],
    *[("CIFAR-10", "VICReg", f"P41_vicreg_views4_800ep_seed{s}", "P143") for s in (1, 2)],
    *[("CIFAR-10", "selected logistic (2, 0.25)", f"P129_JS_c10_a2_k0.25_seed{s}", "P143") for s in (1, 2)],
    *[("CIFAR-100", "VCS", f"P107_AP3_c100_views4_800ep_seed{s}", "P143") for s in (1, 2)],
    *[("CIFAR-100", "SimCLR", f"P91_c100_simclr_views4_800ep_seed{s}", "P143") for s in (1, 2)],
    *[("CIFAR-100", "VICReg", f"P91_c100_vicreg_views4_800ep_seed{s}", "P143") for s in (1, 2)],
    *[("CIFAR-100", "selected logistic (3, 0.5)", f"P129_JS_c100_a3_k0.5_seed{s}", "P143") for s in (1, 2)],
]


def paths(prefix: str, run: str, fam: str):
    if prefix == "P118":
        return (REPO / f"reports/P118_i1_{run}_{fam}_power.json", REPO / f"reports/P118_i1_{run}_{fam}_level.json",
                REPO / f"reports/P109_i1_{run}_{fam}_effects.json")
    return tuple(REPO / f"reports/P143/i1_{run}_{fam}_{k}.json" for k in ("power", "level", "effects"))


def row(ds, name, run, prefix, fam):
    pw, lv, ef = (json.load(open(p)) for p in paths(prefix, run, fam))
    s = STRENGTH[fam]
    cell = pw["cells"][f"planted:{s}:2000"]
    assert cell["n"] == 2000 and cell["repeats"] == 100 and cell["perms"] == 200
    nulls = {c["mode"]: c for c in lv["cells"].values()}
    assert all(c["repeats"] == 200 for c in nulls.values())
    e = ef["prediction_effects"][f"{fam}_s{s}"]; ov = e["overall"]
    sm = cell["summary"]
    flags = sorted(f"{m}:{k}" for m in nulls for k, v in nulls[m]["summary"].items() if v["power"] > FLAG)
    return {"comparison_axis": "fixed_measurement_vary_encoder", "dataset": ds, "encoder_paper_name": name, "run": run,
            "encoder_seed": int(run.rsplit("seed", 1)[1]), "record": prefix, "attribute": fam, "strength_native": s,
            "n_detection": 2000, "n_detection_repeats": 100, "paired_response_n_base_images": e["n_eval_base_images"],
            "H_vcs": sm["h/vcs_closed"]["power"], "O_vcs": sm["logits/vcs_closed"]["power"],
            "H_js": sm["h/js_exact"]["power"], "O_js": sm["logits/js_exact"]["power"],
            "maxT_vcs": sm["any_layer_maxT/vcs_closed"]["power"],
            "null_H_vcs_label|planted": "|".join(f"{nulls[m]['summary']['h/vcs_closed']['power']:.3f}" for m in ("null_label_only", "null_all_planted")),
            "null_O_vcs_label|planted": "|".join(f"{nulls[m]['summary']['logits/vcs_closed']['power']:.3f}" for m in ("null_label_only", "null_all_planted")),
            "vcs_column_flag": any(nulls[m]["summary"][f"{l}/vcs_closed"]["power"] > FLAG for m in nulls for l in ("h", "logits")),
            "flags_any": ";".join(flags) or "none",
            "acc_clean": ov["acc_clean"], "d_acc_points": 100 * ov["d_acc"], "d_acc_ci_points": [100 * x for x in ov["d_acc_ci"]],
            "d_prob_true": ov["d_prob_true"], "d_prob_true_ci": ov["d_prob_true_ci"], "flip_percent": 100 * ov["prediction_flip_rate"]}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = [row(*r, fam) for fam in ("colour", "blur") for r in ROWS]
    gate = json.load(open(REPO / "reports/P143/stage0_gate.json"))
    out = {"stage0_gate": {k: gate[k] for k in ("pass", "max_abs_power_diff", "effects")}, "rows": rows}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["# P143 results — fixed VCS measurement, vary encoder (Table 4 axis); per encoder seed", "",
         f"Stage-0 reproduction gate: pass = {gate['pass']}, max |Δpower| {gate['max_abs_power_diff']:.3f}.", ""]
    for fam in ("colour", "blur"):
        L += [f"## {fam} {STRENGTH[fam]} (n 2 000 × 100 draws; responses on 8 000 pairs)", "",
              "| dataset | encoder / seed | H (VCS) | O (VCS) | H / O (JS stat.) | null H / O (VCS) | Δacc pts [95 %] | flips % | Δp_true | clean acc | flags (any) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x["attribute"] == fam):
            L.append(f"| {r['dataset']} | {r['encoder_paper_name']} / {r['encoder_seed']} | {r['H_vcs']:.2f} | {r['O_vcs']:.2f} | {r['H_js']:.2f} / {r['O_js']:.2f} | "
                     f"{r['null_H_vcs_label|planted']} / {r['null_O_vcs_label|planted']} | {r['d_acc_points']:+.2f} [{r['d_acc_ci_points'][0]:+.2f}, {r['d_acc_ci_points'][1]:+.2f}] | "
                     f"{r['flip_percent']:.1f} | {r['d_prob_true']:+.4f} | {100 * r['acc_clean']:.1f} | {r['flags_any']} |")
        L.append("")
    Path(a.out + ".md").write_text("\n".join(L))
    print(f"{len(rows)} rows -> {a.out}.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
