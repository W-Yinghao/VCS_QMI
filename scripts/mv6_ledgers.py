"""MV6-R0 / MV6-I1 ledgers (CVPR-MV6-HANDOFF r2): rebuild the two comparison-axis tables from the repository's own records.

  measurement_comparison.csv  — fixed encoder, vary estimator (manuscript Table 3; P116 fixture), parsed from reports/P116_results.md
                                (the P116 aggregate written from reports/P116/*.json).
  ssl_retention_response.csv  — fixed VCS measurement, vary encoder (Table 4 + supplement F blur; P109/P118 fixture), read from the raw
                                reports/P118_i1_<run>_<fam>_{power,level}.json and reports/P109_i1_<run>_<fam>_effects.json.

No fitting, no data access, no GPU.  Values are copied, not re-estimated; rounding is left to the presentation layer.
    python scripts/mv6_ledgers.py --out reports/manuscript_v6_handoff/tables
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO.parent / "outputs"
ALIAS = {"P107_AP3_views4_800ep": "VCS", "P41_simclr_views4_800ep": "SimCLR"}
T4_RUNS = [f"{base}_seed{s}" for base in ALIAS for s in (1, 2)]
STRENGTH = {"colour": 0.1, "blur": 0.25}
FLAG = 0.09  # P118 aggregate rule: any raw per-layer null power > 0.09 at R = 200


def sha16(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def table4(out: Path) -> list[dict]:
    rows = []
    for run in T4_RUNS:
        base, seed = run.rsplit("_seed", 1)
        ck = sha16(OUTPUTS / run / "checkpoints" / "epoch_800.pt")
        for fam, s in STRENGTH.items():
            pw = json.load(open(REPO / f"reports/P118_i1_{run}_{fam}_power.json"))
            lv = json.load(open(REPO / f"reports/P118_i1_{run}_{fam}_level.json"))
            ef = json.load(open(REPO / f"reports/P109_i1_{run}_{fam}_effects.json"))
            cell = pw["cells"][f"planted:{s}:2000"]
            assert cell["version"] == f"{fam}_s{s}" and cell["n"] == 2000 and cell["repeats"] == 100, cell["version"]
            nulls = {c["mode"]: c for c in lv["cells"].values()}
            e = ef["prediction_effects"][f"{fam}_s{s}"]
            ov = e["overall"]
            for est in ("vcs_closed", "js_exact"):
                hn = [nulls[m]["summary"][f"h/{est}"]["power"] for m in ("null_label_only", "null_all_planted")]
                on = [nulls[m]["summary"][f"logits/{est}"]["power"] for m in ("null_label_only", "null_all_planted")]
                anyflag = sorted(k for m in nulls for k, v in nulls[m]["summary"].items() if v["power"] > FLAG)
                rows.append({
                    "comparison_axis": "fixed_measurement_vary_encoder",
                    "matched_condition_id": f"C10_{fam}{s}_n2000x100_resp8000",
                    "fixture_id": "F-PAIRED-P109/P118",
                    "paired_response_fixture_id": "F-PAIRED-P109/P118:EVAL8000",
                    "dataset": "CIFAR-10",
                    "encoder_paper_name": ALIAS[base],
                    "encoder_seed": int(seed),
                    "encoder_checkpoint_sha256": ck,
                    "measurement_protocol_id": "P109-I1/P118 conditional test (closed_form_critic | exact_js_critic), "
                                               "exact-Q, B=200 within-class perms, alpha .05",
                    "estimator_method": {"vcs_closed": "VCS", "js_exact": "matched_logistic"}[est],
                    "attribute": fam,
                    "strength_native": s,
                    "n_detection": cell["n"],
                    "n_detection_repeats": cell["repeats"],
                    "paired_response_n_base_images": e["n_eval_base_images"],
                    "h_rejection_rate": cell["summary"][f"h/{est}"]["power"],
                    "output_rejection_rate": cell["summary"][f"logits/{est}"]["power"],
                    "h_null_rejection_label_only|all_planted": "|".join(f"{x:.3f}" for x in hn),
                    "output_null_rejection_label_only|all_planted": "|".join(f"{x:.3f}" for x in on),
                    "h_null_flag": any(x > FLAG for x in hn),
                    "output_null_flag": any(x > FLAG for x in on),
                    "any_null_flag_in_family(all layers, both statistics)": ";".join(anyflag) or "none",
                    "null_all_planted_version": nulls["null_all_planted"]["version"],
                    "h_power_maxT": cell["summary"]["any_layer_maxT/" + est]["power"],
                    "delta_accuracy_points": 100 * ov["d_acc"],
                    "delta_accuracy_points_ci": "|".join(f"{100 * x:.3f}" for x in ov["d_acc_ci"]),
                    "delta_true_probability": ov["d_prob_true"],
                    "delta_true_probability_ci": "|".join(f"{x:.4f}" for x in ov["d_prob_true_ci"]),
                    "flip_percent": 100 * ov["prediction_flip_rate"],
                    "analysis_head_sha256": None,  # stored in the deleted P109_I1_features cache; not recoverable from JSON
                    "source_ref": f"reports/P118_i1_{run}_{fam}_power.json; reports/P118_i1_{run}_{fam}_level.json; "
                                  f"reports/P109_i1_{run}_{fam}_effects.json",
                })
    with open(out / "ssl_retention_response.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


SEC = re.compile(r"^## (\S+) — (\S+), n (\d+), R (\d+) \((\S+)\)")


def table3(out: Path) -> list[dict]:
    lines = (REPO / "reports/P116_results.md").read_text().splitlines()
    keep_cells = {("full_simclr_blur", "blur_s0.5", 1000), ("full_simclr_blur", "blur_s0.5", 2000),
                  ("full_simclr_colour", "colour_s0.2", 2000),
                  ("level_simclr", "colour_null_label_only", 2000), ("level_simclr", "colour_null_all_planted0.2", 2000)}
    keep_algo = {"A1": ("VCS", "ridge_tanh_calibrated (4 ridge solves, 25-point VAL scale)"),
                 "A3@50": ("matched_logistic", "L-BFGS, 50 closures"),
                 "A3@200": ("matched_logistic", "L-BFGS, 200 closures"),
                 "A3@800": ("matched_logistic", "L-BFGS, 800 closures")}
    rows, sec = [], None
    for i, ln in enumerate(lines, 1):
        m = SEC.match(ln)
        if m:
            sec = (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5))
            continue
        if not sec or (sec[0], sec[1], sec[2]) not in keep_cells or not ln.startswith("| A"):
            continue
        c = [x.strip() for x in ln.strip("|").split("|")]
        if c[0] not in keep_algo:
            continue
        est, crit = keep_algo[c[0]]
        planted = sec[4] == "planted"
        attr, s = (sec[1].split("_s")[0], float(sec[1].split("_s")[1])) if planted else ("colour", 0.0)
        rows.append({
            "comparison_axis": "fixed_encoder_vary_estimator",
            "fixture_id": "F-SOLVER-P116",
            "dataset": "CIFAR-10 (45k FIT partition)",
            "encoder_paper_name": "SimCLR (in-repo P5, 2 views, 200 epochs)",
            "encoder_seed": 0,
            "encoder_checkpoint_sha256": "f180f84ccef5ea7c (P45 manifest encoder_sha256)",
            "measurement_protocol_id": "P116 (FIT/EVAL/POOL disjoint n each; FIT 80/20; B=200 within-class perms; alpha .05)",
            "estimator_method": est,
            "critic_class": f"phi=[h_std(2N-1),1] linear; {crit}",
            "attribute": attr if planted else f"null ({sec[4]})",
            "strength_native": s if planted else (0.2 if "planted0.2" in sec[1] else 0.0),
            "n_detection": sec[2],
            "n_repeats": sec[3],
            "rejection_rate": float(c[1]) if planted else None,
            "null_rejection_rate": None if planted else float(c[1]),
            "null_flag": None if planted else float(c[1]) > FLAG,
            "fit_selection_seconds": float(c[8]),
            "closures": int(c[9]),
            "source_ref": f"reports/P116_results.md:{i} (aggregate of reports/P116/{sec[0]}.json)",
        })
    with open(out / "measurement_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/manuscript_v6_handoff/tables")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    r3, r4 = table3(out), table4(out)
    print(f"measurement_comparison.csv: {len(r3)} rows; ssl_retention_response.csv: {len(r4)} rows -> {out}")
