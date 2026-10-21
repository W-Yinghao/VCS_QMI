#!/usr/bin/env python3
"""BF RuLSIF_exact_v2 FIT gate (job 1048980, seed 1017): candidate 5 (bandwidth x.35, ridge .01) per core and per fold.

Reads the gate record and its rows file (sha256 checked), keeps every core including fold 2 and the failing cores, and
reconciles with the summaries stored in the record. Output: OUT/cand5_rows.csv, OUT/cand5_per_core.csv, OUT/cand5_B_export.json.
Usage: rulsif_v2_cand5_B.py OUT_DIR
"""
import csv, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

GATE = Path("/home/infres/yinwang/CS_QMI/handoff/v10_rulsif_v2_20261019/gate/FIT_GATE_RuLSIF_exact_v2.json")
LEVELS = {"A": "deployed_position_heldout", "B": "crossfit4_core_heldout"}


def stats(rs):
    n = len(rs); f = [r["F_true"] for r in rs]
    return {"n": n, "mean_F_true": sum(f) / n, "frac_F_true_positive": sum(x > 0 for x in f) / n,
            "n_F_true_positive": sum(x > 0 for x in f),
            "frac_true_beats_imposed": sum(r["F_true"] > r["F_imposed"] for r in rs) / n,
            "n_true_beats_imposed": sum(r["F_true"] > r["F_imposed"] for r in rs),
            "mean_F_imposed": sum(r["F_imposed"] for r in rs) / n,
            "mean_truth_dependence_term": sum(r["truth_dependence_term"] for r in rs) / n,
            "mean_truth_quadratic_term": sum(r["truth_quadratic_term"] for r in rs) / n}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    g = json.load(open(GATE)); rows_path = Path(g["rows"])
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != g["rows_sha256"]:
        raise SystemExit("rows file does not match the gate record")
    cand = [c for c in g["families"]["RuLSIF_exact_v2"]["candidates"] if c["candidate"]["order"] == 5][0]
    fold_of = {core: i for i, f in enumerate(g["folds"]) for core in f}
    rows = [r for r in map(json.loads, open(rows_path)) if r["order"] == 5]
    with open(out / "cand5_rows.csv", "w", newline="") as f:
        keys = ("level", "fold", "core", "perturbation", "F_true", "F_imposed", "truth_dependence_term", "truth_quadratic_term")
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            lv = "A" if r["level"] == LEVELS["A"] else "B"
            w.writerow([lv, fold_of[r["core"]] if lv == "B" else "", r["core"], r["perturbation"], r["F_true"], r["F_imposed"],
                        r["truth_dependence_term"], r["truth_quadratic_term"]])
    res = {"gate_record": str(GATE), "gate_record_sha256": hashlib.sha256(GATE.read_bytes()).hexdigest(),
           "gate_job": g["job"], "gate_status": g["status"], "seed": g["seed"], "source_commit": g["source_commit"],
           "code_sha256": g["code_sha256"], "rows": str(rows_path), "rows_sha256": g["rows_sha256"],
           "candidate": cand["candidate"], "thresholds": g["thresholds"], "levels": {}}
    per_core = []
    for lv, name in LEVELS.items():
        rs = [r for r in rows if r["level"] == name]
        by_core = defaultdict(list)
        for r in rs: by_core[r["core"]].append(r)
        core_stats = {c: stats(v) for c, v in sorted(by_core.items())}
        for c, s in core_stats.items():
            per_core.append({"level": lv, "fold": fold_of[c] if lv == "B" else "", "core": c, **s})
        lvl = {"overall": stats(rs), "cores_mean_F_true_negative": sum(s["mean_F_true"] < 0 for s in core_stats.values()),
               "n_cores": len(core_stats), "recorded_summary": cand[name]["summary"], "recorded_verdict": cand[name].get("verdict")}
        rec = cand[name]["summary"]; o = lvl["overall"]
        lvl["reconciled"] = {k: abs(o[k] - rec[k]) < 1e-12 for k in ("mean_F_true", "frac_F_true_positive", "frac_true_beats_imposed")}
        if lv == "B":
            lvl["folds"] = [{"fold": i, "held_out": f, **stats([r for r in rs if fold_of[r["core"]] == i]),
                             "fit": {k: v for k, v in cand[name]["folds"][i].items() if k not in ("fold", "held_out")}}
                            for i, f in enumerate(g["folds"])]
        res["levels"][lv] = lvl
    with open(out / "cand5_per_core.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_core[0])); w.writeheader(); w.writerows(per_core)
    (out / "cand5_B_export.json").write_text(json.dumps(res, indent=1) + "\n")
    b = res["levels"]["B"]
    print(json.dumps({"B_overall": {k: b["overall"][k] for k in ("mean_F_true", "n_F_true_positive", "n_true_beats_imposed")},
                      "B_cores_negative": b["cores_mean_F_true_negative"], "B_reconciled": b["reconciled"],
                      "B_fold_means": [round(f["mean_F_true"], 8) for f in b["folds"]],
                      "A_reconciled": res["levels"]["A"]["reconciled"], "A_cores_negative": res["levels"]["A"]["cores_mean_F_true_negative"]}))


if __name__ == "__main__":
    main()
