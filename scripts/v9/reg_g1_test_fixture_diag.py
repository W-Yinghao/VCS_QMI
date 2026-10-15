"""Diagnostic for the V9-G1 test-read fixture (non-test; REG-P1 DEV cores only): is the gap between the fixture's registrations and the recorded
G1 / REG-P2 DEV records run-to-run nondeterminism of REG-P2's pipeline, or a path difference?  Each listed predictor is registered twice in one
process with g1_test.register on the latest fixture root (same cores, same REG-P2 perturbations); the kernel bandwidth h is compared with G1's
recorded h for the chosen multiplier.  Writes <fixture root>/diag.json.  Never touches the sealed archive or the test root."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "src"))
from v9.reg import g1_test as T  # noqa: E402

root = sorted(p for p in T.FIX_ROOT.iterdir() if (p / "fixture_report.json").exists())[-1]; rep0 = json.load(open(root / "fixture_report.json"))
preds = T.load_predictors(); pert = json.load(open(root / "perturbations.json")); cores = rep0["cores"]
ids = sorted(k for k, v in rep0["reproduction"].items() if v["max_abs_d_err_full"] >= 0.5) + ["V9-G1-vcs-frac0.25-eps0.2-s901", "V9-G1-comir-frac1-eps0.2-s902", "V9-G1-logistic-frac1-eps0-s903"]
runs = {}; tag = sys.argv[1] if len(sys.argv) > 1 else ""; sfx = f"_{tag}" if tag else ""
for rep in (1, 2):
    for pid in ids:
        T.register(pid, root, pert, preds, n_pert=1, out=root / f"diag{sfx}_rep{rep}"); runs[(pid, rep)] = json.load(open(root / f"diag{sfx}_rep{rep}" / f"{pid}.json"))
fix = {pid: json.load(open(root / "results" / f"{pid}.json")) for pid in ids}
d = lambda a, b: max(abs(a["test"][c][0]["err_full"] - b["test"][c][0]["err_full"]) for c in cores)
fl = lambda a, b: sum((a["test"][c][0]["err_full"] < 42) != (b["test"][c][0]["err_full"] < 42) for c in cores)
out = {"root": str(root), "device": runs[(ids[0], 1)]["device"], "fixture_device": fix[ids[0]]["device"], "cores": cores, "rows": {}}
for pid in ids:
    ref = json.load(open(preds[pid]["g1_result"] or T.P2_ROOT / "results" / f"{pid}.json")); refv = {"test": {c: ref["dev"][c] for c in cores}}
    row = {"rep1_vs_rep2_max_abs_d_err": d(runs[(pid, 1)], runs[(pid, 2)]), "rep1_vs_rep2_flips42": fl(runs[(pid, 1)], runs[(pid, 2)]),
           "rep1_vs_fixture_max_abs_d_err": d(runs[(pid, 1)], fix[pid]), "rep1_vs_record_max_abs_d_err": d(runs[(pid, 1)], refv), "rep2_vs_record_max_abs_d_err": d(runs[(pid, 2)], refv),
           "fixture_vs_record_max_abs_d_err": d(fix[pid], refv), "err_full": {k: [round(r["test"][c][0]["err_full"], 3) for c in cores] for k, r in (("rep1", runs[(pid, 1)]), ("rep2", runs[(pid, 2)]), ("fixture", fix[pid]), ("record", refv))}}
    if preds[pid]["method"] == "kernel":
        sel = {s["mult"]: s["h"] for s in ref["selection"]}; row["h_fixture_rep1_rep2_record"] = [fix[pid]["kernel_h"], runs[(pid, 1)]["kernel_h"], runs[(pid, 2)]["kernel_h"], sel[ref["chosen_mult"]]]
    out["rows"][pid] = row
json.dump(out, open(root / f"diag{sfx}.json", "w"), indent=1)
for pid, r in out["rows"].items():
    print(f"{pid}: rep1-rep2 {r['rep1_vs_rep2_max_abs_d_err']:.4f} (flips42 {r['rep1_vs_rep2_flips42']}), rep1-fixture {r['rep1_vs_fixture_max_abs_d_err']:.4f}, rep1-record {r['rep1_vs_record_max_abs_d_err']:.4f}, "
          f"fixture-record {r['fixture_vs_record_max_abs_d_err']:.4f}" + (f", h {r['h_fixture_rep1_rep2_record']}" if "h_fixture_rep1_rep2_record" in r else "") + f" | err {r['err_full']}")
print(f"[diag] device {out['device']} (fixture {out['fixture_device']}) -> {root / f'diag{sfx}.json'}")
