"""REV9 readout (reports/REV9/REV9_PROTOCOL_FROZEN_20261014.yaml): paired per-item differences of the intervention units against their frozen
units on GQA DEV / TextVQA val / ChartQA val at the fixed budgets, with the frozen D1-30 scorer's functions (item_scores, load_config, image-grouped
Boot 2,000 seed 0, boot_p, holm).  Primary per objective: I1 - frozen and I2 - frozen, seed-mean of per-item paired differences, Holm over the 24
contrasts (2 interventions x 4 budgets x 3 benchmarks).  Secondary: I2 - I1, per-seed differences, accuracy levels, the I0 guard, soft-hard gaps.
    python scripts/rev9_score.py --out F"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402

D1 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1"); RV = Path("/home/infres/yinwang/CS_QMI/outputs/REV9")
TAGS = ("vcs_lr0.0003", "logistic_lr0.0001", "direct_lr0.0001"); BENCHES = ("gqa_dev", "textvqa_val", "chartqa_val"); IVS = ("I1", "I2")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--d1", default=str(D1)); ap.add_argument("--rv", default=str(RV))
    ap.add_argument("--tags", nargs="+", default=list(TAGS)); a = ap.parse_args()
    d1, rv, tags = Path(a.d1), Path(a.rv), tuple(a.tags)
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    res = {"guard": {}, "gaps": {}, "benches": {}, "primary": {}, "integrity": {}}
    for t in tags:
        g = json.loads((rv / "selectors" / f"{t}_s0_I0.json").read_text())["rev9"]; res["guard"][t] = {**g, "pass": g["max_abs_diff_estimator_vs_frozen"] == 0 and g["max_abs_diff_gate_vs_frozen"] <= 1e-3}
        for s in (0, 1, 2):
            fz = json.loads((d1 / "selectors" / f"{t}_s{s}.json").read_text()).get("soft_hard_gap")
            res["gaps"][f"{t}_s{s}"] = {"frozen": fz, **{iv: json.loads((rv / "selectors" / f"{t}_s{s}_{iv}.json").read_text()).get("soft_hard_gap") for iv in IVS}}
    for b in BENCHES:
        items = RUN.load_items(Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1") / "d1_30" / "sets", b, "0/1"); boot = SC.Boot([it["image_id"] for it in items], reps=2000, seed=0); qids = [it["qid"] for it in items]
        def acc(pdir, cfg):
            recs, n = SC.load_config(pdir / b, cfg); res["integrity"][f"{pdir.name}:{b}:{cfg}"] = {k: v for k, v in n.items() if k != "files"}
            sc = SC.item_scores(b, items, {q: r["pred"] for q, r in recs.items()})
            if len(sc) != len(items):
                raise SystemExit(f"{b} {cfg}: {len(sc)} of {len(items)} items scored")
            return np.array([sc[q] for q in qids], float)
        bench = {}
        for t in tags:
            for k in SC.BUDGETS:
                fr = {s: acc(d1 / "d1_30" / "pred", f"sel:{t}_s{s}:{k}") for s in (0, 1, 2)}
                iv_acc = {iv: {s: acc(rv / "d1_30" / "pred", f"sel:{t}_s{s}_{iv}:{k}") for s in (0, 1, 2)} for iv in IVS}
                row = {"frozen": [float(fr[s].mean() * 100) for s in (0, 1, 2)]}
                for iv in IVS:
                    row[iv] = [float(iv_acc[iv][s].mean() * 100) for s in (0, 1, 2)]
                    d = np.mean([iv_acc[iv][s] - fr[s] for s in (0, 1, 2)], 0) * 100; bs = boot.means(d)
                    row[f"{iv}-frozen"] = {"delta": float(d.mean()), "ci95": SC.ci(bs), "p": SC.boot_p(bs), "per_seed": [float((iv_acc[iv][s] - fr[s]).mean() * 100) for s in (0, 1, 2)]}
                d = np.mean([iv_acc["I2"][s] - iv_acc["I1"][s] for s in (0, 1, 2)], 0) * 100; bs = boot.means(d)
                row["I2-I1"] = {"delta": float(d.mean()), "ci95": SC.ci(bs), "p": SC.boot_p(bs)}
                bench[f"{t}@{k}"] = row
        res["benches"][b] = bench
    for t in tags:
        keys = [(b, f"{t}@{k}", iv) for b in BENCHES for k in SC.BUDGETS for iv in IVS]
        ph = SC.holm([res["benches"][b][c][f"{iv}-frozen"]["p"] for b, c, iv in keys])
        res["primary"][t] = [{"bench": b, "cell": c, "contrast": f"{iv}-frozen", **res["benches"][b][c][f"{iv}-frozen"], "p_holm": float(h)} for (b, c, iv), h in zip(keys, ph)]
    Path(a.out).write_text(json.dumps(res, indent=1))
    for t in tags:
        for r in res["primary"][t]:
            print(f"{r['bench']:12s} {r['cell']:24s} {r['contrast']:10s} {r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] p_holm {r['p_holm']:.3f}")
    print("guard:", {t: g["pass"] for t, g in res["guard"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
