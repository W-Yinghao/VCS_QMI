"""REV9 execution note 6 — SECONDARY readout of the direct objective from its H100 rerun (REV9_FULL_H100: I0 guard + I1 / I2 of seed 0 only).

The frozen readout `rev9_score.py` pairs seeds 0, 1, 2; the H100 secondary (execution note 3) re-ran only `direct_lr0.0001_s0`, so this script
is the same computation restricted to the seeds given (default: 0) — same scorer functions (item_scores, load_config, image-grouped Boot 2,000
seed 0, boot_p, holm over the 24 contrasts), same budgets, benches and interventions, same output structure (with a "seeds" field).  It is a
labelled secondary reading: the primary REV9-FULL readout of the direct objective stays stopped (note 3).  Refuses to overwrite its output.
    python scripts/rev9_score_secondary.py --d1 <frozen D1-FULL dir> --rv <REV9_FULL_H100> --tags direct_lr0.0001 --seeds 0 --out F
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402
from rev9_score import BENCHES, IVS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--d1", required=True); ap.add_argument("--rv", required=True)
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--seeds", nargs="+", type=int, default=[0]); a = ap.parse_args()
    d1, rv, tags, seeds = Path(a.d1), Path(a.rv), tuple(a.tags), tuple(a.seeds)
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    res = {"secondary": "REV9 execution note 6: H100 rerun, seeds restricted", "seeds": list(seeds), "guard": {}, "gaps": {}, "benches": {}, "primary": {}, "integrity": {}}
    for t in tags:
        g = json.loads((rv / "selectors" / f"{t}_s{seeds[0]}_I0.json").read_text())["rev9"]
        res["guard"][t] = {**g, "pass": g["max_abs_diff_estimator_vs_frozen"] == 0 and g["max_abs_diff_gate_vs_frozen"] <= 1e-3}
        for s in seeds:
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
                fr = {s: acc(d1 / "d1_30" / "pred", f"sel:{t}_s{s}:{k}") for s in seeds}
                iv_acc = {iv: {s: acc(rv / "d1_30" / "pred", f"sel:{t}_s{s}_{iv}:{k}") for s in seeds} for iv in IVS}
                row = {"frozen": [float(fr[s].mean() * 100) for s in seeds]}
                for iv in IVS:
                    row[iv] = [float(iv_acc[iv][s].mean() * 100) for s in seeds]
                    d = np.mean([iv_acc[iv][s] - fr[s] for s in seeds], 0) * 100; bs = boot.means(d)
                    row[f"{iv}-frozen"] = {"delta": float(d.mean()), "ci95": SC.ci(bs), "p": SC.boot_p(bs), "per_seed": [float((iv_acc[iv][s] - fr[s]).mean() * 100) for s in seeds]}
                d = np.mean([iv_acc["I2"][s] - iv_acc["I1"][s] for s in seeds], 0) * 100; bs = boot.means(d)
                row["I2-I1"] = {"delta": float(d.mean()), "ci95": SC.ci(bs), "p": SC.boot_p(bs)}
                bench[f"{t}@{k}"] = row
        res["benches"][b] = bench
    for t in tags:
        keys = [(b, f"{t}@{k}", iv) for b in BENCHES for k in SC.BUDGETS for iv in IVS]
        ph = SC.holm([res["benches"][b][c][f"{iv}-frozen"]["p"] for b, c, iv in keys])
        res["primary"][t] = [{"bench": b, "cell": c, "contrast": f"{iv}-frozen", **res["benches"][b][c][f"{iv}-frozen"], "p_holm": float(h)} for (b, c, iv), h in zip(keys, ph)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    for t in tags:
        for r in res["primary"][t]:
            print(f"{r['bench']:12s} {r['cell']:24s} {r['contrast']:10s} {r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] p_holm {r['p_holm']:.3f}")
    print("guard:", {t: g["pass"] for t, g in res["guard"].items()}, "seeds:", list(seeds), "(SECONDARY, note 6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
