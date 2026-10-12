"""R6 REG-P2 readings (reports/R6/R6_REG_P2_PROTOCOL_*_20261011.yaml) over outputs/R6_REG_P2/results/*.json (scripts/r6_reg_p2.py register).

Per method: DEV image-macro success at < 42 px and < 9 px of CoMIR's mean corner error (full resolution; per image over its 10 perturbations,
then the mean over images), median / p90 corner error, median rotation error (°) and translation error (full-resolution px), seconds per
registration; mean ± sd over seeds for seeded methods (vcs, logistic, infonce, comir), deterministic rows for mi / nmi / kernel.  Diagnostics:
failures (>= 42 px) attributed to the energy (F at the estimate >= F at the truth: the energy's maximum is wrong) or to the search (a higher F
at the truth was missed); success by translation stratum (image-macro) and by |θ*| bin (pooled registrations); REG-P1 → REG-P2 change per
method when REG-P1 results exist (descriptive).  comir_wte = CoMIR warp-then-encode, a DEV-only sensitivity row (contrast comir_wte − comir by
seed).  Contrasts VCS − X as REG-P1 (paired by seed or VCS seed vs the deterministic X; 95 % t over
the 3 VCS seeds and an image bootstrap with indices shared across methods) at < 42 px (primary: X = mi, X = comir) and < 9 px (secondary).
Wording: interval containing 0 -> no detectable difference; above / below 0 -> VCS registers x more / fewer.
    python scripts/r6_reg_p2_aggregate.py [--smoke] [--root R] [--p1-results DIR] [--out DIR (default reports/R6)]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_reg_aggregate import per_image, tint  # noqa: E402  (REG-P1 readings, read-only)

ROOT2 = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P2"); P1RES = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG/results")
OUT = Path(__file__).resolve().parents[1] / "reports" / "R6"
SEEDED, DETERMINISTIC, SEEDS, THR = ("vcs", "logistic", "infonce", "comir", "comir_wte"), ("mi", "nmi", "kernel"), (0, 1, 2), (42.0, 9.0)   # comir_wte: sensitivity row (DEV only)
ROT_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0))


def load(rd: Path, sfx: str):
    runs, missing = {}, []
    for m in SEEDED:
        for s in SEEDS:
            f = rd / f"{m}_s{s}{sfx}.json"
            if f.exists():
                runs[(m, s)] = json.load(open(f))
            else:
                missing.append(f.name)
    for m in DETERMINISTIC:
        f = rd / (f"{m}_s0{sfx}.json" if m == "kernel" else f"{m}{sfx}.json")
        if f.exists():
            runs[(m, None)] = json.load(open(f))
        else:
            missing.append(f.name)
    return runs, missing


def macro(run, thr):
    return float(np.mean(list(per_image(run["dev"], thr).values())))


def by_stratum(run, thr):
    out = []
    for j in range(3):
        per = [np.mean([r["err_full"] < thr for i, r in enumerate(v) if i % 3 == j]) for k, v in run["dev"].items() if "#" not in k and len(v) > j]
        out.append(float(np.mean(per)) if per else None)
    return out


def by_rotation(run, thr):
    recs = [r for k, v in run["dev"].items() if "#" not in k for r in v]; out = []
    for lo, hi in ROT_BINS:                                                      # [0, 10), [10, 20), [20, 30]
        sel = [r["err_full"] < thr for r in recs if lo <= abs(r["theta"]) < hi or (hi == ROT_BINS[-1][1] and abs(r["theta"]) == hi)]
        out.append(float(np.mean(sel)) if sel else None)
    return out


def contrast(runs, base, other, thr, g):
    vk = sorted(k for k in runs if k[0] == base); ok = [k for k in runs if k[0] == other]
    if not vk or not ok:
        return None
    pairs = [(v, next((o for o in ok if o[1] == v[1]), None) if other in SEEDED else ok[0]) for v in vk]; pairs = [(v, o) for v, o in pairs if o is not None]
    if not pairs:
        return None
    imgs = sorted(set.intersection(*[set(per_image(runs[v]["dev"], thr)) & set(per_image(runs[o]["dev"], thr)) for v, o in pairs]))
    D = np.array([[per_image(runs[v]["dev"], thr)[i] - per_image(runs[o]["dev"], thr)[i] for i in imgs] for v, o in pairs])      # [pairs, images]
    boot = [float(D[:, g.integers(0, len(imgs), len(imgs))].mean()) for _ in range(2000)] if len(imgs) else []
    return {**tint(D.mean(1)), "image_boot95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else None,
            "n_images": len(imgs), "pairs": [[v[1], o[1]] for v, o in pairs]}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--root", type=Path, default=ROOT2)
    ap.add_argument("--p1-results", type=Path, default=P1RES); ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args(); sfx = "_smoke" if a.smoke else ""
    runs, missing = load((a.root / "smoke" if a.smoke else a.root) / "results", sfx); p1, _ = load(a.p1_results, sfx) if a.p1_results.exists() else ({}, [])
    res = {"protocol": "reports/R6/R6_REG_P2_PROTOCOL_*_20261011.yaml", "methods": {}, "contrasts": {}, "missing": missing}
    for m in SEEDED + DETERMINISTIC:
        keys = [k for k in runs if k[0] == m]
        if not keys:
            continue
        row = {"seeds": [k[1] for k in keys]}
        for thr in THR:
            row[f"success{int(thr)}"] = tint([macro(runs[k], thr) for k in keys])
            row[f"success{int(thr)}_by_t_stratum"] = [by_stratum(runs[k], thr) for k in keys]; row[f"success{int(thr)}_by_rot_bin"] = [by_rotation(runs[k], thr) for k in keys]
        for f in ("dev_median_err", "dev_p90_err", "dev_median_rot_err_deg", "dev_median_trans_err", "dev_sec_per_registration"):
            row[f] = tint([runs[k][f] for k in keys if runs[k].get(f) is not None])
        row["failures42"] = [runs[k]["dev_failures42"] for k in keys]; row["failures42_energy"] = [runs[k]["dev_failures42_energy"] for k in keys]
        if m in ("vcs", "logistic", "infonce"):
            row["chosen_ckpt"] = [runs[k].get("chosen_ckpt") for k in keys]
        if m == "kernel":
            row["chosen_mult"] = runs[keys[0]].get("chosen_mult")
        both = [k for k in keys if k in p1]
        if both:
            row["p1_success42"] = tint([macro(p1[k], 42.0) for k in both]); row["p2_minus_p1_success42"] = tint([macro(runs[k], 42.0) - macro(p1[k], 42.0) for k in both])
            row["p1_success9"] = tint([macro(p1[k], 9.0) for k in both]); row["p2_minus_p1_success9"] = tint([macro(runs[k], 9.0) - macro(p1[k], 9.0) for k in both])
        res["methods"][m] = row
    g = np.random.default_rng(20261012)
    for thr in THR:
        for base, other in [("vcs", o) for o in ("mi", "nmi", "kernel", "logistic", "infonce", "comir", "comir_wte")] + [("comir_wte", "comir")]:
            c = contrast(runs, base, other, thr, g)
            if c:
                res["contrasts"][f"{base}_minus_{other}@{int(thr)}"] = c
    a.out.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out / f"R6_REG_P2_results{sfx}.json", "w"), indent=1)
    p = lambda t: "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")
    v = lambda t, f="{:.1f}": "—" if not t or t.get("mean") is None else f.format(t["mean"])
    q = lambda t: f"{100 * t['mean']:+.1f}" + (f" [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}]" if t.get("ci95") else "") + (f"; image bootstrap [{100 * t['image_boot95'][0]:+.1f}, {100 * t['image_boot95'][1]:+.1f}]" if t.get("image_boot95") else "")
    L = [f"# R6 REG-P2 — BF/SHG rigid registration (θ* ∈ ±30°, REG-P1 translations): DEV results{' (SMOKE)' if a.smoke else ''}", "", f"Missing: {missing or 'none'}.", "",
         "| method | seeds | success < 42 px (%) | success < 9 px (%) | median corner error (px) | p90 (px) | median rotation error (°) | median translation error (px) | failures (energy) | s / registration | REG-P1 → P2 success < 42 (pp) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        L.append(f"| {m} | {r['seeds']} | {p(r['success42'])} | {p(r['success9'])} | {v(r['dev_median_err'])} | {v(r['dev_p90_err'])} | {v(r['dev_median_rot_err_deg'], '{:.2f}')} | "
                 f"{v(r['dev_median_trans_err'])} | {sum(r['failures42'])} ({sum(r['failures42_energy'])}) | {v(r['dev_sec_per_registration'], '{:.2f}')} | "
                 f"{q(r['p2_minus_p1_success42']) if r.get('p2_minus_p1_success42') else '—'} |")
    L += ["", "comir = CoMIR encode-once (primary row); comir_wte = CoMIR warp-then-encode (sensitivity row, DEV only).",
          "", "| contrast (image-macro success) | value |", "|---|---|"] + [f"| {k} | {q(c)} |" for k, c in res["contrasts"].items()]
    (a.out / f"R6_REG_P2_tables{sfx}.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
