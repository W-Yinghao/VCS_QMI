"""R6 REG-P3 readings (reports/R6/R6_REG_P3_PROTOCOL_*_20261012.yaml) over outputs/R6_REG_P3/results/*.json (scripts/r6_reg_p3.py register).

Every statistic aggregates per independent unit first (the manifest's window -> unit map: an EPFL scene with its near-duplicates merged; for
the Zurich path, a tile or near-adjacent tiles): a unit's success is the fraction of its registrations (all its windows × 10 perturbations) below
the threshold, and DEV success is the mean over units.  Thresholds: CoMIR's mean corner error < 10.5 / < 2.25 native px (REG-P2's 42 / 9 px in
REG-P2-equivalent units, the results' err_full).  Per method: DEV unit-macro success at both thresholds (mean ± sd over seeds for vcs, logistic,
infonce, comir — and comir_wte if it was run; deterministic rows for mi / nmi / kernel), median / p90 corner error (native px; metres when the
manifest has a ground sampling distance), median rotation and translation error,
failures attributed to the energy or the search, success by translation stratum (|t*| bins of REG-P1's strata) and by |θ*| bin, s / registration.
CAL unit-macro success < 42 per method feeds the pre-registered ceiling readout.  Contrasts VCS − X as REG-P2 (paired by seed, or each VCS seed
against the deterministic X; 95 % t over the 3 VCS seeds and a unit bootstrap of 2 000 resamples) at < 42 (primary: X = mi, X = comir) and < 9;
plus the sensitivity contrast comir_wte − comir.  Every contrast uses the same bootstrap unit indices (a fresh generator with one seed per
contrast; all contrasts share the DEV unit set), so the intervals do not depend on the order of the contrasts.
REG-P2's aggregation code is reused read-only on unit-pooled views of the runs (scripts/r6_reg_p2_aggregate.py, scripts/r6_reg_aggregate.py).
    python scripts/r6_reg_p3_aggregate.py [--smoke] [--root R] [--out DIR (default reports/R6)]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r6_reg_aggregate import tint  # noqa: E402  (REG-P1 readings, read-only)
from r6_reg_p2_aggregate import SEEDED, THR, by_rotation, contrast, load, macro  # noqa: E402  (REG-P2 readings, read-only)

ROOT3 = Path("/home/infres/yinwang/CS_QMI/outputs/R6_REG_P3"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R6"
DOWN_EQ, BOOT_SEED, CEIL = 4.0, 20261012, 0.95                                    # err_full / DOWN_EQ = native px
STRATA_PX = (17.0, 34.0)                                                          # REG-P1's translation strata 0–17 / 17–34 / 34–51 px (at 512)


def unit_view(run: dict, unit_of: dict) -> dict:
    """A copy of a run whose CAL / DEV records are pooled per unit (window keys -> unit keys; '#profile' entries dropped)."""
    out = dict(run)
    for part in ("cal", "dev"):
        pooled = {}
        for k in sorted(run.get(part) or {}):
            if "#" not in k:
                pooled.setdefault(unit_of[k], []).extend(run[part][k])
        out[part] = pooled
    return out


def stratum(r: dict) -> int:
    m = math.hypot(*r["t"]); return 0 if m < STRATA_PX[0] else (1 if m < STRATA_PX[1] else 2)


def by_stratum_units(run: dict, thr: float) -> list:
    """Unit-macro success per translation stratum (strata read from |t*|, so they survive the pooling of windows)."""
    out = []
    for j in range(3):
        per = [np.mean([r["err_full"] < thr for r in v if stratum(r) == j]) for v in run["dev"].values() if any(stratum(r) == j for r in v)]
        out.append(float(np.mean(per)) if per else None)
    return out


def cal_macro(run: dict, thr: float):
    per = [np.mean([r["err_full"] < thr for r in v]) for v in (run.get("cal") or {}).values() if v]; return float(np.mean(per)) if per else None


def pooled_stats(run: dict) -> dict:
    """Registration-level medians / p90 over all DEV registrations, in native px (and the rotation / translation errors)."""
    recs = [r for v in run["dev"].values() for r in v]
    if not recs:
        return {}
    e = np.array([r["err_full"] for r in recs]) / DOWN_EQ; te = np.array([r["trans_err_full"] for r in recs]) / DOWN_EQ
    fail = [r for r in recs if r["err_full"] >= THR[0]]
    return {"median_err_native": float(np.median(e)), "p90_err_native": float(np.percentile(e, 90)), "median_rot_err_deg": float(np.median([r["rot_err_deg"] for r in recs])),
            "median_trans_err_native": float(np.median(te)), "failures42": len(fail), "failures42_energy": int(sum(r["F_hat"] >= r["F_true"] for r in fail)),
            "n_registrations": len(recs), "sec_per_registration": float(np.mean([r["seconds"] for r in recs]))}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--root", type=Path, default=ROOT3)
    ap.add_argument("--out", type=Path, default=OUT); a = ap.parse_args(); sfx = "_smoke" if a.smoke else ""; base = a.root / "smoke" if a.smoke else a.root
    man = json.loads((base / ("manifest_smoke.json" if a.smoke else "manifest.json")).read_text()); unit_of = {w: v["unit"] for w, v in man["windows"].items()}
    gsd = man.get("gsd_m") or (float(np.mean([t["pixel_scale_m"][0] for t in man["tiles"].values()])) if "tiles" in man else None)
    raw, missing = load(base / "results", sfx); runs = {k: unit_view(v, unit_of) for k, v in raw.items()}
    not_run = [f for f in missing if f.startswith("comir_wte")]; missing = [f for f in missing if not f.startswith("comir_wte")]   # wte: not run in REG-P3
    dev_units = sorted({unit_of[w] for w, r in man["roles"].items() if r == "DEV"})
    res = {"protocol": "reports/R6/R6_REG_P3_PROTOCOL_*_20261012.yaml", "dataset": man.get("dataset"), "unit": "manifest unit (scene / tile group)",
           "dev_units": dev_units, "n_dev_windows": sum(r == "DEV" for r in man["roles"].values()), "gsd_m": gsd, "thresholds_native_px": [t / DOWN_EQ for t in THR],
           "methods": {}, "contrasts": {}, "missing": missing, "not_run": not_run}
    for m in SEEDED + ("mi", "nmi", "kernel"):
        keys = [k for k in runs if k[0] == m]
        if not keys:
            continue
        row = {"seeds": [k[1] for k in keys]}
        for thr in THR:
            row[f"success{int(thr)}"] = tint([macro(runs[k], thr) for k in keys])
            row[f"success{int(thr)}_by_t_stratum"] = [by_stratum_units(runs[k], thr) for k in keys]; row[f"success{int(thr)}_by_rot_bin"] = [by_rotation(runs[k], thr) for k in keys]
        st = [pooled_stats(runs[k]) for k in keys]
        for f in ("median_err_native", "p90_err_native", "median_rot_err_deg", "median_trans_err_native", "sec_per_registration"):
            row[f] = tint([s[f] for s in st if s])
        row["failures42"] = [s.get("failures42") for s in st]; row["failures42_energy"] = [s.get("failures42_energy") for s in st]
        row["n_registrations"] = [s.get("n_registrations") for s in st]; cal = [cal_macro(runs[k], THR[0]) for k in keys]
        row["cal_success42"] = tint([c for c in cal if c is not None]) if any(c is not None for c in cal) else None
        row["dev_success_per_unit42"] = {str(k[1]): {u: float(np.mean([r["err_full"] < THR[0] for r in v])) for u, v in runs[k]["dev"].items()} for k in keys}
        if m in ("vcs", "logistic", "infonce"):
            row["chosen_ckpt"] = [raw[k].get("chosen_ckpt") for k in keys]
        if m == "kernel":
            row["chosen_mult"] = raw[keys[0]].get("chosen_mult")
        res["methods"][m] = row
    for thr in THR:                                                              # one generator seed per contrast: identical unit indices everywhere
        for b, o in [("vcs", x) for x in ("mi", "nmi", "kernel", "logistic", "infonce", "comir", "comir_wte")] + [("comir_wte", "comir")]:
            c = contrast(runs, b, o, thr, np.random.default_rng(BOOT_SEED))
            if c:
                res["contrasts"][f"{b}_minus_{o}@{int(thr)}"] = c
    cal_rows = {m: r["cal_success42"]["mean"] for m, r in res["methods"].items() if r.get("cal_success42") and m != "comir_wte"}
    res["ceiling_readout"] = {"cal_success42_unit_macro": cal_rows, "threshold": CEIL, "missing_primary_rows": missing,
                              "triggered": None if missing else (bool(cal_rows) and all(v >= CEIL for v in cal_rows.values()))}
    a.out.mkdir(parents=True, exist_ok=True); (a.out / f"R6_REG_P3_results{sfx}.json").write_text(json.dumps(res, indent=1))
    p = lambda t: "—" if not t or t.get("mean") is None else f"{100 * t['mean']:.1f}" + (f" ± {100 * t['sd']:.1f}" if t.get("sd") is not None else "")
    v = lambda t, f="{:.2f}": "—" if not t or t.get("mean") is None else f.format(t["mean"])
    q = lambda t: f"{100 * t['mean']:+.1f}" + (f" [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}]" if t.get("ci95") else "") + (f"; unit bootstrap [{100 * t['image_boot95'][0]:+.1f}, {100 * t['image_boot95'][1]:+.1f}]" if t.get("image_boot95") else "")
    metres = lambda x: f" / {x * gsd:.2f}" if gsd else ""
    L = [f"# R6 REG-P3 — {man.get('dataset')}: rigid registration (θ* ∈ ±30°, |t*| <= 51 px, 512-px windows): DEV results{' (SMOKE)' if a.smoke else ''}", "",
         f"Missing: {missing or 'none'}{f' (not run by protocol: {len(not_run)} CoMIR warp-then-encode files)' if not_run else ''}.  DEV: {len(dev_units)} units, "
         f"{res['n_dev_windows']} windows.  Success thresholds: mean corner error < 10.5 / < 2.25 native px{f' (= {10.5 * gsd:.2f} / {2.25 * gsd:.2f} m)' if gsd else ''} "
         "(REG-P2's 42 / 9 px in REG-P2-equivalent units).  Unit-macro success.", "",
         f"| method | seeds | success < 10.5 px (%) | success < 2.25 px (%) | median corner error (px{' / m' if gsd else ''}) | p90 (px) | median rotation error (°) | median translation error (px) | failures (energy) | CAL success < 10.5 (%) | s / registration |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        L.append(f"| {m} | {r['seeds']} | {p(r['success42'])} | {p(r['success9'])} | {v(r['median_err_native'])}{metres(r['median_err_native']['mean'])} | {v(r['p90_err_native'], '{:.1f}')} | "
                 f"{v(r['median_rot_err_deg'])} | {v(r['median_trans_err_native'])} | {sum(x or 0 for x in r['failures42'])} ({sum(x or 0 for x in r['failures42_energy'])}) | "
                 f"{p(r['cal_success42'])} | {v(r['sec_per_registration'])} |")
    ce = res["ceiling_readout"]
    L += ["", "comir = CoMIR encode-once (the CoMIR row); comir_wte (CoMIR warp-then-encode) is not run in REG-P3.",
          f"Ceiling readout (every method's CAL success < 10.5 px >= {CEIL}): "
          f"{'undetermined (primary rows missing)' if ce['triggered'] is None else ('TRIGGERED' if ce['triggered'] else 'not triggered')}.",
          "", "| contrast (unit-macro success; @42 = < 10.5 px, @9 = < 2.25 px) | value |", "|---|---|"] + [f"| {k} | {q(c)} |" for k, c in res["contrasts"].items()]
    (a.out / f"R6_REG_P3_tables{sfx}.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
