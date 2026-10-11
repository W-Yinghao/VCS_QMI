"""P151 r4 (R4-01, V6 server package r4) — corrected re-aggregation of the P85 / P86 / P108 estimator records.

The r3 aggregator (scripts/p151_e0_aggregate.py, kept unchanged) filled the SAME-target panel's `J_eval` / `J_minus_S` columns and the stability /
resolution panels from `native.*` fields.  For the JS families the native value is a JS-type target (own truth JS2), and for `s_kde:common_risk` it is
the plug-in S; neither is the common bounded J.  This version reads the common readout only from `posterior.*` and never substitutes a native value:

  common_J          <- posterior.J_eval            (required; else null + reason)
  common_J_minus_S  <- posterior.J_eval_minus_S
  posterior_mse     <- posterior.posterior_mse     (unchanged by the fix)
  common_J_boot_se  <- posterior.J_eval_boot_se    (the common-J bootstrap draws were not saved -> ordering probabilities are null + reason)
  native_*          <- native.value / own_truth / abs_error / boot_values, with the row's `estimand` and `evaluation_readout` (own panel only)

Selection: the single registered `selected` row per family; families with zero or several flags are excluded and listed (no fallback).  Identity check
per row: R_eval = 1 - J_eval on the same evaluation; a violating row's common readout is excluded (status identity_violation) and listed.  S_oracle per
ladder cell is the seed mean of truth.S (per-seed values kept).  Writes schema-versioned JSON, a per-row provenance table, an old-vs-new diff table
and a markdown summary to --out; raw JSONs and the r3 outputs are not touched.
    python scripts/p151_e0_aggregate_r4.py --out reports/R4/P151_r4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p151_e0_aggregate import LADDER, NATIVE, O, SAME, family, load_cells  # noqa: E402

SCHEMA = "p151_r4.2"
NO_BOOT = "common-J bootstrap draws were not saved in the P85 records (only posterior.J_eval_boot_se); no ordering probability is formed from native draws"


def selected_rows(d: dict) -> dict:
    """Registered selection only: the single row flagged `selected` per family.  A family with zero or several flags is NOT selected (no order-based
    or risk-based fallback); it is reported under metadata.selection_problems and left out of every panel."""
    by = defaultdict(list)
    for r in d["rows"]:
        if r.get("selected"):
            by[family(r["method"])].append(r)
    return {fam: rows[0] for fam, rows in by.items() if len(rows) == 1}


def selection_problems(d: dict) -> list:
    by = defaultdict(list)
    for r in d["rows"]:
        by[family(r["method"])].append(r)
    return [{"family": fam, "n_selected": sum(1 for r in rows if r.get("selected"))} for fam, rows in by.items() if sum(1 for r in rows if r.get("selected")) != 1 and fam in SAME + NATIVE]


def common_fields(row: dict) -> dict:
    """Common readout of one selected row; null + reason when absent, never a native fallback."""
    p = row.get("posterior") or {}
    if not p.get("defined", False) or p.get("J_eval") is None:
        return {"common_J": None, "common_J_minus_S": None, "posterior_mse": p.get("posterior_mse"), "common_J_boot_se": None,
                "status": "missing_common_readout", "reason": p.get("reason") or "posterior.J_eval absent; native fallback forbidden"}
    j = float(p["J_eval"])
    if not math.isfinite(j):
        return {"common_J": None, "common_J_minus_S": None, "posterior_mse": p.get("posterior_mse"), "common_J_boot_se": None,
                "status": "nonfinite_common_readout", "reason": "posterior.J_eval is not finite"}
    r = p.get("R_eval"); ident = None if r is None else bool(math.isclose(float(r), 1 - j, rel_tol=1e-6, abs_tol=1e-8))
    if ident is False:                                              # same-evaluation identity violated: the row's common readout is not used
        return {"common_J": None, "common_J_minus_S": None, "posterior_mse": p.get("posterior_mse"), "common_J_boot_se": None,
                "status": "identity_violation", "reason": f"R_eval {r} != 1 - J_eval {1 - j}; inspect schema / weights", "identity_R_eq_1_minus_J": False}
    return {"common_J": j, "common_J_minus_S": p.get("J_eval_minus_S"), "posterior_mse": p.get("posterior_mse"), "common_J_boot_se": p.get("J_eval_boot_se"),
            "status": "available", "reason": None, "identity_R_eq_1_minus_J": ident}


def native_fields(row: dict) -> dict:
    n = row.get("native") or {}
    return {"native_value": n.get("value"), "native_own_truth": n.get("own_truth"), "native_abs_error": n.get("abs_error"), "native_signed_error": n.get("signed_error"),
            "native_boot_se": n.get("boot_se"), "fit_criterion": row.get("estimand"), "evaluation_readout": row.get("evaluation_readout"), "native_readout_note": n.get("readout")}


def msd(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return {"mean": float(x.mean()) if len(x) else None, "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "n": int(len(x))}


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="reports/R4/P151_r4"); ap.add_argument("--old", default="reports/P151_E0"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    inputs = sorted(glob.glob(str(O / "P85_estim_benchmark" / "P85_*.json")) + glob.glob(str(O / "P85_estim_benchmark_smile_fix" / "*.json")))
    cells = load_cells()
    provenance, identity_fail, multi_sel = [], [], []
    lad = defaultdict(lambda: defaultdict(dict))
    for name, d in cells.items():
        c = d["cell"]
        for prob in selection_problems(d):
            multi_sel.append({"cell": name, **prob})
        if c["N"] != 4096 or c["B"] != 256 or c["updates"] != 2000 or c["I"] not in LADDER:
            continue
        if not ((c["setting"] in ("gaussian", "cubic") and c["d_signal"] == 20 and c["d_total"] == 20) or c["setting"] == "xor_mixture"):
            continue
        sr = selected_rows(d)
        lad[c["setting"]][int(c["I"])][c["seed"]] = (d["truth"], sr)
        for fam in SAME + NATIVE:
            if fam in sr:
                cf = common_fields(sr[fam])
                provenance.append({"cell": name, "generator": c["setting"], "I": int(c["I"]), "seed": c["seed"], "family": fam, "method": sr[fam]["method"],
                                   "selected_flag": bool(sr[fam].get("selected")), "negative_construction": sr[fam].get("negative_construction"),
                                   "split_manifest_hash": sr[fam].get("split_manifest_hash"), **cf, **native_fields(sr[fam])})
                if cf.get("identity_R_eq_1_minus_J") is False:
                    identity_fail.append({"cell": name, "family": fam})
    common, native, stab, reso = {}, {}, {}, {}
    for gen, byI in lad.items():
        common[gen], native[gen], stab[gen], reso[gen] = {}, {}, {}, {}
        for I in LADDER:
            if I not in byI:
                continue
            seeds = byI[I]; truth = next(iter(seeds.values()))[0]; S_seeds = [t["S"] for t, _ in seeds.values()]
            C_ = {"S_oracle": float(np.mean(S_seeds)), "S_oracle_seed_sd": float(np.std(S_seeds, ddof=1)) if len(S_seeds) > 1 else None, "S_oracle_per_seed": S_seeds,
                  "S_se": truth.get("S_se"), "J_oracle": truth.get("J_oracle"), "families": {}}
            N_ = {"MI": truth.get("MI"), "JS2": truth.get("JS2"), "families": {}}
            St = {}
            for fam in SAME + NATIVE:
                rows = [sr[fam] for _, sr in seeds.values() if fam in sr]
                if not rows:
                    continue
                cfs = [common_fields(r) for r in rows]; nfs = [native_fields(r) for r in rows]
                if any(cf["status"] == "available" for cf in cfs):
                    C_["families"][fam] = {"common_J": msd([cf["common_J"] for cf in cfs]), "common_J_minus_S": msd([cf["common_J_minus_S"] for cf in cfs]),
                                           "posterior_mse": msd([cf["posterior_mse"] for cf in cfs]), "common_J_boot_se_mean": msd([cf["common_J_boot_se"] for cf in cfs])["mean"],
                                           "n_seeds": len(rows), "fit_criterion": nfs[0]["fit_criterion"], "evaluation_readout": "posterior.J_eval (common bounded J)"}
                else:
                    C_["families"][fam] = {"common_J": None, "status": cfs[0]["status"], "reason": cfs[0]["reason"]}
                N_["families"][fam] = {"native_value": msd([nf["native_value"] for nf in nfs]), "native_own_truth": nfs[0]["native_own_truth"],
                                       "native_abs_error": msd([nf["native_abs_error"] for nf in nfs]), "fit_criterion": nfs[0]["fit_criterion"],
                                       "evaluation_readout": nfs[0]["evaluation_readout"], "native_readout_note": nfs[0]["native_readout_note"]}
                curve_tail = [float(np.std([p[1] for p in (r.get("select", {}).get("curve") or [])[-5:]], ddof=1)) for r in rows if len(r.get("select", {}).get("curve") or []) >= 5]
                St[fam] = {"seed_sd_common_J": msd([cf["common_J"] for cf in cfs])["sd"],
                           "common_J_boot_se_mean": msd([cf["common_J_boot_se"] for cf in cfs])["mean"],
                           "seed_sd_native_value": msd([nf["native_value"] for nf in nfs])["sd"], "native_boot_se_mean": msd([nf["native_boot_se"] for nf in nfs])["mean"],
                           "select_curve_tail_sd": msd(curve_tail)["mean"], "nonfinite_steps_total": int(sum(r.get("numerical", {}).get("nonfinite_steps", 0) or 0 for r in rows)),
                           "fit_seconds_mean": msd([r.get("fit_seconds") for r in rows])["mean"]}
            common[gen][I] = C_; native[gen][I] = N_; stab[gen][I] = St
        for k in range(len(LADDER) - 1):
            I0, I1 = LADDER[k], LADDER[k + 1]
            if I0 not in byI or I1 not in byI:
                continue
            R_ = {"delta_S_oracle": float(np.mean([t["S"] for t, _ in byI[I1].values()]) - np.mean([t["S"] for t, _ in byI[I0].values()])), "common": {}, "native": {}}
            for fam in SAME + NATIVE:
                dm_c, se_c, dm_n, probs_n = [], [], [], []
                for s in sorted(set(byI[I0]) & set(byI[I1])):
                    r0, r1 = byI[I0][s][1].get(fam), byI[I1][s][1].get(fam)
                    if not r0 or not r1:
                        continue
                    c0, c1 = common_fields(r0), common_fields(r1)
                    if c0["common_J"] is not None and c1["common_J"] is not None:
                        dm_c.append(c1["common_J"] - c0["common_J"])
                        if c0["common_J_boot_se"] is not None and c1["common_J_boot_se"] is not None:
                            se_c.append(math.sqrt(0.5 * (c0["common_J_boot_se"] ** 2 + c1["common_J_boot_se"] ** 2)))
                    dm_n.append(r1["native"]["value"] - r0["native"]["value"])
                    b0, b1 = r0["native"].get("boot_values"), r1["native"].get("boot_values")
                    if b0 and b1:
                        probs_n.append(float((np.asarray(b1, float) > np.asarray(b0, float)).mean()))
                if dm_c:
                    dm = np.array(dm_c); pooled = float(math.sqrt(np.mean(np.square(se_c)) + (dm.var(ddof=1) if len(dm) > 1 else 0.0))) if se_c else None
                    R_["common"][fam] = {"delta_mean_common_J": float(dm.mean()), "pooled_sd": pooled, "standardised": float(abs(dm.mean()) / pooled) if pooled else None,
                                         "adjacent_order_probability": None, "adjacent_order_probability_reason": NO_BOOT, "n_seeds": int(len(dm)),
                                         "pooled_sd_note": "sqrt(mean boot SE^2 + seed variance of the paired difference), common J"}
                elif fam in SAME:
                    R_["common"][fam] = {"delta_mean_common_J": None, "reason": "common readout missing for this family"}
                if dm_n:
                    R_["native"][fam] = {"delta_mean_native": float(np.mean(dm_n)), "adjacent_order_probability_native": float(np.mean(probs_n)) if probs_n else None,
                                         "scale": "the family's own native target (not S)"}
            reso[gen][f"{I0}->{I1}"] = R_
    # ---- N / batch axes (gaussian d20 I4): posterior MSE (unchanged) and common |J - S| (new) beside native |value - own truth|
    axes = {"N_axis": defaultdict(lambda: defaultdict(list)), "batch_equal_updates": defaultdict(lambda: defaultdict(list)), "batch_equal_exposure": defaultdict(lambda: defaultdict(list))}
    for name, d in cells.items():
        c = d["cell"]
        if not (c["setting"] == "gaussian" and c["d_signal"] == 20 and c["d_total"] == 20 and c["I"] == 4.0):
            continue
        key = ("N_axis", str(c["N"])) if (c["B"] == 256 and c["updates"] == 2000) else ("batch_equal_updates", str(c["B"])) if (c["N"] == 4096 and c["updates"] == 2000) else \
              ("batch_equal_exposure", f"B{c['B']}_U{c['updates']}") if (c["N"] == 4096 and (c["B"], c["updates"]) in ((64, 8000), (1024, 500))) else None
        if key is None:
            continue
        for fam, r in selected_rows(d).items():
            if fam in SAME + NATIVE:
                cf = common_fields(r)
                axes[key[0]][key[1]][fam].append({"posterior_mse": cf["posterior_mse"], "abs_common_J_minus_S": None if cf["common_J_minus_S"] is None else abs(cf["common_J_minus_S"]),
                                                  "native_abs_error": r.get("native", {}).get("abs_error")})
    axes_s = {ax: {lvl: {fam: {k: msd([x[k] for x in xs]) for k in ("posterior_mse", "abs_common_J_minus_S", "native_abs_error")} for fam, xs in fams.items()} for lvl, fams in levels.items()}
              for ax, levels in axes.items()}
    # ---- old vs new diff (r3 e_precision same_target J_eval / J_minus_S vs corrected common fields)
    diff = []
    oldp = Path(a.old) / "e_precision.json"
    if oldp.exists():
        old = json.load(open(oldp))
        for gen, byI in common.items():
            for I, C_ in byI.items():
                o = old.get(gen, {}).get(str(I), {}).get("same_target", {})
                for fam, v in C_["families"].items():
                    if fam not in SAME or fam not in o:
                        continue
                    for of, nf in (("J_eval", "common_J"), ("J_minus_S", "common_J_minus_S"), ("posterior_mse", "posterior_mse")):
                        ov = o[fam].get(of, {}).get("mean"); nv = (v.get(nf) or {}).get("mean") if isinstance(v.get(nf), dict) else None
                        diff.append({"generator": gen, "I": I, "family": fam, "r3_field": of, "r4_field": nf, "r3_mean": ov, "r4_mean": nv,
                                     "changed": (ov is None) != (nv is None) or (ov is not None and nv is not None and abs(ov - nv) > 1e-9),
                                     "abs_change": None if ov is None or nv is None else abs(ov - nv)})
    meta = {"schema_version": SCHEMA, "inputs": {p: sha(p) for p in inputs}, "r3_outputs_compared": str(oldp) if oldp.exists() else None,
            "selection": "registered `selected` row per family, exactly one required (no fallback)", "selection_problems": multi_sel,
            "identity_R_eq_1_minus_J_failures": identity_fail, "common_bootstrap_draws": NO_BOOT,
            "fields": {"common_J": "posterior.J_eval", "common_J_minus_S": "posterior.J_eval_minus_S", "posterior_mse": "posterior.posterior_mse",
                       "common_J_boot_se": "posterior.J_eval_boot_se", "native_value": "native.value (own estimand; own panel only)"}}
    json.dump(meta, open(out / "metadata.json", "w"), indent=1)
    json.dump(common, open(out / "e_precision_common.json", "w"), indent=1); json.dump(native, open(out / "e_precision_native.json", "w"), indent=1)
    json.dump(stab, open(out / "e_stability_r4.json", "w"), indent=1); json.dump(reso, open(out / "e_resolution_r4.json", "w"), indent=1)
    json.dump(axes_s, open(out / "axes_N_batch_r4.json", "w"), indent=1); json.dump(provenance, open(out / "row_provenance.json", "w"), indent=1)
    json.dump(diff, open(out / "diff_r3_r4.json", "w"), indent=1)
    changed = [x for x in diff if x["changed"]]
    by_field = defaultdict(lambda: defaultdict(list))
    for x in changed:
        by_field[x["r3_field"]][x["family"]].append(x["abs_change"] or 0.0)
    L = [f"# P151 r4 — corrected common-readout aggregation (schema {SCHEMA})", "",
         f"Cells loaded: {len(cells)}; ladder rows with provenance: {len(provenance)}; R = 1 − J identity failures: {len(identity_fail)}; multi-selected families: {len(multi_sel)}.", "",
         "## Old (r3) vs new (r4) — same-target panel means that changed", "", "| r3 field → r4 field | family | cells changed | max abs change |", "|---|---|---|---|"]
    for of, fams in sorted(by_field.items()):
        for fam, v in sorted(fams.items()):
            L.append(f"| {of} → {dict(J_eval='common_J', J_minus_S='common_J_minus_S', posterior_mse='posterior_mse')[of]} | {fam} | {len(v)} | {max(v):.4f} |")
    L += ["", f"Unchanged: {sum(1 for x in diff if not x['changed'])} of {len(diff)} compared means (posterior MSE is read from posterior.* in both versions).", ""]
    for gen in common:
        L += [f"## {gen}: common bounded J − S (mean ± sd over seeds; oracle S)", "", "| I | S | " + " | ".join(SAME) + " |", "|---|---|" + "---|" * len(SAME)]
        for I in LADDER:
            if I in common[gen]:
                C_ = common[gen][I]
                L.append(f"| {I} | {C_['S_oracle']:.3f} | " + " | ".join((f"{v['common_J_minus_S']['mean']:+.3f} ± {v['common_J_minus_S']['sd']:.3f}" if (v := C_['families'].get(f)) and isinstance(v.get('common_J_minus_S'), dict) and v['common_J_minus_S']['mean'] is not None and v['common_J_minus_S']['sd'] is not None else "—") for f in SAME) + " |")
        L.append("")
    (out / "tables_r4.md").write_text("\n".join(L))
    print(f"{len(cells)} cells; {len(provenance)} provenance rows; {len(changed)} of {len(diff)} compared means changed; identity failures {len(identity_fail)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
