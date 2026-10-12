"""r7 R7-40a readout on Qwen2.5-VL-7B: finite teacher residuals of the binary risk object, the frozen D1-10 readout re-applied (CAL primary,
FIT descriptive, DEV sealed: --allow-dev only at R7-40c).  Same statistics as scripts/r7_d1_analyze.py (imported, unchanged): balanced task
law M with the FIT class weights of the frozen manifest, image-cluster bootstrap (2,000 reps), the coordinator's calibrated-teacher rank
ceiling (rank_criterion note) and the pre-declared yes / no-mass >= 0.9 secondary (yes_no_mass note).  Budgets are fractions of each image's
own token count (cells 'mechanism:0.125' ...); the realised mean kept tokens per cell are reported.
    python scripts/r7_d1_40_analyze.py readout --teacher-dir T --manifest-dir D [--roles CAL FIT]
    python scripts/r7_d1_40_analyze.py complete --teacher-dir T --manifest-dir D --frozen-sha256 SHA [--roles CAL FIT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts")); sys.path.insert(0, str(R / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_analyze as AN  # noqa: E402  (frozen D1-10 analyser: statistics, coverage guard, tables)
import r7_vl_common as V  # noqa: E402
import vcs_vl_reference as REF  # noqa: E402

T = AN.T


def analyse40(rows, weights):
    C = np.array([r["C"] for r in rows], float); g = np.array([r["image_id"] for r in rows]); w = np.where(C > 0, weights["w_yes"], weights["w_no"])
    Tb = np.array([T(r["full"]) for r in rows]); Tt = np.array([T(r["text"]) for r in rows])
    nv = np.array([r["n_vis"] for r in rows], float)
    out = {"n_questions": len(rows), "n_images": int(len(np.unique(g))),
           "visual_tokens": {"mean": float(nv.mean()), "median": float(np.median(nv)), "min": int(nv.min()), "max": int(nv.max())},
           "teacher": {"brier_full": AN.wmean((C - Tb) ** 2, w), "brier_full_ci": AN.boot((C - Tb) ** 2, w, g),
                       "brier_text_only": AN.wmean((C - Tt) ** 2, w), "acc_full": AN.wmean((np.sign(Tb) == C).astype(float), w),
                       "acc_text_only": AN.wmean((np.sign(Tt) == C).astype(float), w),
                       "mean_residual_text_only": AN.wmean(REF.finite_risk_delta(C, Tt, Tb)[2], w),
                       "yes_no_mass_qc": AN.mass_qc(rows)}, "cells": {}}
    for m in M4.MECHANISMS:
        for f, key in zip(M4.FRACTIONS, M4.FKEYS):
            Ta = np.array([T(r["masked"][m][key]) for r in rows]); hh, dd, res = REF.finite_risk_delta(C, Ta, Tb)
            ks = np.array([r["budgets"][key] for r in rows], float)
            out["cells"][f"{m}:{key}"] = {
                "mechanism": m, "budget_fraction": f, "budget_tokens": float(ks.mean()), "budget_tokens_range": [int(ks.min()), int(ks.max())],
                "mean_h": AN.wmean(hh, w), "mean_h_ci": AN.boot(hh, w, g), "mean_d": AN.wmean(dd, w), "mean_d_ci": AN.boot(dd, w, g),
                "mean_residual": AN.wmean(res, w), "mean_residual_ci": AN.boot(res, w, g), "identity_max_abs_error": float(np.max(np.abs(hh - dd - res))),
                "frac_h_negative": AN.wmean((hh < 0).astype(float), w), "frac_h_exceeds_d": AN.wmean((hh > dd).astype(float), w),
                "pearson_h_d": float(np.corrcoef(hh, dd)[0, 1]) if hh.std() > 0 and dd.std() > 0 else None, "spearman_h_d": AN.spearman(hh, dd),
                "acc_masked": AN.wmean((np.sign(Ta) == C).astype(float), w),
                "rank_ceiling": AN.rank_ceiling(C, Ta, Tb, g, seed=int(hashlib.sha256(f"r7-40/{m}:{key}".encode()).hexdigest()[:8], 16)),
                "by_structural": {s: {"n": int(np.sum(sel)), "mean_h": AN.wmean(hh[sel], w[sel]), "mean_d": AN.wmean(dd[sel], w[sel])}
                                  for s in sorted({r["structural"] for r in rows if r["structural"]}) for sel in [np.array([r["structural"] == s for r in rows])]}}
    return out


def low_mass_by_type(rows, threshold=0.9):
    """QC (descriptive): items whose full-pass first-token yes / no mass is below the threshold, by GQA structural / semantic type."""
    by, allby = {}, {}
    for r in rows:
        k = f"{r.get('structural')}/{r.get('semantic')}"; allby[k] = allby.get(k, 0) + 1
        if sum(r["full"]) < threshold:
            by[k] = by.get(k, 0) + 1
    return {k: {"low": v, "of": allby[k]} for k, v in sorted(by.items(), key=lambda kv: -kv[1])}


def secondary_mass40(rows_by_role, weights, table_roles=("CAL",), threshold=0.9):
    out = {"threshold": threshold, "excluded_per_role": {}, "tables": {}, "low_mass_by_type": {}}
    for role, rows in rows_by_role.items():
        keep = [r for r in rows if sum(r["full"]) >= threshold]
        out["excluded_per_role"][role] = {"excluded": len(rows) - len(keep), "of": len(rows)}; out["low_mass_by_type"][role] = low_mass_by_type(rows, threshold)
        if role in table_roles:
            out["tables"][role] = analyse40(keep, weights)
    return out


def markdown(res):
    L = ["# R7-40a readout on Qwen2.5-VL-7B (generated by scripts/r7_d1_40_analyze.py)\n",
         "Budgets are fractions of each image's own merged-token count; 'budget' below is the fraction (mean kept tokens in the JSON).\n"]
    for role, R_ in res["roles"].items():
        vt = R_["visual_tokens"]
        L.append(f"## {role}: {R_['n_questions']} questions, {R_['n_images']} images (balanced weights) — primary: every item\n")
        L.append(f"Visual tokens per image: mean {vt['mean']:.1f}, median {vt['median']:.0f}, range {vt['min']}–{vt['max']}.\n")
        AN.role_tables(L, R_)
    sec = res.get("secondary_mass_ge_0p9")
    if sec:
        L.append("## Secondary: excludes yes/no mass < 0.9 (frozen D1 coordinator_note_20261012_yes_no_mass, re-applied)\n")
        L.append("Excluded items per role (full-pass first-token yes / no mass < 0.9): "
                 + "; ".join(f"{r}: {v['excluded']} of {v['of']}" for r, v in sec["excluded_per_role"].items()) + ".\n")
        for role, lm in sec["low_mass_by_type"].items():
            L.append(f"Low-mass items by type ({role}): " + (", ".join(f"{k} {v['low']}/{v['of']}" for k, v in list(lm.items())[:8]) or "none") + ".\n")
        for role, R_ in sec["tables"].items():
            L.append(f"### {role} without low-mass items: {R_['n_questions']} questions, {R_['n_images']} images\n")
            AN.role_tables(L, R_)
    return "\n".join(L) + "\n"


def cmd_readout(a):
    td, md = Path(a.teacher_dir), Path(a.manifest_dir)
    if "DEV" in a.roles and not a.allow_dev:
        raise SystemExit("DEV is sealed until R7-40c (--allow-dev)")
    weights = json.loads((md / "d1_manifest_meta.json").read_text())["class_weights_from_FIT"]
    res = {"host": "Qwen2.5-VL-7B-Instruct", "weights": weights, "roles": {}, "coverage": {}}; by_role = {}
    for role in a.roles:
        rows, cov = AN.load_role(td, md, role); res["roles"][role] = analyse40(rows, weights); res["coverage"][role] = cov; by_role[role] = rows
    res["secondary_mass_ge_0p9"] = secondary_mass40(by_role, weights, table_roles=("CAL",))
    (td / "r7_40a_results.json").write_text(V.jdump(res, indent=1)); (td / "r7_40a_results.md").write_text(markdown(res)); print(markdown(res))
    return 0


def cmd_complete(a):
    """Every shard of the requested roles has its done-record with present == expected, 0 duplicates, 0 skipped images; per-role totals equal
    the frozen manifest; manifest sha256 equals the frozen value.  Reads done-records and the manifest only (no teacher probabilities)."""
    td, md = Path(a.teacher_dir), Path(a.manifest_dir)
    plan = json.loads((md / "d1_shards.json").read_text()); meta = json.loads((md / "d1_manifest_meta.json").read_text())
    sha = hashlib.sha256((md / "d1_manifest.jsonl").read_bytes()).hexdigest()
    man = [json.loads(x) for x in (md / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]; per_role = {}
    for r in man:
        if r["role"] in a.roles:
            per_role[r["role"]] = per_role.get(r["role"], 0) + 1
    shards, bad, tot = {}, [], {}
    for s in plan["shards"]:
        if s.split("-")[0] not in a.roles:
            continue
        f = td / f"{s}.done.json"
        if not f.exists():
            bad.append(f"{s}: missing done-record"); continue
        d = json.loads(f.read_text()); shards[s] = {k: d[k] for k in ("expected", "present", "duplicates", "gpu")}; shards[s]["skipped"] = len(d["skipped_images"])
        if d["present"] != d["expected"] or d["duplicates"] or d["skipped_images"]:
            bad.append(f"{s}: {shards[s]}")
        tot[s.split("-")[0]] = tot.get(s.split("-")[0], 0) + d["present"]
    res = {"roles": list(a.roles), "shards": len([s for s in plan["shards"] if s.split("-")[0] in a.roles]), "done_records": len(shards), "problems": bad,
           "present_per_role": tot, "manifest_per_role": per_role, "manifest_sha256": sha, "frozen_sha256": a.frozen_sha256,
           "sha256_unchanged": sha == a.frozen_sha256 == meta["manifest_sha256"], "frozen_flag": meta.get("frozen"), "gpus": sorted({v["gpu"] for v in shards.values()})}
    res["complete"] = (not bad and res["sha256_unchanged"] and all(tot.get(k) == per_role[k] for k in per_role) and res["frozen_flag"] is True)
    print(json.dumps(res, indent=1)); (td / "r7_40a_completeness.json").write_text(json.dumps(res, indent=1))
    return 0 if res["complete"] else 7


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("readout"); r.add_argument("--teacher-dir", required=True); r.add_argument("--manifest-dir", required=True)
    r.add_argument("--roles", nargs="+", default=["CAL", "FIT"]); r.add_argument("--allow-dev", action="store_true")
    c = sub.add_parser("complete"); c.add_argument("--teacher-dir", required=True); c.add_argument("--manifest-dir", required=True)
    c.add_argument("--frozen-sha256", required=True); c.add_argument("--roles", nargs="+", default=["CAL", "FIT"])
    a = ap.parse_args()
    return {"readout": cmd_readout, "complete": cmd_complete}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
