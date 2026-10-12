"""r7 R7-30 D1-10 readout: finite teacher residuals of the binary risk object (package D1-10; theory doc §3).
For every record: C ∈ {−1,+1}; b = T(full) = 2 p̃ − 1, a = T(masked), p̃ = P(yes) / (P(yes) + P(no));
h = (C − a)² − (C − b)² (actual loss change), d = (b − a)² (squared disagreement), h − d = 2 (C − b)(b − a) (finite-model residual; its mean is
0 for a calibrated full teacher).  All means are taken under the balanced task law M (FIT class weights from the manifest meta);
intervals resample IMAGES (2000 reps).  CAL is the pre-registered readout; FIT is descriptive; DEV stays sealed (--allow-dev only at D1-30).
The analyser refuses partial coverage.
    python scripts/r7_d1_analyze.py --teacher-dir OUT --manifest-dir DIR [--roles CAL FIT]
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
import r7_d1_masks as MK  # noqa: E402
import r7_vl_common as V  # noqa: E402
import vcs_vl_reference as REF  # noqa: E402


def T(pair):
    y, n = pair
    return 2.0 * y / max(y + n, 1e-12) - 1.0


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def boot(x, w, g, reps=2000, seed=0):
    """Image-cluster bootstrap of a weighted mean."""
    x, w, g = np.asarray(x, float), np.asarray(w, float), np.asarray(g); ug, inv = np.unique(g, return_inverse=True)
    sx, sw = np.bincount(inv, weights=w * x, minlength=len(ug)), np.bincount(inv, weights=w, minlength=len(ug))
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(ug), size=(reps, len(ug)))
    est = sx[idx].sum(1) / sw[idx].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else None


def rank_ceiling(C, Ta, Tb, groups, draws=200, boot=1000, seed=0):
    """Coordinator note 2026-10-12 (rank criterion), fixed before any CAL readout.  Calibrated-teacher ceiling: redraw C* ~ Bernoulli(p̃_full)
    with p̃_full = (T_b + 1) / 2, recompute Spearman(h*, d) per draw (d does not involve C); 2.5–97.5 % band over `draws`.  Observed
    Spearman(h, d) with an image-clustered bootstrap; "d does not rank per-question losses" only if the bootstrap 97.5 % point (observed +
    its margin) lies below the band's lower end.  The superseded 0.5 line is reported alongside."""
    C, Ta, Tb, g = (np.asarray(x) for x in (C, Ta, Tb, groups)); rng = np.random.default_rng(seed)
    d = (Tb - Ta) ** 2; obs = spearman((C - Ta) ** 2 - (C - Tb) ** 2, d); pt = np.clip((Tb + 1) / 2, 0, 1); ceil = []
    for _ in range(draws):
        Cs = np.where(rng.uniform(size=len(pt)) < pt, 1.0, -1.0); ceil.append(spearman((Cs - Ta) ** 2 - (Cs - Tb) ** 2, d))
    lo, hi = (float(np.nanpercentile(ceil, q)) for q in (2.5, 97.5))
    ug, inv = np.unique(g, return_inverse=True); members = [np.flatnonzero(inv == j) for j in range(len(ug))]; bs = []
    hh = (C - Ta) ** 2 - (C - Tb) ** 2
    for _ in range(boot):
        idx = np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))]); bs.append(spearman(hh[idx], d[idx]))
    b_lo, b_hi = (float(np.nanpercentile(bs, q)) for q in (2.5, 97.5))
    return {"observed": obs, "ceiling_band": [lo, hi], "ceiling_median": float(np.nanmedian(ceil)), "observed_boot_ci": [b_lo, b_hi],
            "margin": b_hi - obs, "verdict": "d_does_not_rank_per_question_losses" if b_hi < lo else "not_below_calibrated_ceiling",
            "rule_0p5_superseded": {"observed_below_0p5": bool(obs < 0.5), "label": "superseded (unattainable for a calibrated teacher)"},
            "draws": draws, "boot": boot}


def load_role(teacher_dir: Path, manifest_dir: Path, role: str):
    man = [json.loads(x) for x in (manifest_dir / "d1_manifest.jsonl").read_text().splitlines() if x.strip()]
    expected = {r["qid"] for r in man if r["role"] == role}; plan = json.loads((manifest_dir / "d1_shards.json").read_text())
    rows, skipped_q = {}, 0
    for shard in plan["shards"]:
        if not shard.startswith(role + "-"):
            continue
        for r in V.read_jsonl(teacher_dir / f"{shard}.jsonl"):
            rows[r["qid"]] = r
        done = teacher_dir / f"{shard}.done.json"
        if done.exists():
            sk = {s["image_id"] for s in json.loads(done.read_text()).get("skipped_images", [])}
            skipped_q += sum(1 for r in man if r["role"] == role and r["image_id"] in sk)
    missing = len(expected - set(rows)) - skipped_q
    if missing > 0 or set(rows) - expected:
        raise RuntimeError(f"coverage guard: role {role}: expected {len(expected)}, present {len(set(rows) & expected)}, "
                           f"reason-coded skips {skipped_q}, unexpected {len(set(rows) - expected)}")
    return [rows[q] for q in sorted(rows)], {"expected": len(expected), "present": len(rows), "skipped_questions": skipped_q}


def analyse(rows, weights):
    C = np.array([r["C"] for r in rows], float); g = np.array([r["image_id"] for r in rows]); w = np.where(C > 0, weights["w_yes"], weights["w_no"])
    Tb = np.array([T(r["full"]) for r in rows]); Tt = np.array([T(r["text"]) for r in rows])
    out = {"n_questions": len(rows), "n_images": int(len(np.unique(g))),
           "teacher": {"brier_full": wmean((C - Tb) ** 2, w), "brier_full_ci": boot((C - Tb) ** 2, w, g),
                       "brier_text_only": wmean((C - Tt) ** 2, w), "acc_full": wmean((np.sign(Tb) == C).astype(float), w),
                       "acc_text_only": wmean((np.sign(Tt) == C).astype(float), w),
                       "mean_residual_text_only": wmean(REF.finite_risk_delta(C, Tt, Tb)[2], w)}, "cells": {}}
    for m in MK.MECHANISMS:
        for k in MK.BUDGETS:
            Ta = np.array([T(r["masked"][m][str(k)]) for r in rows]); hh, dd, res = REF.finite_risk_delta(C, Ta, Tb)
            out["cells"][f"{m}:{k}"] = {
                "mechanism": m, "budget_tokens": k, "budget_fraction": MK.FRACTIONS[k],
                "mean_h": wmean(hh, w), "mean_h_ci": boot(hh, w, g), "mean_d": wmean(dd, w), "mean_d_ci": boot(dd, w, g),
                "mean_residual": wmean(res, w), "mean_residual_ci": boot(res, w, g), "identity_max_abs_error": float(np.max(np.abs(hh - dd - res))),
                "frac_h_negative": wmean((hh < 0).astype(float), w), "frac_h_exceeds_d": wmean((hh > dd).astype(float), w),
                "pearson_h_d": float(np.corrcoef(hh, dd)[0, 1]) if hh.std() > 0 and dd.std() > 0 else None, "spearman_h_d": spearman(hh, dd),
                "acc_masked": wmean((np.sign(Ta) == C).astype(float), w),
                "rank_ceiling": rank_ceiling(C, Ta, Tb, g, seed=int(hashlib.sha256(f"{m}:{k}".encode()).hexdigest()[:8], 16)),
                "by_structural": {s: {"n": int(np.sum(sel)), "mean_h": wmean(hh[sel], w[sel]), "mean_d": wmean(dd[sel], w[sel])}
                                  for s in sorted({r["structural"] for r in rows if r["structural"]}) for sel in [np.array([r["structural"] == s for r in rows])]}}
    return out


def markdown(res):
    L = ["# R7-30 D1-10 readout (generated by scripts/r7_d1_analyze.py)\n"]
    for role, R_ in res["roles"].items():
        t = R_["teacher"]
        L.append(f"## {role}: {R_['n_questions']} questions, {R_['n_images']} images (balanced weights)\n")
        L.append(f"Teacher: balanced Brier full {t['brier_full']:.4f} {t['brier_full_ci']}, text-only {t['brier_text_only']:.4f}; "
                 f"accuracy full {t['acc_full']:.3f}, text-only {t['acc_text_only']:.3f}.\n")
        L.append("| mechanism | budget | mean h [95 %] | mean d [95 %] | mean h − d [95 %] | P(h<0) | P(h>d) | Pearson / Spearman (h,d) | acc |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in R_["cells"].values():
            f = lambda v, ci: f"{v:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"  # noqa: E731
            L.append(f"| {c['mechanism']} | {c['budget_fraction']:.3g} | {f(c['mean_h'], c['mean_h_ci'])} | {f(c['mean_d'], c['mean_d_ci'])} | "
                     f"{f(c['mean_residual'], c['mean_residual_ci'])} | {c['frac_h_negative']:.2f} | {c['frac_h_exceeds_d']:.2f} | "
                     f"{c['pearson_h_d'] if c['pearson_h_d'] is None else round(c['pearson_h_d'], 2)} / "
                     f"{c['spearman_h_d'] if c['spearman_h_d'] is None else round(c['spearman_h_d'], 2)} | {c['acc_masked']:.3f} |")
        L.append("\nPer-item rank criterion (coordinator note 2026-10-12): observed Spearman(h, d) [image-clustered 95 %] against the "
                 "calibrated-teacher ceiling band (C* ~ Bernoulli(p̃_full), 200 draws); the 0.5 line is superseded.\n")
        L.append("| mechanism | budget | observed [95 %] | ceiling band (median) | verdict | 0.5 line (superseded) |"); L.append("|---|---|---|---|---|---|")
        for c in R_["cells"].values():
            rc = c["rank_ceiling"]
            L.append(f"| {c['mechanism']} | {c['budget_fraction']:.3g} | {rc['observed']:.3f} [{rc['observed_boot_ci'][0]:.3f}, {rc['observed_boot_ci'][1]:.3f}] | "
                     f"[{rc['ceiling_band'][0]:.3f}, {rc['ceiling_band'][1]:.3f}] ({rc['ceiling_median']:.3f}) | {rc['verdict']} | "
                     f"{'below 0.5' if rc['rule_0p5_superseded']['observed_below_0p5'] else 'at or above 0.5'} |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher-dir", required=True); ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--roles", nargs="+", default=["CAL", "FIT"]); ap.add_argument("--allow-dev", action="store_true")
    a = ap.parse_args(); td, md = Path(a.teacher_dir), Path(a.manifest_dir)
    if "DEV" in a.roles and not a.allow_dev:
        raise SystemExit("DEV is sealed until D1-30 (--allow-dev)")
    weights = json.loads((md / "d1_manifest_meta.json").read_text())["class_weights_from_FIT"]; res = {"weights": weights, "roles": {}, "coverage": {}}
    for role in a.roles:
        rows, cov = load_role(td, md, role); res["roles"][role] = analyse(rows, weights); res["coverage"][role] = cov
    (td / "d1_10_results.json").write_text(V.jdump(res, indent=1)); (td / "d1_10_results.md").write_text(markdown(res)); print(markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
