"""r7 E2 readout (CPU; SparseVLM env as R7-40c's scorer, for the m4c TextVQA evaluator).  The coordinator's step after the E2 driver's done
line; refuses unless the frozen D1 protocol carries the top-level keys 'r7_40c_staged_freeze', 'r7_e2a_staged_freeze' (teacher stage) and
'r7_e2_staged_freeze' (cache, selectors, evaluation, readout); never overwrites.
  1. the frozen R7-40c readout (scripts/r7_d1_40c_score.py, imported unchanged) on E2's rows -> <out>/r7_40c_results.{json,md}: E2's 45 selector
     rows, E2's in-job full row and the uniform-grid rows (R7-40c's, copied after the reuse check, or E2's own with ROWS=core).  Qwen has no B*
     (R7-40a reference timing, job 1035608): primary = the fixed-fraction family S1 − S2, S1 − S3, S1 − grid × 3 benchmarks at 0.25 / 0.5 / 0.75,
     Holm over 9 per point.  No timing (E2 runs no latency unit), no batch-1 anchor (--anchor none).
  2. the claim rule E2-8 (the Qwen analogue of D1-FULL's DF-8) -> <out>/e2_claims.{json,md}: per primary contrast the 10k scale (R7-40c's frozen
     readout) and the full scale (E2) side by side; a claim only where both scales agree in sign and both pass their own Holm-corrected tests
     (alpha 0.05); the full-scale numbers are the headline; any disagreement is reported as scale-dependent.
  3. the descriptive scale contrast (in e2_claims.json): full − 10k per objective (S1, S2, S3) at each fixed fraction (incl. the 0.125 stress
     point) = the seed-mean per-item score difference (seed i vs seed i) with the shared image / chart-grouped paired bootstrap (2,000 replicates,
     seed 0); S1 dynamic rows: quality and realised mean kept fraction per scale and seed (no contrast: their budgets differ).  No multiplicity
     adjustment, no claim.
    python -I scripts/r7_e2_score.py [--out O] [--r40c-results R]      (defaults: E2 / R7-40c paths)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "r7" / "package_r7_20261011" / "reference"))
import r7_d1_30_score as SC  # noqa: E402  (frozen D1-30 statistics)
import r7_d1_40c_plan as PL  # noqa: E402  (frozen R7-40c grammar)
import r7_d1_40c_score as SCO  # noqa: E402  (frozen R7-40c readout)
import r7_vl_common as V  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
R40 = OUT / "R7_D1" / "r7_40"
R40C = R40 / "eval_40c"
E2 = OUT / "R7_E2"
PROTOCOL = HERE.parent / "reports" / "R7" / "R7_D1_PROTOCOL_FROZEN_20261012.yaml"
KEYS = ("r7_40c_staged_freeze", "r7_e2a_staged_freeze", "r7_e2_staged_freeze")
ALPHA = 0.05
RULE = ("E2-8 (DF-8 analogue): a Qwen D1 claim is made only where the two Qwen scales (10k FIT: R7-40c; FIT_FULL: E2) agree in sign and both pass "
        "their own Holm-corrected primary tests (alpha 0.05); the full-scale numbers are the headline; any disagreement is reported as scale-dependent; "
        "the cross-host reading (LLaVA D1-FULL vs Qwen E2) stays descriptive: no pooling, no cross-host test")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require_keys(protocol):
    txt = Path(protocol).read_text() if Path(protocol).exists() else ""
    missing = [k for k in KEYS if not re.search(rf"^{k}\s*:", txt, re.M)]
    if missing:
        raise SystemExit(f"the E2 readout refuses to run: top-level key(s) {missing} absent in {protocol}")


def verdict(c10, cf, alpha=ALPHA):
    """c10 / cf: the 10k (R7-40c) and full-scale (E2) contrast records (delta in points, p_holm)."""
    if not c10 or not cf or "p_holm" not in c10 or "p_holm" not in cf:
        return "missing"
    s10, sf = np.sign(c10["delta"]), np.sign(cf["delta"]); p10, pf = c10["p_holm"] < alpha, cf["p_holm"] < alpha
    if p10 and pf and s10 == sf and sf != 0:
        return "claim: S1 above" if sf > 0 else "claim: S1 below"
    if p10 and pf:
        return "scale-dependent: significant with opposite signs"
    if p10 or pf:
        return f"not claimed: significant at the {'10k' if p10 else 'full'} scale only (scale-dependent)"
    return "not claimed: neither scale significant"


def claims(r10, rf, alpha=ALPHA):
    keep = ("delta", "ci95", "p", "p_holm", "holm_family_size", "seed_deltas", "seed_sd")
    rows = []
    for b, Bf in rf["benches"].items():
        B10 = r10.get("benches", {}).get(b, {}).get("contrasts", {})
        for k, cf in sorted(Bf["contrasts"].items()):
            if cf.get("role") != "primary":
                continue
            c10 = B10.get(k)
            rows.append({"bench": b, "contrast": k, "full": {x: cf.get(x) for x in keep},
                         "k10": None if c10 is None else dict({x: c10.get(x) for x in keep}, role=c10.get("role")), "verdict": verdict(c10, cf, alpha)})
    return rows


def scale_contrast(bench, sets_dir, e2_pred, k10_pred, e2_main, k10_main, reps):
    """Descriptive full − 10k per objective at each fixed fraction (paired items, one shared bootstrap) + the S1 dynamic rows per scale."""
    cfgs = lambda m: [f"sel:{u}:{f}" for o in PL.OBJECTIVES for u in m[o] for f in PL.FKEYS] + [f"dyn:{u}:{op}" for u in m["vcs"] for op in PL.OPS]  # noqa: E731
    bE = SCO.Bench40(bench, sets_dir, e2_pred, None, cfgs(e2_main), reps, e2_main["vcs"][0])
    bK = SCO.Bench40(bench, sets_dir, k10_pred, None, cfgs(k10_main), reps, k10_main["vcs"][0])
    if bE.qids != bK.qids or not np.array_equal(bE.boot.W, bK.boot.W):
        raise SystemExit(f"{bench}: the two scales do not share items / bootstrap replicates")
    out = {"fixed_fraction": {}, "dynamic": {}, "role": "descriptive (no multiplicity adjustment, no claim)"}
    for o in PL.OBJECTIVES:
        for f in PL.FKEYS:
            a = [f"sel:{u}:{f}" for u in e2_main[o]]; b = [f"sel:{u}:{f}" for u in k10_main[o]]
            d = np.mean([bE.S[c] for c in a], 0) - np.mean([bK.S[c] for c in b], 0)
            r = SC.contrast(d.mean(), bE.boot.means(d), [bE.q(x) - bK.q(y) for x, y in zip(a, b)])
            r.update(full_quality=100 * float(np.mean([bE.q(c) for c in a])), k10_quality=100 * float(np.mean([bK.q(c) for c in b])),
                     per_seed=SC.per_seed([bE.q(c) for c in a], [bE.bm(c) for c in a], [bK.q(c) for c in b], [bK.bm(c) for c in b]))
            out["fixed_fraction"][f"{PL.NAMES[o]}@f{f}"] = r
    for op in PL.OPS:
        out["dynamic"][f"S1_dyn@op{op}"] = {side: {"quality": [100 * bn.q(c) for c in (f"dyn:{u}:{op}" for u in m["vcs"])],
                                                   "mean_kept_fraction": [bn.kept_fraction(c) for c in (f"dyn:{u}:{op}" for u in m["vcs"])]}
                                            for side, bn, m in (("full", bE, e2_main), ("k10", bK, k10_main))}
    return out


def markdown(res):
    L = [f"# r7 E2 claims (Qwen2.5-VL-7B, 10k FIT vs FIT_FULL) — {res['created_utc']}", "", res["claim_rule"], "",
         "| bench | contrast | 10k Δ [95% CI] p_holm | full Δ [95% CI] p_holm | verdict |", "|---|---|---|---|---|"]
    f = lambda c: "—" if not c else f"{c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] {c.get('p_holm', float('nan')):.3g}"  # noqa: E731
    for r in res["claims"]:
        L.append(f"| {r['bench']} | {r['contrast']} | {f(r['k10'])} | {f(r['full'])} | {r['verdict']} |")
    L += ["", "Scale contrast full − 10k (descriptive):", "", "| bench | family@fraction | 10k | full | Δ [95% CI] |", "|---|---|---|---|---|"]
    for b, S in res["scale_contrast"].items():
        for k, c in S["fixed_fraction"].items():
            L.append(f"| {b} | {k} | {c['k10_quality']:.2f} | {c['full_quality']:.2f} | {c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", default=str(PL.D30_SETS)); ap.add_argument("--e2-root", default=str(E2))
    ap.add_argument("--r40c", default=str(R40C)); ap.add_argument("--r40-lr-choice", default=str(R40 / "selectors" / "lr_choice.json"))
    ap.add_argument("--r40c-results", default=str(R40C / "results" / "r7_40c_results.json")); ap.add_argument("--out", default=str(E2 / "eval" / "results"))
    ap.add_argument("--benches", nargs="+", default=list(PL.BENCHES)); ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--protocol", default=str(PROTOCOL)); ap.add_argument("--timing-report", default=str(PL.TIMING_REPORT))
    a = ap.parse_args(argv)
    require_keys(a.protocol); out = Path(a.out); e2 = Path(a.e2_root)
    if (out / "e2_claims.json").exists():
        raise SystemExit(f"{out / 'e2_claims.json'} exists: a re-verification writes to a new --out directory")
    SCO.main(["--sets-dir", a.sets_dir, "--pred-dir", str(e2 / "eval" / "pred"), "--configs-dir", str(e2 / "eval" / "configs"), "--lr-choice",
              str(e2 / "selectors" / "lr_choice.json"), "--out", str(out), "--benches", *a.benches, "--reps", str(a.reps), "--protocol", a.protocol,
              "--timing-report", a.timing_report, "--anchor", "none"])
    rf = json.loads((out / "r7_40c_results.json").read_text()); r10 = json.loads(Path(a.r40c_results).read_text())
    if rf["primary_basis"] != "fraction" or r10.get("primary_basis") != rf["primary_basis"]:
        raise SystemExit(f"primary bases differ or are not the fixed-fraction basis (E2 {rf['primary_basis']}, R7-40c {r10.get('primary_basis')})")
    rows = claims(r10, rf, a.alpha)
    e2_main = PL.unit_names(json.loads((e2 / "selectors" / "lr_choice.json").read_text())); k_main = PL.unit_names(json.loads(Path(a.r40_lr_choice).read_text()))
    res = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "claim_rule": RULE, "alpha": a.alpha, "claims": rows,
           "verdict_counts": dict(Counter(r["verdict"] for r in rows)), "units": {"full": e2_main, "k10": k_main},
           "sources": {"e2_results": str(out / "r7_40c_results.json"), "e2_results_sha256": sha(out / "r7_40c_results.json"), "r7_40c_results": a.r40c_results,
                       "r7_40c_results_sha256": sha(a.r40c_results)},
           "scale_contrast": {b: scale_contrast(b, a.sets_dir, e2 / "eval" / "pred", Path(a.r40c) / "pred", e2_main, k_main, a.reps) for b in a.benches}}
    (out / "e2_claims.json").write_text(V.jdump(res, indent=1)); (out / "e2_claims.md").write_text(markdown(res))
    print(f"[r7_e2_score] {len(rows)} primary contrasts; verdicts {res['verdict_counts']} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
