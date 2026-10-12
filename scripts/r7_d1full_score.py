"""r7 R7-30 D1-FULL readout (CPU, SparseVLM env for the m4c evaluator).  Run by the coordinator ONLY AFTER the D1-30 readout: it reads D1-30's
prediction and timing roots.  It refuses unless the frozen protocol carries both 'd1_30_staged_freeze' and 'd1_full_staged_freeze'.
Rows: full / static / SparseVLM (+ TextVQA OCR anchors) and the D1-20 (10k-FIT) selector rows from the D1-30 prediction root; the D1-FULL
selector rows (same tags, trained on FIT_FULL) from the D1-FULL prediction root, re-keyed 'F-<tag>'.  One image (chart) bootstrap per benchmark
(frozen D1-30 Boot, 2,000 replicates, seed 0) is shared by every statistic, so both scales are paired.  Latency: D1-30's timing (same
architecture: every sel row takes the representative unit's sel:k latency, as in D1-30); the D1-FULL dynamic rows take their own latency
from the D1-FULL timing units (frozen D1-30 timing procedure, nodesumo01; coordinator 2026-10-12) — the readout refuses if any is missing
(--allow-missing-dyn-timing only for a labelled partial readout).
  primary (D1-FULL, pre-registration draft): S1F - S2F and S1F - S3F at L25 / L50 / L75 (B* = 4) per benchmark; Holm across the 6 contrasts at each
          latency point.  Every frozen D1-30 secondary is computed for the D1-FULL units with the frozen scorer (score_bench); its S1 - SparseVLM
          rows are secondary here.
  scale (descriptive): full - 10k per objective (seed means of the per-seed interpolated quality; paired bootstrap; per-seed deltas) at L25 / L50 / L75
          and at each fixed budget (S1-S3, A1, A2); dynamic rows: quality and realised mean tokens per scale (no contrast: budgets differ).
  10k   : the D1-30 primary family recomputed with the same functions (it must equal the D1-30 readout) for the side-by-side table.
    python scripts/r7_d1full_score.py [--d30-root R --df-root F --out O --benches ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_30_score as SC  # noqa: E402   (frozen D1-30 scorer; read-only use)
import r7_vl_common as V  # noqa: E402

OUT = Path("/home/infres/yinwang/CS_QMI/outputs")
D1, DF = OUT / "R7_D1", OUT / "R7_D1_FULL"
PROTOCOL = HERE.parent / "reports" / "R7" / "R7_D1_PROTOCOL_FROZEN_20261012.yaml"
PFX = "F-"


def fkey(cfg: str) -> str:
    kind, u, x = cfg.split(":")
    return f"{kind}:{PFX}{u}:{x}"


def renamed(names: dict):
    return {k: [PFX + u for u in v] for k, v in names.items()}


def add_full_rows(bn, df_pred: Path, cfgs, df_timing: Path | None):
    """D1-FULL selector rows into the D1-30 Bench (coverage guard as frozen)."""
    for c in cfgs:
        recs, integ = SC.load_config(df_pred / bn.bench, c); preds = {q: r["pred"] for q, r in recs.items()}
        if set(preds) != set(bn.qids):
            raise SystemExit(f"coverage guard: D1-FULL {bn.bench} {c}: {len(set(preds) & set(bn.qids))} / {len(bn.qids)} items, "
                             f"{len(set(preds) - set(bn.qids))} unexpected; integrity {integ}")
        sc = SC.item_scores(bn.bench, bn.items, preds); kept = {q: r.get("kept", RUN.N_VIS) for q, r in recs.items()}; k = fkey(c)
        bn.S[k] = np.array([sc[q] for q in bn.qids]); bn.K[k] = np.array([kept[q] for q in bn.qids], float); bn.integrity[k] = integ
    timed = 0
    if df_timing is not None:
        p = next((q for q in (df_timing / bn.bench / f"timing_{bn.bench}.json", df_timing / f"timing_{bn.bench}.json") if q.exists()), None)
        if p is not None:
            for r in json.loads(p.read_text())["rows"]:
                if r["config"].startswith("dyn:"):
                    bn.T[(fkey(r["config"]), r["B"])] = r["per_item_ms_median"]; bn.Tci[(fkey(r["config"]), r["B"])] = r.get("per_item_ms_median_ci95"); timed += 1
    return timed


def scale_contrasts(bn, m10, a10, mF, aF):
    F10, FF = SC.families(m10, a10), SC.families(mF, aF); C = {}
    for name in ("S1", "S2", "S3"):
        for L, k in SC.LPOINTS.items():
            Lms = bn.lat(f"static:grid:{k}", SC.PRIMARY_B)
            a = None if Lms is None else SC.seed_mean_at(bn, FF[name], Lms, SC.PRIMARY_B); b = None if Lms is None else SC.seed_mean_at(bn, F10[name], Lms, SC.PRIMARY_B)
            key = f"{name}:full-10k@{L}"
            if a is None or b is None:
                C[key] = {"not_reached": True, "role": "descriptive"}; continue
            C[key] = dict(SC.contrast(a[0] - b[0], a[1] - b[1], [x - y for x, y in zip(a[2], b[2])]), role="descriptive",
                          full=100 * a[0], k10=100 * b[0])
    for k in SC.BUDGETS:
        for name, us_f, us_10 in [(SC.NAMES[o], mF[o], m10[o]) for o in m10] + [(x, aF[x], a10[x]) for x in a10]:
            xa, xb = [f"sel:{u}:{k}" for u in us_f], [f"sel:{u}:{k}" for u in us_10]
            if all(c in bn.S for c in xa + xb):
                d = np.mean([bn.S[c] for c in xa], 0) - np.mean([bn.S[c] for c in xb], 0)
                C[f"{name}:full-10k@k{k}"] = dict(SC.contrast(d.mean(), bn.boot.means(d), [bn.q(p) - bn.q(q) for p, q in zip(xa, xb)]), role="descriptive")
    dyn = {}
    for o in m10:
        for f in RUN.FRACTIONS:
            for scale, us in (("full", mF[o]), ("10k", m10[o])):
                cs = [f"dyn:{u}:{f}" for u in us]
                if all(c in bn.S for c in cs):
                    dyn[f"{SC.NAMES[o]}_dyn@f{f}:{scale}"] = {"quality": 100 * float(np.mean([bn.q(c) for c in cs])),
                                                             "mean_visual_tokens": float(np.mean([bn.K[c].mean() for c in cs])), "seed_quality": [100 * bn.q(c) for c in cs]}
    return C, dyn


def require(protocol: Path):
    t = Path(protocol).read_text()
    for key in ("d1_30_staged_freeze", "d1_full_staged_freeze"):
        if key not in t:
            raise SystemExit(f"{key} absent in {protocol}: the D1-FULL readout is not allowed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--d30-root", default=str(D1 / "d1_30")); ap.add_argument("--d30-lr-choice", default=str(D1 / "selectors" / "lr_choice.json"))
    ap.add_argument("--df-root", default=str(DF / "d1_30")); ap.add_argument("--df-lr-choice", default=str(DF / "selectors" / "lr_choice.json"))
    ap.add_argument("--out", default=str(DF / "d1_30" / "results")); ap.add_argument("--benches", nargs="+", default=["gqa_dev", "textvqa_val", "chartqa_val"])
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--no-sparsevlm", action="store_true"); ap.add_argument("--protocol", default=str(PROTOCOL))
    ap.add_argument("--allow-missing-dyn-timing", action="store_true")
    a = ap.parse_args(argv); require(Path(a.protocol))
    import r7_d1_30_sparsevlm as SVM
    d30, dfr = Path(a.d30_root), Path(a.df_root)
    m10, a10 = RUN.unit_names(json.loads(Path(a.d30_lr_choice).read_text())); mF0, aF0 = RUN.unit_names(json.loads(Path(a.df_lr_choice).read_text()))
    mF, aF = renamed(mF0), renamed(aF0); rep_unit = m10["vcs"][0]
    cfgF = [c for c in RUN.config_list([u for o in RUN.OBJECTIVES for u in mF0[o]], aF0["A1"] + aF0["A2"]) if c.startswith(("sel:", "dyn:"))]
    res = {"representative_unit_for_latency": rep_unit, "unit_prefix_full_scale": PFX, "benches": {}, "notes": [
        "D1-FULL primary: S1F - S2F, S1F - S3F at L25 / L50 / L75 (B* = 4); Holm across the 6 contrasts per latency point",
        "scale contrasts full - 10k are descriptive (no multiplicity adjustment, no claim)",
        "sel rows of both scales take the D1-30 representative unit's measured sel:k latency (same architecture)",
        "the 10k block recomputes the frozen D1-30 primary family with the same functions and bootstrap; it must equal the D1-30 readout"]}
    for bench in a.benches:
        exp = RUN.config_list([u for o in m10.values() for u in o], [u for o in a10.values() for u in o], ocr_anchor=bench.startswith("textvqa"))
        sv = [] if a.no_sparsevlm else [SVM.row_tag(T, o) for T, o in SVM.rows_for(bench, list(SVM.ANCHORS + SVM.ADAPTED))]
        bn = SC.Bench(bench, d30 / "sets", d30 / "pred", d30 / "timing", exp + sv, a.reps, rep_unit)
        timed = add_full_rows(bn, dfr / "pred", cfgF, dfr / "timing")
        miss = [(fkey(c), B) for c in cfgF if c.startswith("dyn:") for B in sorted({B for (_, B) in bn.T}) if (fkey(c), B) not in bn.T]
        if miss and not a.allow_missing_dyn_timing:                                     # coordinator 2026-10-12: the D1-FULL dynamic rows carry their own latency
            raise SystemExit(f"D1-FULL dynamic-row timing missing for {len(miss)} (row, B), e.g. {miss[:3]} (timing unit on nodesumo01)")
        RF = SC.score_bench(bn, mF, aF, sv, "budget")
        for k, v in RF["contrasts"].items():                                            # D1-FULL primary family: S1 - S2, S1 - S3 only
            if k.startswith("S1-SparseVLM"):
                v["role"] = "secondary"
        R10 = SC.score_bench(bn, m10, a10, sv, "budget"); sc, dyn = scale_contrasts(bn, m10, a10, mF, aF)
        res["benches"][bench] = {"full_scale": RF, "k10_primary_family": {k: v for k, v in R10["contrasts"].items() if v.get("role") == "primary"},
                                 "k10_at_latency": R10["at_latency"], "scale_contrasts": sc, "dynamic_rows_by_scale": dyn,
                                 "full_scale_dynamic_rows_timed": timed}
    for L in SC.LPOINTS:                                                                # Holm: 6 D1-FULL primaries per latency point; 9 for the 10k family
        for blk, fam in (("full_scale", "contrasts"), ("k10_primary_family", None)):
            keys = []
            for b, r in res["benches"].items():
                d = r[blk][fam] if fam else r[blk]
                keys += [(b, k) for k, v in d.items() if f"@{L}" in k and v.get("role") == "primary" and "p" in v]
            if keys:
                get = (lambda b, k: res["benches"][b][blk][fam][k]) if fam else (lambda b, k: res["benches"][b][blk][k])
                for (b, k), pv in zip(keys, SC.holm([get(b, k)["p"] for b, k in keys])):
                    get(b, k)["p_holm"] = pv; get(b, k)["holm_family_size"] = len(keys)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "d1full_results.json").write_text(V.jdump(res, indent=1))
    summ = {b: {"full_primary": {k: v for k, v in r["full_scale"]["contrasts"].items() if v.get("role") == "primary"},
                "scale": {k: v for k, v in r["scale_contrasts"].items() if "@L" in k}} for b, r in res["benches"].items()}
    print(V.jdump(summ, indent=1)[:8000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
