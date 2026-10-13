"""r7 R7-40c readout (CPU; SparseVLM env for the m4c TextVQA evaluator, as D1-30).  Refuses to run unless the coordinator's freeze key
'r7_40c_staged_freeze' is in the frozen D1 protocol.  The frozen D1-30 readout (scripts/r7_d1_30_score.py, imported unchanged) re-applied to
the Qwen2.5-VL rows: official per-item scores (GQA exact match after LLaVA's convert_gqa_for_eval normalisation; TextVQA m4c soft accuracy;
ChartQA relaxed accuracy pinned to pix2struct 6fe25c1), the coverage guard (every configured row covers exactly the benchmark's items; first
complete record per qid, dropped / duplicate lines counted: E1), image- (GQA, TextVQA) / chart-grouped (ChartQA) paired bootstrap with 2,000
replicates shared by every statistic of a benchmark, linear interpolation of each family's (latency, quality) points incl. full, seed handling
as D1-30 (seed mean of per-seed interpolated qualities; per-seed contrasts with their own CI; the p-value entering Holm is the group
bootstrap of the seed-averaged contrast, seeds fixed; two-sided with +1 correction).
Primary family (R7-40 draft, for_40c): S1 − S3, S1 − S2 and S1 − uniform grid × 3 benchmarks = 9 contrasts per point, Holm per point.
  - Qwen has a B* (R7-40a reference timing, D40a-7): points L25 / L50 / L75 = uniform-grid 0.25 / 0.5 / 0.75 per-item latencies at B*
    (nodesumo01), quality at fixed measured latency; fixed-fraction contrasts are secondary.
  - no B* (OPEN-R740-8): points = fixed fractions 0.25 / 0.5 / 0.75 (quality at fixed kept-token fraction), no acceleration claim; latency
    rows (if measured) are secondary.
Secondaries: every family at L25 / L50 / L75 for B = 1 / 4 / 16 / 32; fixed-fraction contrasts incl. the 0.125 stress point; S1 − random /
qsim (if run); A3 dynamic − fixed at the dynamic row's realised mean kept fraction; fixed-fraction table (quality, retention, mean visual
tokens, mean kept fraction, answers cut at max_new_tokens); latency to stay within 1.0 point of full at the primary batch; preserve / correct
when full is wrong; dynamic fraction distribution; GQA structural types, ChartQA human / augmented; integrity; the batch-1 official
full-row anchor next to the batched full row (decision 15: quality with CI, full − anchor, identical-answer fraction); cross-host side-by-side with
the D1-30 LLaVA readout (descriptive only: no pooling, no cross-host test; only when --d1-30-results is given).
    python scripts/r7_d1_40c_score.py --sets-dir S --pred-dir P --timing-dir T --configs-dir C --lr-choice L --out O [--protocol F]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_score as SC  # noqa: E402  (frozen D1-30 readout: evaluators, loader, bootstrap, interpolation, contrasts, Holm)
import r7_d1_40c_plan as PL  # noqa: E402
import r7_vl_common as V  # noqa: E402

FREEZE_KEY = "r7_40c_staged_freeze"
PROTOCOL = Path(__file__).resolve().parents[1] / "reports" / "R7" / "R7_D1_PROTOCOL_FROZEN_20261012.yaml"
LPOINTS = {"L25": "0.25", "L50": "0.5", "L75": "0.75"}
FPOINTS = {"f0.25": "0.25", "f0.5": "0.5", "f0.75": "0.75"}
PRIMARY_OTHERS = ("S2", "S3", "grid")


def require_freeze(protocol):
    p = Path(protocol)
    if not p.exists() or FREEZE_KEY not in p.read_text():
        raise SystemExit(f"R7-40c is not frozen ({FREEZE_KEY} absent in {p}): the readout refuses to run")


class Bench40(SC.Bench):
    """The frozen D1-30 Bench (scores, bootstrap, latency curves) with Qwen's per-item token counts: K kept tokens, N the image's merged
    tokens, F the configuration's fraction (chosen fraction for dynamic rows), X answers cut at max_new_tokens."""

    def __init__(self, bench, sets_dir, pred_dir, timing_dir, expected, reps, rep_unit):
        self.bench = bench; self.items = [json.loads(x) for x in (Path(sets_dir) / f"{bench}.jsonl").read_text().splitlines() if x.strip()]
        self.qids = [it["qid"] for it in self.items]; self.boot = SC.Boot([it["image_id"] for it in self.items], reps)
        self.S, self.K, self.N, self.F, self.X, self.P, self._bm, self.integrity = {}, {}, {}, {}, {}, {}, {}, {}
        for cfg in expected:
            recs, self.integrity[cfg] = SC.load_config(Path(pred_dir) / bench, cfg); preds = {q: r["pred"] for q, r in recs.items()}
            if set(preds) != set(self.qids):
                raise SystemExit(f"coverage guard: {bench} {cfg}: {len(set(preds) & set(self.qids))} / {len(self.qids)} items, "
                                 f"{len(set(preds) - set(self.qids))} unexpected; integrity {self.integrity[cfg]}")
            sc = SC.item_scores(bench, self.items, preds)
            self.S[cfg] = np.array([sc[q] for q in self.qids]); self.K[cfg] = np.array([recs[q]["kept"] for q in self.qids], float)
            self.N[cfg] = np.array([recs[q]["n_vis"] for q in self.qids], float); self.F[cfg] = np.array([recs[q]["f"] for q in self.qids], float)
            self.X[cfg] = int(sum(bool(recs[q].get("hit_max")) for q in self.qids)); self.P[cfg] = [preds[q] for q in self.qids]
        self.T, self.Tci, self.timing_meta = {}, {}, {}
        p = next((q for q in (Path(timing_dir) / bench / f"timing_{bench}.json", Path(timing_dir) / f"timing_{bench}.json") if q.exists()), None) if timing_dir else None
        if p is not None:
            j = json.loads(p.read_text()); self.timing_meta = {k: j.get(k) for k in ("gpu", "host", "b_star", "primary_batch", "pools", "allow_any_gpu")}
            for r in j["rows"]:
                self.T[(r["config"], r["B"])] = r["per_item_ms_median"]; self.Tci[(r["config"], r["B"])] = r.get("per_item_ms_median_ci95")
        self.rep = rep_unit

    def kept_fraction(self, cfg):
        return float(np.mean(self.K[cfg] / self.N[cfg]))


def families(main, present):
    F = {PL.NAMES[o]: [[f"sel:{u}:{f}" for f in PL.FKEYS] for u in main[o]] for o in PL.OBJECTIVES}
    F.update({f"{PL.NAMES[o]}_dyn": [[f"dyn:{u}:{op}" for op in PL.OPS] for u in main[o]] for o in PL.OBJECTIVES})
    F.update({m: [[f"static:{m}:{f}" for f in PL.FKEYS]] for m in ("grid", "random", "qsim")})
    return {k: v for k, v in F.items() if all(c in present for seed in v for c in seed)}


def fixed_contrast(bn, a_cfgs, b_cfgs):
    """Fixed-fraction contrast: seed-mean per-item scores of family a minus family b (seed i vs seed i, or vs the single seedless row)."""
    da = np.mean([bn.S[c] for c in a_cfgs], 0); db = np.mean([bn.S[c] for c in b_cfgs], 0); d = da - db
    seed_d = [bn.q(x) - bn.q(y) for x, y in zip(a_cfgs, b_cfgs)] if len(b_cfgs) == len(a_cfgs) else [bn.q(x) - bn.q(b_cfgs[0]) for x in a_cfgs]
    out = SC.contrast(d.mean(), bn.boot.means(d), seed_d)
    out["per_seed"] = SC.per_seed([bn.q(c) for c in a_cfgs], [bn.bm(c) for c in a_cfgs], [bn.q(c) for c in b_cfgs], [bn.bm(c) for c in b_cfgs])
    return out


def latency_contrast(bn, Fm, other, Lms, B):
    s1 = None if Lms is None else SC.seed_mean_at(bn, Fm["S1"], Lms, B); o = None if Lms is None else SC.seed_mean_at(bn, Fm[other], Lms, B)
    if s1 is None or o is None:
        return {"not_reached": True}
    seed_d = [a - b for a, b in zip(s1[2], o[2])] if len(o[2]) == len(s1[2]) else [a - o[2][0] for a in s1[2]]
    out = SC.contrast(s1[0] - o[0], s1[1] - o[1], seed_d); out["per_seed"] = SC.per_seed(s1[2], s1[3], o[2], o[3])
    return out


def anchor_block(bn: Bench40, anchor: str):
    """Decision 15: the batch-1 official full-row anchor next to the batched full row (secondary diagnostic, never in the primary family)."""
    d = bn.S["full"] - bn.S[anchor]; same = np.array([x == y for x, y in zip(bn.P["full"], bn.P[anchor])])
    return {"row": anchor, "role": "secondary diagnostic (coordinator decision 15): the full row through the official HF generate at batch 1",
            "quality": 100 * bn.q(anchor), "ci95": [100 * x for x in SC.ci(bn.bm(anchor))], "full_batched_quality": 100 * bn.q("full"),
            "full_batched_minus_official": dict(SC.contrast(d.mean(), bn.boot.means(d)), role="secondary"),
            "identical_answer_fraction": float(same.mean()), "items_with_different_score": int(np.sum(d != 0)),
            "mean_visual_tokens": float(bn.K[anchor].mean()), "answers_cut_at_max_new_tokens": bn.X[anchor]}


def score_bench(bn: Bench40, main, b_star, primary_basis, anchor=None):
    Fm = families(main, set(bn.S)); R = {"n_items": len(bn.items), "n_groups": bn.boot.G, "families": sorted(Fm), "timing_meta": bn.timing_meta}
    tot = {k: sum(v[k] for v in bn.integrity.values()) for k in ("dropped_lines", "blank_lines", "duplicate_qids")}
    R["integrity"] = {"totals": tot, "configs_with_issues": {c: v for c, v in bn.integrity.items() if v["dropped_lines"] or v["blank_lines"] or v["duplicate_qids"]},
                      "files_per_config_max": max(len(v["files"]) for v in bn.integrity.values()),
                      "rule": "first complete record per qid wins; unparseable / incomplete lines dropped and counted"}
    full_q = bn.q("full")
    R["fixed_fraction"] = {c: {"quality": 100 * bn.q(c), "ci95": [100 * x for x in SC.ci(bn.bm(c))], "retention_rel": bn.q(c) / full_q if full_q else None,
                               "retention_abs_drop": 100 * (full_q - bn.q(c)), "mean_visual_tokens": float(bn.K[c].mean()),
                               "mean_kept_fraction": bn.kept_fraction(c), "answers_cut_at_max_new_tokens": bn.X[c]} for c in bn.S}
    R["visual_tokens_full"] = {"mean": float(bn.N["full"].mean()), "median": float(np.median(bn.N["full"])), "min": int(bn.N["full"].min()), "max": int(bn.N["full"].max())}
    Bs = sorted({B for (_, B) in bn.T}); pB = int(b_star) if b_star else 4
    R["latency_points_ms"] = {B: {L: bn.lat(f"static:grid:{f}", B) for L, f in LPOINTS.items()} for B in Bs}
    R["latency_points_ci95_ms"] = {B: {L: bn.Tci.get((f"static:grid:{f}", B)) for L, f in LPOINTS.items()} for B in Bs}
    R["row_latency_ms"] = {f"{c}@B{B}": {"median": v, "ci95": bn.Tci.get((c, B))} for (c, B), v in sorted(bn.T.items())}
    R["at_latency"] = {}
    for B in Bs:
        R["at_latency"][B] = {}
        for name, seeds in Fm.items():
            R["at_latency"][B][name] = {}
            for L, f in LPOINTS.items():
                Lms = bn.lat(f"static:grid:{f}", B); r = None if Lms is None else SC.seed_mean_at(bn, seeds, Lms, B)
                R["at_latency"][B][name][L] = None if r is None else {"mean": 100 * r[0], "ci95": [100 * x for x in SC.ci(r[1])], "seed_sd": 100 * float(np.std(r[2]))}
    C = {}
    for L, f in LPOINTS.items():                                                       # quality at fixed measured latency (primary iff B*)
        Lms = bn.lat(f"static:grid:{f}", pB) if Bs else None
        for other in ("S2", "S3", "grid", "random", "qsim"):
            if other not in Fm or "S1" not in Fm or not Bs:
                continue
            C[f"S1-{other}@{L}"] = dict(latency_contrast(bn, Fm, other, Lms, pB), batch=pB,
                                        role="primary" if (primary_basis == "latency" and other in PRIMARY_OTHERS) else "secondary")
    for fk in PL.FKEYS:                                                                 # quality at fixed kept-token fraction (primary iff no B*)
        pt = f"f{fk}"
        for other in ("S2", "S3", "grid", "random", "qsim"):
            if other not in Fm or "S1" not in Fm:
                continue
            a = [f"sel:{u}:{fk}" for u in main["vcs"]]
            b = [f"sel:{u}:{fk}" for u in main[{"S2": "logistic", "S3": "direct"}[other]]] if other in ("S2", "S3") else [f"static:{other}:{fk}"]
            C[f"S1-{other}@{pt}"] = dict(fixed_contrast(bn, a, b), role="primary" if (primary_basis == "fraction" and other in PRIMARY_OTHERS and pt in FPOINTS) else "secondary")
    for o in PL.OBJECTIVES:                                                             # A3: dynamic − fixed at the realised mean kept fraction
        if f"{PL.NAMES[o]}_dyn" not in Fm:
            continue
        for op in PL.OPS:
            pt, bs = [], []
            for u in main[o]:
                dc = f"dyn:{u}:{op}"; m = bn.kept_fraction(dc); cfgs = [f"sel:{u}:{fk}" for fk in PL.FKEYS] + ["full"]
                xs = [bn.kept_fraction(c) for c in cfgs]; br = SC.bracket(xs, m)
                if br is None:
                    pt = None; break
                i, j, w = br; a_, b_ = cfgs[i], cfgs[j]
                pt.append(bn.q(dc) - ((1 - w) * bn.q(a_) + w * bn.q(b_))); bs.append(bn.bm(dc) - ((1 - w) * bn.bm(a_) + w * bn.bm(b_)))
            C[f"A3:{PL.NAMES[o]}_dyn-fixed@op{op}"] = {"not_reached": True, "role": "secondary"} if pt is None else dict(SC.contrast(np.mean(pt), np.mean(bs, 0), pt), role="secondary")
    R["contrasts"] = C
    if Bs and pB in Bs:
        target = full_q - 0.01; R[f"within_1pt_of_full_ms_B{pB}"] = {}
        for name, seeds in Fm.items():
            vals = []
            for c in seeds:
                cv = bn.curve(c, pB); vals.append(None if cv is None else SC.first_within([p[0] for p in cv], [bn.q(p[1]) for p in cv], target))
            R[f"within_1pt_of_full_ms_B{pB}"][name] = vals
    full = bn.S["full"]; wrong = full < 0.5; right = ~wrong
    R["preserve_correct"] = {"threshold": "score < 0.5 = wrong (TextVQA soft accuracy; 0 / 1 elsewhere)", "full_wrong": int(wrong.sum())}
    for c, x in bn.S.items():
        if c != "full":
            R["preserve_correct"][c] = {"kept_error": float(np.mean(x[wrong] < 0.5)) if wrong.any() else None,
                                        "corrected": float(np.mean(x[wrong] >= 0.5)) if wrong.any() else None,
                                        "broke_correct": float(np.mean(x[right] < 0.5)) if right.any() else None}
    R["dynamic_fraction_distribution"] = {c: {PL.fkey(f): float(np.mean(bn.F[c] == f)) for f in PL.DYN_GRID} for c in bn.S if c.startswith("dyn:")}
    subsets = {}; key = "structural" if bn.bench.startswith("gqa") else ("type" if bn.bench.startswith("chartqa") else None)
    if key:
        lab = np.array([str(it.get(key)) for it in bn.items])
        if bn.bench.startswith("chartqa"):
            lab = np.array(["human" if "human" in v.lower() else ("augmented" if "augmented" in v.lower() else "other:" + v) for v in lab])
        for v in sorted(set(lab)):
            m = lab == v; subsets[v] = {"n": int(m.sum()), "quality": {c: 100 * float(x[m].mean()) for c, x in bn.S.items()}}
    R["subsets"] = subsets
    if anchor:
        R["official_anchor"] = anchor_block(bn, anchor)
    return R


def llava_fixed_delta(d130_bench, other, k):
    """D1-30 (LLaVA) fixed-budget seed-mean difference S1 − other at k tokens from its fixed_budget table (descriptive; no CI)."""
    fb = d130_bench.get("fixed_budget", {})
    def rows(prefix):
        return [v["quality"] for c, v in fb.items() if c.startswith(prefix) and c.endswith(f":{k}") and not c.split(":")[1].endswith(("_noq", "_dtarget"))]
    s1 = rows("sel:vcs_")
    o = rows({"S2": "sel:logistic_", "S3": "sel:direct_"}[other]) if other in ("S2", "S3") else rows(f"static:{other}:")
    return (float(np.mean(s1)) - float(np.mean(o))) if s1 and o else None


def cross_host(res, d130_path):
    """Descriptive side-by-side of the D1-30 (LLaVA) and R7-40c (Qwen) contrasts S1 − S2, S1 − S3, S1 − grid: sign and size only.  Latency
    points: D1-30's contrast at the same point (B* = 4 on LLaVA); fraction points: D1-30's fixed-budget seed-mean difference at k = f · 576."""
    d = json.loads(Path(d130_path).read_text())["benches"]; out = {"note": "descriptive only: no pooling across hosts, no cross-host test", "rows": []}
    for b, R in res["benches"].items():
        for key, v in R["contrasts"].items():
            if v.get("role") != "primary":
                continue
            other, pt = key.split("@")[0].split("-")[1], key.split("@")[1]
            if pt.startswith("L"):
                lv = d.get(b, {}).get("contrasts", {}).get(f"S1-{other}@{pt}")
                ld, lci, src = (None if not lv or lv.get("not_reached") else lv.get("delta")), (None if not lv else lv.get("ci95")), f"D1-30 contrast S1-{other}@{pt} (role {None if not lv else lv.get('role')})"
            else:
                k = int(round(float(pt[1:]) * 576)); ld, lci, src = llava_fixed_delta(d.get(b, {}), other, k), None, f"D1-30 fixed-budget seed means at {k} tokens"
            out["rows"].append({"bench": b, "contrast": key, "qwen_delta": v.get("delta"), "qwen_ci95": v.get("ci95"), "llava_delta": ld, "llava_ci95": lci, "llava_source": src})
    return out


def markdown(res):
    L = ["# R7-40c readout on Qwen2.5-VL-7B (generated by scripts/r7_d1_40c_score.py)", "",
         f"Primary basis: {res['primary_basis']} (B* = {res['b_star']}); Holm per point over the primary family (S1 − S2, S1 − S3, S1 − grid × benchmarks).", ""]
    for b, R in res["benches"].items():
        L += [f"## {b}: {R['n_items']} items, {R['n_groups']} groups; full = {R['fixed_fraction']['full']['quality']:.2f}", "",
              "| contrast | Δ (points) | 95 % CI | p | p (Holm) |", "|---|---|---|---|---|"]
        for k, v in R["contrasts"].items():
            if v.get("role") == "primary":
                L.append(f"| {k} | " + ("not reached | | |" if v.get("not_reached") else f"{v['delta']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | {v['p']:.4f} | {v.get('p_holm', float('nan')):.4f} |"))
        A = R.get("official_anchor")
        if A:
            L += [f"Official batch-1 full row (secondary diagnostic): {A['quality']:.2f} vs batched full {A['full_batched_quality']:.2f}; "
                  f"identical answers {100 * A['identical_answer_fraction']:.1f} %; full − official {A['full_batched_minus_official']['delta']:+.2f} points.", ""]
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", default=str(PL.D30_SETS)); ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--timing-dir", default=None); ap.add_argument("--configs-dir", required=True); ap.add_argument("--lr-choice", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--benches", nargs="+", default=list(PL.BENCHES)); ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--protocol", default=str(PROTOCOL)); ap.add_argument("--timing-report", default=str(PL.TIMING_REPORT))
    ap.add_argument("--b-star", default=None, help="smoke only: override the R7-40a report (int or 'none')"); ap.add_argument("--d1-30-results", default=None)
    ap.add_argument("--anchor", default="full_official", help="the batch-1 official full-row anchor (decision 15); 'none' only for smokes / tests without it")
    a = ap.parse_args(argv)
    require_freeze(a.protocol)
    import r7_d1_40c_timing as TMG
    if a.b_star is not None:
        b_star, brec = (None if a.b_star == "none" else int(a.b_star)), {"override": a.b_star, "smoke": True}
    else:
        b_star, brec = TMG.read_b_star(Path(a.timing_report))
    basis = "latency" if b_star else "fraction"
    main_u = PL.unit_names(json.loads(Path(a.lr_choice).read_text())); rep = main_u["vcs"][0]
    res = {"host": "Qwen2.5-VL-7B-Instruct", "representative_unit": rep, "b_star": b_star, "b_star_source": brec, "primary_basis": basis,
           "primary_points": list(LPOINTS) if basis == "latency" else list(FPOINTS), "benches": {}, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "notes": ["primary family: S1 − S2, S1 − S3, S1 − uniform grid × benchmarks; Holm per point (R7-40 draft for_40c)",
                     "latency medians carry bootstrap CIs over timing batches; not propagated into the quality-at-latency interpolation",
                     "claims limited to the LLM side; the vision encoder always runs in full (vision share printed in every timing row)"]}
    for bench in a.benches:
        exp = PL.read_configs(Path(a.configs_dir) / f"configs_{bench}.txt"); anchor = None if a.anchor == "none" else a.anchor
        bn = Bench40(bench, a.sets_dir, a.pred_dir, a.timing_dir, exp + ([anchor] if anchor else []), a.reps, rep)   # coverage guard covers the anchor
        if basis == "latency" and not bn.T:
            raise SystemExit(f"{bench}: Qwen has a B* but no timing rows were found in {a.timing_dir}: the primary endpoint cannot be computed")
        if bn.T and bn.timing_meta.get("b_star") != b_star and a.b_star is None:
            raise SystemExit(f"{bench}: timing file B* {bn.timing_meta.get('b_star')} != reference report B* {b_star}")
        res["benches"][bench] = score_bench(bn, main_u, b_star, basis, anchor)
    pts = LPOINTS if basis == "latency" else FPOINTS
    for P in pts:                                                                      # Holm over the primary family at each point
        keys = [(b, k) for b in res["benches"] for k, v in res["benches"][b]["contrasts"].items() if k.endswith(f"@{P}") and v.get("role") == "primary" and "p" in v]
        if keys:
            for (b, k), pv in zip(keys, SC.holm([res["benches"][b]["contrasts"][k]["p"] for b, k in keys])):
                res["benches"][b]["contrasts"][k]["p_holm"] = pv; res["benches"][b]["contrasts"][k]["holm_family_size"] = len(keys)
    if a.d1_30_results:
        res["cross_host_descriptive"] = cross_host(res, a.d1_30_results)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "r7_40c_results.json").exists():
        raise SystemExit(f"{out / 'r7_40c_results.json'} exists: a re-verification writes to a new --out directory")
    (out / "r7_40c_results.json").write_text(V.jdump(res, indent=1)); (out / "r7_40c_results.md").write_text(markdown(res))
    print(f"[r7_d1_40c_score] {len(res['benches'])} benchmarks scored -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
