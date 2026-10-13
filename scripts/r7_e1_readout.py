"""r7 E1 readout (CPU; SparseVLM env for the m4c TextVQA evaluator, as R7-40c's scorer) — the analysis of the frozen R7-E1 design
(reports/R7/R7_E1_EXTERNAL_PROTOCOL_DRAFT_20261013.yaml 'analysis', reports/R7/R7_E1_STAGED_FREEZE_20261013.yaml D-E1-1..11).
Refuses to run unless the key 'r7_e1_staged_freeze' is in reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml and eval_e1/completeness_e1.json says
complete; never overwrites a readout.  The frozen R7-40c / D1-30 machinery is reused unchanged: r7_d1_40c_score.Bench40 (coverage guard: every
configured row covers exactly the benchmark's items; first complete record per qid; integrity counts), r7_d1_30_score.item_scores through it (GQA exact
match after LLaVA's convert_gqa_for_eval normalisation; TextVQA m4c soft accuracy; ChartQA relaxed accuracy pinned to pix2struct 6fe25c1) and
r7_d1_30_score.Boot (image- / chart-grouped bootstrap, 2,000 replicates, seed 0; the E1 and R7-40c benches share the same replicate weights).
Reported (descriptive only):
  - per E1 row ext:<method>:<f>: quality with its 95 % CI, realised mean kept fraction, mean visual tokens kept and N, answers cut at
    max_new_tokens, retention vs the method's in-env anchor ext:<method>:1 and vs R7-40c's full_official (relative and absolute, point estimates as
    R7-40c's fixed-fraction table), kept == the method's own rule, median per-item 'ms' (L40S, batch 1: descriptive, not a latency result, D-E1-6),
    integrity, GQA structural types / ChartQA human-augmented subsets;
  - the env diagnostic ext:<method>:1 vs R7-40c full_official (same items, same batch-1 call, other env): both qualities with CIs, the paired
    difference with its bootstrap CI (no p-value: a diagnostic, never an effect), identical-answer fraction, items whose score differs;
  - the host-column table: at each nominal fraction R7-40c's fixed-fraction rows (uniform grid; S1 / S2 / S3 as the mean of the three seed rows
    with the CI of that item-level mean and the per-seed values) next to the E1 columns; at f = 1 R7-40c's batched full row and full_official next to
    the E1 anchors.  No contrast, difference or test between an E1 column and S1 / S2 / S3 / grid is computed (owner I1, D-E1-7).
    python -I scripts/r7_e1_readout.py --out O [--sets-dir S] [--pred-dir P] [--configs-dir C] [--r40c-pred-dir P40] [--r40c-configs-dir C40]
                                       [--lr-choice L] [--completeness F] [--protocol F] [--methods mmtok hiprune] [--r7-40c-results J]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402  (frozen D1-30: prediction-file layout)
import r7_d1_30_score as SC  # noqa: E402  (frozen D1-30 readout: loader, evaluators, Boot, ci)
import r7_d1_40c_plan as PL  # noqa: E402  (frozen R7-40c: benchmarks, fraction keys, unit names)
import r7_d1_40c_score as SCO  # noqa: E402  (frozen R7-40c readout: Bench40)
import r7_e1_external as EX  # noqa: E402  (frozen E1 glue: configuration names, methods, k rules, markers)
import r7_vl_common as V  # noqa: E402

FREEZE_KEY = "r7_e1_staged_freeze"
PROTOCOL = Path(__file__).resolve().parents[1] / "reports" / "R7" / "R7_D1_PROTOCOL_FROZEN_20261012.yaml"
OFFICIAL = "full_official"                                   # R7-40c's batch-1 official full row (decision 15 there)
FAMILIES = {"grid": None, "S1": "vcs", "S2": "logistic", "S3": "direct"}
NOTE = ("E1 columns: an external method's official code in its own pinned env at batch 1 (MMTok @8e26868, HiPrune @8278100) on R7-40c's items, "
        "prompts and evaluators; a separate host column, descriptive only — no contrast or test with S1 / S2 / S3 / grid is computed (owner I1, D-E1-7)")


def require_gates(protocol, completeness):
    p = Path(protocol)
    if not p.exists() or FREEZE_KEY not in p.read_text():
        raise SystemExit(f"R7-E1 is not frozen ({FREEZE_KEY} absent in {p}): the readout refuses to run")
    c = Path(completeness)
    if not c.exists():
        raise SystemExit(f"{c} is missing: run the stage-2 completeness check first (the readout refuses to run)")
    j = json.loads(c.read_text())
    if j.get("complete") is not True:
        raise SystemExit(f"{c}: not complete ({j.get('problems')}): the readout refuses to run")
    return j


def e1_configs(configs_dir, bench, methods):
    """The frozen configuration list (eval_e1/configs/configs_e1_<bench>.txt) restricted to the methods read out; D-E1-1: every method with its anchor."""
    cfgs = PL.read_configs(Path(configs_dir) / f"configs_e1_{bench}.txt"); [EX.parse_cfg(c) for c in cfgs]
    sel = [c for c in cfgs if EX.parse_cfg(c)[0] in methods]
    want = EX.config_list(tuple(methods), anchor=True)
    if sorted(sel) != sorted(want):
        raise SystemExit(f"{bench}: E1 configuration list {sel} != the frozen rows {want} (D-E1-1: four fractions + the anchor per method)")
    return want


def r40c_configs(configs_dir, bench, main):
    """R7-40c's rows at the E1 fractions: full, the uniform grid and S1 / S2 / S3 x 3 seeds at 0.125 / 0.25 / 0.5 / 0.75, + full_official."""
    have = set(PL.read_configs(Path(configs_dir) / f"configs_{bench}.txt"))
    need = ["full"] + [f"static:grid:{f}" for f in PL.FKEYS] + [f"sel:{u}:{f}" for o in PL.OBJECTIVES for u in main[o] for f in PL.FKEYS]
    missing = [c for c in need if c not in have]
    if missing:
        raise SystemExit(f"{bench}: R7-40c rows {missing[:5]} not in its configuration list")
    return need + [OFFICIAL]


def build_identities(pred_dir, bench, cfg):
    """Build identities recorded by every shard session of an E1 configuration (eval_e1/pred/<bench>/_e1_meta): exactly one is allowed."""
    ids, gpus = [], set()
    for p in sorted((Path(pred_dir) / bench / "_e1_meta").glob(f"{RUN.cfg_file(cfg)}@*.json")):
        rec = json.loads(p.read_text()); ids.append(json.dumps(rec.get("identity"), sort_keys=True))
        gpus |= {s.get("gpu") for s in rec.get("sessions", []) if s.get("gpu")}
    return sorted(set(ids)), sorted(gpus), len(ids)


def cell(bn, cfgs):
    """Quality of one row, or of the item-level mean of a family's seed rows, with the CI of that mean (shared replicates); descriptive."""
    x = np.mean([bn.S[c] for c in cfgs], 0)
    out = {"rows": list(cfgs), "quality": 100 * float(x.mean()), "ci95": [100 * v for v in SC.ci(bn.boot.means(x))],
           "mean_kept_fraction": float(np.mean([bn.kept_fraction(c) for c in cfgs])), "mean_visual_tokens": float(np.mean([bn.K[c].mean() for c in cfgs]))}
    if len(cfgs) > 1:
        q = [100 * bn.q(c) for c in cfgs]; out.update(per_seed=q, seed_sd=float(np.std(q)))
    return out


def subsets(bn, cfgs):
    key = "structural" if bn.bench.startswith("gqa") else ("type" if bn.bench.startswith("chartqa") else None)
    if not key:
        return {}
    lab = np.array([str(it.get(key)) for it in bn.items])
    if bn.bench.startswith("chartqa"):
        lab = np.array(["human" if "human" in v.lower() else ("augmented" if "augmented" in v.lower() else "other:" + v) for v in lab])
    return {v: {"n": int((lab == v).sum()), "quality": {c: 100 * float(bn.S[c][lab == v].mean()) for c in cfgs}} for v in sorted(set(lab))}


def e1_row(bnE, bnR, cfg, recs):
    method, f = EX.parse_cfg(cfg); q = bnE.q(cfg); anchor = EX.cfg_name(method, EX.ANCHOR); qa = bnE.q(anchor); qo = bnR.q(OFFICIAL)
    rows = [recs[q_] for q_ in bnE.qids]
    return {"method": method, "fraction": f, "quality": 100 * q, "ci95": [100 * v for v in SC.ci(bnE.bm(cfg))],
            "mean_kept_fraction": bnE.kept_fraction(cfg), "mean_visual_tokens": float(bnE.K[cfg].mean()), "mean_n_vis": float(bnE.N[cfg].mean()),
            "answers_cut_at_max_new_tokens": bnE.X[cfg],
            "retention_vs_in_env_anchor": None if cfg == anchor else {"anchor": anchor, "rel": q / qa if qa else None, "abs_drop": 100 * (qa - q)},
            "retention_vs_r7_40c_full_official": {"rel": q / qo if qo else None, "abs_drop": 100 * (qo - q)},
            "kept_equals_method_rule": int(sum(r.get("kept") == EX.k_rule(method, f, r.get("n_vis", -1)) for r in rows)), "items": len(rows),
            "ms_per_item_median_descriptive": float(np.median([r["ms"] for r in rows if r.get("ms") is not None])) if any(r.get("ms") is not None for r in rows) else None,
            "integrity": bnE.integrity[cfg]}


def env_diagnostic(bnE, bnR, method):
    """ext:<method>:1 vs R7-40c full_official: a diagnostic of env / backend effects (never an effect; no p-value)."""
    a = EX.cfg_name(method, EX.ANCHOR); d = bnE.S[a] - bnR.S[OFFICIAL]
    same = np.array([x == y for x, y in zip(bnE.P[a], bnR.P[OFFICIAL])])
    return {"row": a, "reference": f"{OFFICIAL} (R7-40c, batch-1 official HF generate, vl_baselines env)",
            "role": "diagnostic of env / backend effects (transformers, attention backend, torch); never an effect, no test",
            "quality": 100 * bnE.q(a), "ci95": [100 * v for v in SC.ci(bnE.bm(a))], "reference_quality": 100 * bnR.q(OFFICIAL),
            "reference_ci95": [100 * v for v in SC.ci(bnR.bm(OFFICIAL))],
            "anchor_minus_official": {"delta": 100 * float(d.mean()), "ci95": [100 * v for v in SC.ci(bnE.boot.means(d))]},
            "identical_answer_fraction": float(same.mean()), "items_with_different_score": int(np.sum(d != 0)),
            "mean_visual_tokens": {"anchor": float(bnE.K[a].mean()), "official": float(bnR.K[OFFICIAL].mean())}}


def host_table(bnE, bnR, main, methods):
    T = {}
    for fk in PL.FKEYS:
        T[fk] = {"r7_40c": {fam: cell(bnR, [f"static:grid:{fk}"] if o is None else [f"sel:{u}:{fk}" for u in main[o]]) for fam, o in FAMILIES.items()},
                 "e1": {m: cell(bnE, [EX.cfg_name(m, float(fk))]) for m in methods}}
    T["1"] = {"r7_40c": {"full (batched, R7-40c)": cell(bnR, ["full"]), f"{OFFICIAL} (batch 1)": cell(bnR, [OFFICIAL])},
              "e1": {m: cell(bnE, [EX.cfg_name(m, EX.ANCHOR)]) for m in methods}}
    return T


def score_bench(bench, a, main, methods):
    cfgE = e1_configs(a.configs_dir, bench, methods); cfgR = r40c_configs(a.r40c_configs_dir, bench, main)
    ident = {}
    for c in cfgE:
        ids, gpus, files = build_identities(a.pred_dir, bench, c)
        if len(ids) != 1:
            raise SystemExit(f"{bench} {c}: {len(ids)} build identities over {files} shard records (one build per configuration required)")
        ident[c] = {"shard_records": files, "gpus": gpus, "identity": json.loads(ids[0])}
    bnE = SCO.Bench40(bench, a.sets_dir, a.pred_dir, None, cfgE, a.reps, None)            # coverage guard on every E1 row
    bnR = SCO.Bench40(bench, a.sets_dir, a.r40c_pred_dir, None, cfgR, a.reps, None)       # and on the R7-40c rows used
    if bnE.qids != bnR.qids or not np.array_equal(bnE.boot.W, bnR.boot.W):
        raise SystemExit(f"{bench}: the E1 and R7-40c benches do not share items / bootstrap replicates")
    recs = {c: SC.load_config(Path(a.pred_dir) / bench, c)[0] for c in cfgE}
    rows = {c: e1_row(bnE, bnR, c, recs[c]) for c in cfgE}
    R = {"n_items": len(bnE.items), "n_groups": bnE.boot.G, "note": NOTE, "e1_rows": rows,
         "env_diagnostic": {m: env_diagnostic(bnE, bnR, m) for m in methods},
         "host_column_table": host_table(bnE, bnR, main, methods),
         "subsets": {"e1": subsets(bnE, cfgE), "r7_40c_full_official": subsets(bnR, [OFFICIAL])},
         "builds": ident,
         "integrity": {"totals": {k: sum(v[k] for v in bnE.integrity.values()) for k in ("dropped_lines", "blank_lines", "duplicate_qids")},
                       "files_per_config_max": max(len(v["files"]) for v in bnE.integrity.values()),
                       "kept_not_equal_rule": {c: r["items"] - r["kept_equals_method_rule"] for c, r in rows.items() if r["items"] != r["kept_equals_method_rule"]},
                       "rule": "first complete record per qid wins; unparseable / incomplete lines dropped and counted (R7-40c loaders)"}}
    return R, bnR


def check_r40c_results(res, path):
    """Optional cross-check: every R7-40c row recomputed here equals R7-40c's own readout (fixed_fraction quality, official anchor)."""
    ref = json.loads(Path(path).read_text())["benches"]; worst = 0.0
    for b, R in res["benches"].items():
        ff = ref[b]["fixed_fraction"]
        for fk, row in R["host_column_table"].items():
            for name, cl in row["r7_40c"].items():
                for c in cl["rows"]:
                    q = (ref[b].get("official_anchor") or {}).get("quality") if c == OFFICIAL else ff[c]["quality"]
                    mine = cl["per_seed"][cl["rows"].index(c)] if "per_seed" in cl else cl["quality"]
                    worst = max(worst, abs(q - mine))
    if worst > 1e-9:
        raise SystemExit(f"R7-40c rows recomputed here differ from {path} by up to {worst} points: stop")
    return {"file": str(path), "max_abs_diff_points": worst}


def fmt(c):
    return f"{c['quality']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"


def markdown(res):
    L = ["# R7-E1 readout: external Qwen2.5-VL baselines as a separate host column (generated by scripts/r7_e1_readout.py)", "", NOTE, "",
         "Quality in %, 95 % bootstrap CI (image / chart groups, 2,000 replicates, seed 0, shared with R7-40c's rows). S1 / S2 / S3 = mean of the "
         "three seed rows. Fractions are nominal: each method keeps its own k (MMTok floor, HiPrune round half to even, R7-40c round half up); "
         "realised kept fractions are listed per E1 row.", ""]
    for b, R in res["benches"].items():
        ms = [m for m in res["methods"]]
        L += [f"## {b}: {R['n_items']} items, {R['n_groups']} groups", "",
              "| f | grid | S1 | S2 | S3 | " + " | ".join(f"{m} (E1)" for m in ms) + " |", "|" + "---|" * (5 + len(ms))]
        for fk in PL.FKEYS:
            row = R["host_column_table"][fk]
            L.append(f"| {fk} | " + " | ".join(fmt(row["r7_40c"][fam]) for fam in FAMILIES) + " | " + " | ".join(fmt(row["e1"][m]) for m in ms) + " |")
        row = R["host_column_table"]["1"]
        L.append("| 1 | full (batched) " + fmt(row["r7_40c"]["full (batched, R7-40c)"]) + f" | {OFFICIAL} " + fmt(row["r7_40c"][f"{OFFICIAL} (batch 1)"])
                 + " | | | " + " | ".join("anchor " + fmt(row["e1"][m]) for m in ms) + " |")
        L += ["", "| E1 row | quality | kept fraction | visual tokens | cut at max_new | retention vs anchor (rel / abs) | vs full_official (rel / abs) | ms median (L40S, descriptive) |",
              "|---|---|---|---|---|---|---|---|"]
        for c, r in R["e1_rows"].items():
            ra = r["retention_vs_in_env_anchor"]; ro = r["retention_vs_r7_40c_full_official"]
            L.append(f"| {c} | {fmt(r)} | {r['mean_kept_fraction']:.4f} | {r['mean_visual_tokens']:.1f} | {r['answers_cut_at_max_new_tokens']} | "
                     + ("anchor" if ra is None else f"{ra['rel']:.4f} / {ra['abs_drop']:+.2f}") + f" | {ro['rel']:.4f} / {ro['abs_drop']:+.2f} | "
                     + ("n/a" if r["ms_per_item_median_descriptive"] is None else f"{r['ms_per_item_median_descriptive']:.0f}") + " |")
        L.append("")
        for m, d in R["env_diagnostic"].items():
            L.append(f"Env diagnostic (not an effect): {d['row']} {d['quality']:.2f} vs {OFFICIAL} {d['reference_quality']:.2f}; anchor − official "
                     f"{d['anchor_minus_official']['delta']:+.2f} [{d['anchor_minus_official']['ci95'][0]:+.2f}, {d['anchor_minus_official']['ci95'][1]:+.2f}] points; "
                     f"identical answers {100 * d['identical_answer_fraction']:.1f} %.")
        L.append("")
    if res.get("gaps"):
        L += [f"Not read out (disclosed gap): {', '.join(res['gaps'])}.", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets-dir", default=str(PL.D30_SETS)); ap.add_argument("--pred-dir", default=str(EX.E1 / "pred"))
    ap.add_argument("--configs-dir", default=str(EX.E1 / "configs")); ap.add_argument("--r40c-pred-dir", default=str(PL.R40C / "pred"))
    ap.add_argument("--r40c-configs-dir", default=str(PL.R40C / "configs")); ap.add_argument("--lr-choice", default=str(PL.R40 / "selectors" / "lr_choice.json"))
    ap.add_argument("--completeness", default=str(EX.E1 / "completeness_e1.json")); ap.add_argument("--protocol", default=str(PROTOCOL))
    ap.add_argument("--methods", nargs="+", default=list(EX.METHODS), choices=list(EX.METHODS)); ap.add_argument("--benches", nargs="+", default=list(PL.BENCHES))
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--r7-40c-results", default=None, help="optional cross-check against R7-40c's own readout")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gate = require_gates(a.protocol, a.completeness)
    out = Path(a.out)
    if (out / "r7_e1_results.json").exists():
        raise SystemExit(f"{out / 'r7_e1_results.json'} exists: a re-verification writes to a new --out directory")
    methods = [m for m in EX.METHODS if m in a.methods]; main_u = PL.unit_names(json.loads(Path(a.lr_choice).read_text()))
    res = {"protocol": "R7-E1 (reports/R7/R7_E1_STAGED_FREEZE_20261013.yaml)", "host": "Qwen2.5-VL-7B-Instruct", "methods": methods,
           "gaps": [m for m in EX.METHODS if m not in methods], "fractions": list(PL.FKEYS) + ["1 (in-env anchor)"],
           "method_commits": {m: EX.METHODS[m]["commit"] for m in methods}, "k_rules": {m: EX.METHODS[m]["k_rule"] for m in methods},
           "completeness": {"file": str(a.completeness), "utc": gate.get("utc")}, "reps": a.reps, "benches": {},
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "notes": [NOTE, "R7-40c rows recomputed from its predictions with the same Bench40 / item_scores / Boot (identical to its readout)",
                     "retention values are point estimates (as R7-40c's fixed-fraction table); the env diagnostic CI is a paired bootstrap without a p-value",
                     "per-item 'ms' is batch-1 wall time on the quality GPUs (L40S, shared nodes): descriptive only, no latency claim (D-E1-6)"]}
    for b in a.benches:
        res["benches"][b], _ = score_bench(b, a, main_u, methods)
    if a.r7_40c_results:
        res["r7_40c_results_check"] = check_r40c_results(res, a.r7_40c_results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "r7_e1_results.json").write_text(V.jdump(res, indent=1)); (out / "r7_e1_results.md").write_text(markdown(res))
    print(f"[r7_e1_readout] {len(res['benches'])} benchmarks, methods {methods} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
