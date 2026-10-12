"""r7 R7-30 D1-30 readout (CPU, SparseVLM env for the m4c evaluator).  Official per-item scores: GQA exact match after LLaVA's
convert_gqa_for_eval normalisation; TextVQA m4c soft accuracy; ChartQA relaxed accuracy (pinned, r7_d1_30_eval).  Coverage guard: every
expected configuration must cover exactly the benchmark's items, else the readout stops.
Resampling: groups = images (GQA, TextVQA) / charts (ChartQA); 2,000 group-bootstrap replicates shared by every statistic of a benchmark
(paired); a configuration's replicate means come from group sums (W @ s / W @ n).  Latencies are the timing runs' medians (fixed, not
resampled).  Curves: each configuration family's points (per-item latency at batch B, quality) including full; quality at a latency is
linear interpolation between the bracketing points (None outside the measured range, reported as 'not reached').
Primary (B* = 4, nodesumo01): quality at L25 / L50 / L75 = uniform-grid 144 / 288 / 432-token latencies, for our pipeline's rows;
contrasts S1 − S2, S1 − S3 (seed mean of the per-seed interpolated qualities; paired bootstrap; per-seed CIs) and S1 − SparseVLM on the
coordinator's basis D30-1: (a) budget-matched — S1 fixed budget k vs the adapted SparseVLM row at k = 144 / 288 / 432 — primary, (b)
latency-matched at B = 1 (both interpolated at uniform-grid's B = 1 latencies) secondary.  No latency claim is made against SparseVLM at
B* = 4: its official code is batch-1 only.  Holm across the 9 primary contrasts (3 × 3 benchmarks) at each latency point.  Latency CIs
(bootstrap over timing batches) are reported with the latency points and are not propagated into the interpolation.
Secondaries: every family at L25 / L50 / L75 for B = 1 / 16 / 32; S1 − grid / random / qsim; A1 − S1 and A2 − S3 at each fixed budget;
A3 dynamic − fixed at the dynamic row's realised mean budget; fixed-budget table with absolute / relative retention and visual tokens;
latency to stay within 1.0 point of full; preserve / correct when the full model is wrong; dynamic budget distribution; subsets (GQA
structural type, ChartQA human / augmented).
    python scripts/r7_d1_30_score.py --sets-dir S --pred-dir P --timing-dir T --lr-choice L --out O [--benches ...] [--configs-file C]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_eval as EV  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_vl_common as V  # noqa: E402

BUDGETS = (72, 144, 288, 432)
LPOINTS = {"L25": 144, "L50": 288, "L75": 432}
PRIMARY_B = 4
NAMES = {"vcs": "S1", "logistic": "S2", "direct": "S3"}


def item_scores(bench, items, preds):
    """Per-item official score (qid -> float) for the items that have a prediction."""
    tv = V.textvqa_scorer() if bench.startswith("textvqa") else None; out = {}
    for it in items:
        p = preds.get(it["qid"])
        if p is None:
            continue
        if bench.startswith("gqa"):
            out[it["qid"]] = V.gqa_correct(p, it["answer"])
        elif bench.startswith("textvqa"):
            out[it["qid"]] = float(tv(p, it["answers"]))
        else:
            out[it["qid"]] = EV.chartqa_score(p, it["answer"])
    return out


class Boot:
    """Shared group-bootstrap replicates: W [reps, groups] multinomial counts; means(x) -> replicate means of an item vector."""

    def __init__(self, groups, reps=2000, seed=0):
        ug, self.inv = np.unique(np.asarray(groups), return_inverse=True); G = len(ug); rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(reps)]).astype(np.float64)
        self.n = self.W @ np.bincount(self.inv, minlength=G).astype(np.float64); self.G = G

    def means(self, x):
        return (self.W @ np.bincount(self.inv, weights=np.asarray(x, float), minlength=self.G)) / self.n


def ci(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_p(bs):
    bs = np.asarray(bs); n = len(bs)
    return float(min(1.0, 2 * min((np.sum(bs <= 0) + 1) / (n + 1), (np.sum(bs >= 0) + 1) / (n + 1))))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i])); adj[i] = run
    return adj.tolist()


def bracket(lat, L):
    """(i, j, w) with L = (1 − w) lat[i] + w lat[j] over the latency-sorted points; None outside the range."""
    o = np.argsort(lat, kind="stable"); x = np.asarray(lat, float)[o]
    if len(x) == 0 or L < x[0] - 1e-9 or L > x[-1] + 1e-9:
        return None
    j = int(np.searchsorted(x, L, side="left"))
    if j == 0 or abs(x[j] - L) < 1e-12:
        return int(o[j]), int(o[j]), 0.0
    i = j - 1; w = (L - x[i]) / (x[j] - x[i]) if x[j] > x[i] else 0.0
    return int(o[i]), int(o[j]), float(w)


def first_within(lat, q, target):
    """Smallest latency on the piecewise-linear curve (points sorted by latency) where quality reaches target; None if never."""
    o = np.argsort(lat, kind="stable"); x, y = np.asarray(lat, float)[o], np.asarray(q, float)[o]
    if y[0] >= target:
        return float(x[0])
    for i in range(1, len(x)):
        if y[i] >= target:
            return float(x[i - 1] + (target - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])) if y[i] != y[i - 1] else float(x[i])
    return None


class Bench:
    def __init__(self, bench, sets_dir, pred_dir, timing_dir, expected, reps, rep_unit):
        self.bench = bench; self.items = [json.loads(x) for x in (Path(sets_dir) / f"{bench}.jsonl").read_text().splitlines() if x.strip()]
        self.qids = [it["qid"] for it in self.items]; self.boot = Boot([it["image_id"] for it in self.items], reps); self.S, self.K, self._bm = {}, {}, {}
        for cfg in expected:
            f = Path(pred_dir) / bench / f"{RUN.cfg_file(cfg)}.jsonl"; rows = V.read_jsonl(f); preds = {r["qid"]: r["pred"] for r in rows}
            if set(preds) != set(self.qids):
                raise SystemExit(f"coverage guard: {bench} {cfg}: {len(set(preds) & set(self.qids))} / {len(self.qids)} items, "
                                 f"{len(set(preds) - set(self.qids))} unexpected")
            sc = item_scores(bench, self.items, preds); kept = {r["qid"]: r.get("kept", RUN.N_VIS) for r in rows}
            self.S[cfg] = np.array([sc[q] for q in self.qids]); self.K[cfg] = np.array([kept[q] for q in self.qids], float)
        self.T, self.Tci = {}, {}
        for name in (f"timing_{bench}.json", f"sparsevlm_timing_{bench}.json"):
            p = next((q for q in (Path(timing_dir) / bench / name, Path(timing_dir) / name) if q.exists()), None)
            if p is not None:
                for r in json.loads(p.read_text())["rows"]:
                    self.T[(r["config"], r["B"])] = r["per_item_ms_median"]; self.Tci[(r["config"], r["B"])] = r.get("per_item_ms_median_ci95")
        self.rep = rep_unit

    def bm(self, cfg):
        if cfg not in self._bm:
            self._bm[cfg] = self.boot.means(self.S[cfg])
        return self._bm[cfg]

    def q(self, cfg):
        return float(self.S[cfg].mean())

    def lat(self, cfg, B):
        key = cfg
        if cfg.startswith("sel:"):
            key = f"sel:{self.rep}:{cfg.split(':')[2]}"
        return self.T.get((key, B))

    def curve(self, cfgs, B):
        pts = [(self.lat(c, B), c) for c in ["full"] + list(cfgs) if c in self.S]
        return None if any(p[0] is None for p in pts) else pts

    def at(self, pts, L, rep=False):
        """Interpolated quality (scalar) or replicate vector at latency L; None if not reached."""
        if pts is None:
            return None
        br = bracket([p[0] for p in pts], L)
        if br is None:
            return None
        i, j, w = br; a, b = pts[i][1], pts[j][1]
        return (1 - w) * (self.bm(a) if rep else self.q(a)) + w * (self.bm(b) if rep else self.q(b))


def families(main, abl):
    F = {NAMES[o]: [[f"sel:{u}:{k}" for k in BUDGETS] for u in main[o]] for o in main}
    F.update({f"{NAMES[o]}_dyn": [[f"dyn:{u}:{f}" for f in RUN.FRACTIONS] for u in main[o]] for o in main})
    F.update({a: [[f"sel:{u}:{k}" for k in BUDGETS] for u in abl[a]] for a in abl})
    F.update({m: [[f"static:{m}:{k}" for k in BUDGETS]] for m in ("grid", "random", "qsim")})
    return F


def seed_mean_at(bn, seeds, L, B):
    """Seed mean of the interpolated quality: (scalar, replicate vector, per-seed values, per-seed replicate vectors); None when any seed
    does not reach L."""
    curves = [bn.curve(c, B) for c in seeds]; pt = [bn.at(cv, L) for cv in curves]
    if any(v is None for v in pt):
        return None
    reps = [bn.at(cv, L, rep=True) for cv in curves]
    return float(np.mean(pt)), np.mean(reps, 0), pt, reps


def per_seed(a_pts, a_reps, b_pts, b_reps):
    """Per-seed paired contrasts (seed i vs seed i, or vs the single row of a seedless method) with their own bootstrap CI."""
    out = []
    for i in range(len(a_pts)):
        j = i if len(b_pts) == len(a_pts) else 0; d = a_reps[i] - b_reps[j]
        out.append({"delta": 100 * float(a_pts[i] - b_pts[j]), "ci95": [100 * x for x in ci(d)]})
    return out


def contrast(d_pt, d_bs, seed_pts=None):
    out = {"delta": 100 * float(d_pt), "ci95": [100 * x for x in ci(d_bs)], "p": boot_p(d_bs)}
    if seed_pts is not None:
        out["seed_deltas"] = [100 * float(x) for x in seed_pts]; out["seed_sd"] = 100 * float(np.std(seed_pts))
    return out


def score_bench(bn: Bench, main, abl, sparse_rows, sv_primary):
    F = families(main, abl); R = {"n_items": len(bn.items), "n_groups": bn.boot.G}
    R["fixed_budget"] = {c: {"quality": 100 * bn.q(c), "ci95": [100 * x for x in ci(bn.bm(c))], "retention_rel": bn.q(c) / bn.q("full") if bn.q("full") else None,
                             "retention_abs_drop": 100 * (bn.q("full") - bn.q(c)), "mean_visual_tokens": float(bn.K[c].mean())} for c in bn.S}
    Bs = sorted({B for (_, B) in bn.T})
    R["latency_points_ms"] = {B: {L: bn.lat(f"static:grid:{k}", B) for L, k in LPOINTS.items()} for B in Bs}
    R["latency_points_ci95_ms"] = {B: {L: bn.Tci.get((f"static:grid:{k}", B)) for L, k in LPOINTS.items()} for B in Bs}
    R["row_latency_ms"] = {f"{c}@B{B}": {"median": v, "ci95": bn.Tci.get((c, B))} for (c, B), v in sorted(bn.T.items())}
    R["at_latency"] = {}
    for B in Bs:
        R["at_latency"][B] = {}
        for name, seeds in F.items():
            R["at_latency"][B][name] = {}
            for L, k in LPOINTS.items():
                Lms = bn.lat(f"static:grid:{k}", B); r = None if Lms is None else seed_mean_at(bn, seeds, Lms, B)
                R["at_latency"][B][name][L] = None if r is None else {"mean": 100 * r[0], "ci95": [100 * x for x in ci(r[1])], "seed_sd": 100 * float(np.std(r[2]))}
        if sparse_rows and B == 1:
            sv = [c for c in sparse_rows if not c.endswith(":ocr")]
            R["at_latency"][B]["SparseVLM"] = {}
            for L, k in LPOINTS.items():
                v = bn.at(bn.curve(sv, 1), bn.lat(f"static:grid:{k}", 1)) if bn.lat(f"static:grid:{k}", 1) is not None else None
                R["at_latency"][B]["SparseVLM"][L] = None if v is None else {"mean": 100 * v}
    C = {}
    for L, k in LPOINTS.items():                                                       # primary, B* = 4
        Lms = bn.lat(f"static:grid:{k}", PRIMARY_B)
        s1 = None if Lms is None else seed_mean_at(bn, F["S1"], Lms, PRIMARY_B)
        for other in ("S2", "S3", "grid", "random", "qsim"):
            o = None if Lms is None else seed_mean_at(bn, F[other], Lms, PRIMARY_B)
            key = f"S1-{other}@{L}"
            if s1 is None or o is None:
                C[key] = {"not_reached": True}
                continue
            seed_d = [a - b for a, b in zip(s1[2], o[2])] if len(o[2]) == len(s1[2]) else None
            C[key] = contrast(s1[0] - o[0], s1[1] - o[1], seed_d); C[key]["per_seed"] = per_seed(s1[2], s1[3], o[2], o[3])
            C[key]["role"] = "primary" if other in ("S2", "S3") else "secondary"
        sv = next((c for c in sparse_rows if c.startswith(f"sparsevlm:{k}:") and not c.endswith(":ocr")), None)
        if sv:
            s1k = [f"sel:{u}:{k}" for u in main["vcs"]]
            d = np.mean([bn.S[c] for c in s1k], 0) - bn.S[sv]
            C[f"S1-SparseVLM@{L}[budget]"] = dict(contrast(d.mean(), bn.boot.means(d), [bn.q(c) - bn.q(sv) for c in s1k]),
                                                  per_seed=per_seed([bn.q(c) for c in s1k], [bn.bm(c) for c in s1k], [bn.q(sv)], [bn.bm(sv)]),
                                                  matched=f"equal-token budget {k} (SparseVLM row {sv})", role="primary" if sv_primary == "budget" else "secondary")
            L1 = bn.lat(f"static:grid:{k}", 1); s1b1 = None if L1 is None else seed_mean_at(bn, F["S1"], L1, 1)
            cv = bn.curve([c for c in sparse_rows if not c.endswith(":ocr")], 1); svq = None if L1 is None else bn.at(cv, L1)
            key = f"S1-SparseVLM@{L}[latency_b1]"
            if s1b1 is None or svq is None:
                C[key] = {"not_reached": True, "role": "primary" if sv_primary == "latency_b1" else "secondary"}
            else:
                svr = bn.at(cv, L1, rep=True)
                C[key] = dict(contrast(s1b1[0] - svq, s1b1[1] - svr, [x - svq for x in s1b1[2]]), per_seed=per_seed(s1b1[2], s1b1[3], [svq], [svr]),
                              role="primary" if sv_primary == "latency_b1" else "secondary")
    for k in BUDGETS:                                                                  # ablations at fixed budget (same masks and information)
        for a, base in (("A1", "vcs"), ("A2", "direct")):
            xa = [f"sel:{u}:{k}" for u in abl[a]]; xb = [f"sel:{u}:{k}" for u in main[base]]
            if all(c in bn.S for c in xa + xb):
                d = np.mean([bn.S[c] for c in xa], 0) - np.mean([bn.S[c] for c in xb], 0)
                C[f"{a}-{NAMES[base]}@k{k}"] = dict(contrast(d.mean(), bn.boot.means(d), [bn.q(p) - bn.q(q) for p, q in zip(xa, xb)]), role="secondary")
    for o in main:                                                                     # A3: dynamic − fixed at the realised mean budget
        for f in RUN.FRACTIONS:
            pt, bs = [], []
            for u in main[o]:
                dc = f"dyn:{u}:{f}"; m = float(bn.K[dc].mean()); ks = list(BUDGETS) + [RUN.N_VIS]
                cfgs = [f"sel:{u}:{k}" for k in BUDGETS] + ["full"]; br = bracket(ks, m)
                i, j, w = br; a, b = cfgs[i], cfgs[j]
                pt.append(bn.q(dc) - ((1 - w) * bn.q(a) + w * bn.q(b))); bs.append(bn.bm(dc) - ((1 - w) * bn.bm(a) + w * bn.bm(b)))
            C[f"A3:{NAMES[o]}_dyn-fixed@f{f}"] = dict(contrast(np.mean(pt), np.mean(bs, 0), pt), role="secondary")
    R["contrasts"] = C
    target = bn.q("full") - 0.01; R["within_1pt_of_full_ms_B4"] = {}
    for name, seeds in F.items():
        vals = []
        for c in seeds:
            cv = bn.curve(c, PRIMARY_B); vals.append(None if cv is None else first_within([p[0] for p in cv], [bn.q(p[1]) for p in cv], target))
        R["within_1pt_of_full_ms_B4"][name] = vals
    full = bn.S["full"]; wrong = full < 0.5; right = ~wrong; R["preserve_correct"] = {"threshold": "score < 0.5 = wrong (TextVQA soft accuracy; 0 / 1 elsewhere)",
                                                                                      "full_wrong": int(wrong.sum())}
    for c, x in bn.S.items():
        if c != "full":
            R["preserve_correct"][c] = {"kept_error": float(np.mean(x[wrong] < 0.5)) if wrong.any() else None,
                                        "corrected": float(np.mean(x[wrong] >= 0.5)) if wrong.any() else None,
                                        "broke_correct": float(np.mean(x[right] < 0.5)) if right.any() else None}
    R["dynamic_budget_distribution"] = {c: {str(int(k)): float(np.mean(bn.K[c] == k)) for k in list(BUDGETS) + [RUN.N_VIS]} for c in bn.S if c.startswith("dyn:")}
    subsets = {}
    key = "structural" if bn.bench.startswith("gqa") else ("type" if bn.bench.startswith("chartqa") else None)
    if key:
        lab = np.array([str(it.get(key)) for it in bn.items])
        if bn.bench.startswith("chartqa"):
            lab = np.array(["human" if "human" in v.lower() else ("augmented" if "augmented" in v.lower() else "other:" + v) for v in lab])
        for v in sorted(set(lab)):
            m = lab == v; subsets[v] = {"n": int(m.sum()), "quality": {c: 100 * float(x[m].mean()) for c, x in bn.S.items()}}
    R["subsets"] = subsets
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sets-dir", required=True); ap.add_argument("--pred-dir", required=True); ap.add_argument("--timing-dir", required=True)
    ap.add_argument("--lr-choice", required=True); ap.add_argument("--out", required=True); ap.add_argument("--benches", nargs="+", default=["gqa_dev", "textvqa_val", "chartqa_val"])
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--sparsevlm-primary", choices=["budget", "latency_b1"], default="budget")
    ap.add_argument("--no-sparsevlm", action="store_true"); a = ap.parse_args(argv)
    import r7_d1_30_sparsevlm as SV
    main_u, abl_u = RUN.unit_names(json.loads(Path(a.lr_choice).read_text())); rep = main_u["vcs"][0]
    res = {"representative_unit": rep, "benches": {}, "notes": [
        "primary endpoint: quality at fixed measured latency at B* = 4 (nodesumo01) for our pipeline's rows",
        f"S1 - SparseVLM primary basis: {a.sparsevlm_primary} (coordinator decision D30-1: budget-matched primary, batch-1 latency-matched secondary)",
        "no latency claim is made against SparseVLM at B* = 4: its official code is batch-1 only",
        "latency medians carry bootstrap CIs over timing batches; they are not propagated into the quality-at-latency interpolation"]}
    for bench in a.benches:
        exp = RUN.config_list([u for o in main_u.values() for u in o], [u for o in abl_u.values() for u in o], ocr_anchor=bench.startswith("textvqa"))
        sv = [] if a.no_sparsevlm else [SV.row_tag(T, o) for T, o in SV.rows_for(bench, list(SV.ANCHORS + SV.ADAPTED))]
        bn = Bench(bench, a.sets_dir, a.pred_dir, a.timing_dir, exp + sv, a.reps, rep)
        res["benches"][bench] = score_bench(bn, main_u, abl_u, sv, a.sparsevlm_primary)
    for L in LPOINTS:                                                                  # Holm over the 9 primary contrasts at each latency point
        keys = [(b, k) for b in res["benches"] for k, v in res["benches"][b]["contrasts"].items() if f"@{L}" in k and v.get("role") == "primary" and "p" in v]
        if keys:
            for (b, k), pv in zip(keys, holm([res["benches"][b]["contrasts"][k]["p"] for b, k in keys])):
                res["benches"][b]["contrasts"][k]["p_holm"] = pv; res["benches"][b]["contrasts"][k]["holm_family_size"] = len(keys)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "d1_30_results.json").write_text(V.jdump(res, indent=1))
    print(V.jdump({b: {k: v for k, v in r["contrasts"].items() if v.get("role") == "primary"} for b, r in res["benches"].items()}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
