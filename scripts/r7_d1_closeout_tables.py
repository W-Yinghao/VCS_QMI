#!/usr/bin/env python3
"""R7 D1 closeout tables (2026-10-17).

Every results table (T0-T18) and every result number quoted in reports/R7/R7_D1_CLOSEOUT_REPORT_20261017.md (and in the Corrections sections
it adds to earlier reports) is printed by this script from the committed results files in SOURCES; the report's clause table in §2 is a
hand-written summary of these outputs.  It formats, counts, takes seed means / sd / minima / maxima and re-applies frozen decision rules (DF-8, E2-8, the
REV10 rule) to the stored point estimates and Holm p-values.  It does no resampling and computes no new test or p-value.  Run from the
repository root:

    python3 scripts/r7_d1_closeout_tables.py                                                     # print everything
    python3 scripts/r7_d1_closeout_tables.py --check reports/R7/R7_D1_CLOSEOUT_REPORT_20261017.md   # + verify the tables appear verbatim
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {  # key: (path, results-only commit that added the file; none of these files was edited afterwards)
    "d1_10": ("reports/R7/R7_D1_10_results.json", "4997cb6"),
    "d1_20": ("reports/R7/R7_D1_20_CAL_READOUT.json", "3edb9cc"),
    "d1_30": ("reports/R7/R7_D1_30_RESULTS.json", "750dc61"),
    "d1_full": ("reports/R7/R7_D1_FULL_results/d1full_results.json", "7a89736"),
    "r50_L": ("reports/R7/R7_50_results_L/r7_50_results.json", "81c89df"),
    "r50_F": ("reports/R7/R7_50_results_F/r7_50_results.json", "ee172c7"),
    "r50_Q": ("reports/R7/R7_50_results_Q/r7_50_results.json", "1bbeb8e"),
    "r40a_t": ("reports/R7/R7_40A_QWEN_REFERENCE_TIMING.json", "6ca09ec"),  # Qwen regime rule (R7-40a)
    "r40c": ("reports/R7/R7_40C_results/r7_40c_results.json", "e8061c7"),  # Qwen 10k per-seed rows
    "e1": ("reports/R7/R7_E1_results/r7_e1_results.json", "7d98a17"),
    "e2": ("reports/R7/R7_E2_results/e2_claims.json", "895ca7b"),
    "e2_rows": ("reports/R7/R7_E2_results/r7_40c_results_E2_recompute.json", "895ca7b"),  # Qwen full-FIT per-seed rows
    "rev8": ("reports/REV8/REV8_DIAG_RESULTS/rev8_diag_results.json", "5dd8f37"),
    "rev9": ("reports/REV9/REV9_RESULTS/rev9_results.json", "821ef73"),
    "rev9f": ("reports/REV9/REV9_FULL_RESULTS/rev9_full_results.json", "be9a829"),
    "rev9h": ("reports/REV9/REV9_FULL_H100_RESULTS/rev9_full_h100_direct_secondary.json", "9637b5b"),
    "rev10": ("reports/REV9/REV10_RESULTS/rev10_results.json", "241a2c9"),
}
BENCH = ("gqa_dev", "textvqa_val", "chartqa_val")
TEST = {"gqa_dev": "gqa_testdev", "chartqa_val": "chartqa_test"}  # TextVQA test is not run (R7-50)
PTS = ("L25", "L50", "L75")
FRS = ("f0.25", "f0.5", "f0.75")
BUDGETS = (72, 144, 288, 432)
ALPHA = 0.05
DASH = "—"


def load(key: str) -> dict:
    return json.loads((ROOT / SOURCES[key][0]).read_text())


def ci(c) -> str:
    return f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def cell(c: dict | None, holm: bool = True) -> str:
    if c is None:
        return DASH
    p = c["p_holm"] if holm else c["p"]
    return f"{c['delta']:+.2f} {ci(c['ci95'])}, p {p:.3f}"


def neg(c: dict | None) -> str:
    if c is None:
        return DASH
    sd = c.get("seed_deltas") or []
    return f"{sum(v < 0 for v in sd)}/{len(sd)}"


def msd(v: list[float], nd: int = 3) -> str:
    return f"{st.mean(v):.{nd}f} ± {st.stdev(v):.{nd}f}"


def df8(full: dict, k10: dict) -> str:
    """DF-8 (R7_D1_FULL_STAGED_FREEZE_20261012.yaml:21-24), labels as in the D1-FULL report."""
    if (full["delta"] > 0) != (k10["delta"] > 0):
        return "scale-dependent sign"
    if full["p_holm"] < ALPHA and k10["p_holm"] < ALPHA:
        return "claim: S1 " + ("higher" if full["delta"] > 0 else "lower")
    return "no claim (Holm)"


def e2_8(full: dict, k10: dict) -> str:
    """E2-8 (R7_E2_STAGED_FREEZE_20261013.yaml:16-17), labels as stored by scripts/r7_e2_score.py."""
    sf, sk, same = full["p_holm"] < ALPHA, k10["p_holm"] < ALPHA, (full["delta"] > 0) == (k10["delta"] > 0)
    if sf and sk:
        return ("claim: S1 " + ("above" if full["delta"] > 0 else "below")) if same else "scale-dependent: significant with opposite signs"
    if sk:
        return "not claimed: significant at the 10k scale only (scale-dependent)"
    if sf:
        return "not claimed: significant at the full scale only (scale-dependent)"
    return "not claimed: neither scale significant"


SHORT_E2 = {"claim: S1 below": "claim: S1 below", "claim: S1 above": "claim: S1 above",
            "scale-dependent: significant with opposite signs": "opposite signs",
            "not claimed: significant at the 10k scale only (scale-dependent)": "10k only",
            "not claimed: significant at the full scale only (scale-dependent)": "full only",
            "not claimed: neither scale significant": "neither scale"}


# ------------------------------------------------------------------------------------------------------------------------------ LLaVA
def llava_key(name: str, p: str) -> str:
    return f"S1-SparseVLM@{p}[budget]" if name == "S1-SparseVLM" else f"{name}@{p}"


def llava_primary(name: str, S: dict, facts: dict) -> list[str]:
    D30, DF, L, F = S["d1_30"], S["d1_full"], S["r50_L"], S["r50_F"]
    out = [f"| benchmark | point | 10k FIT, DEV / val (D1-30) | full FIT, DEV / val (D1-FULL) | DF-8 | 10k FIT, official test (R7-50 L) "
           f"| full FIT, official test (R7-50 F) | seeds < 0: 10k / full / L / F |", "|---|---|---|---|---|---|---|---|"]
    for b in BENCH:
        for p in PTS:
            k = llava_key(name, p)
            k10, ful = D30["benches"][b]["contrasts"][k], DF["benches"][b]["full_scale"]["contrasts"][k]
            rec = DF["benches"][b]["k10_primary_family"][k]  # D1-FULL's recomputed 10k block must equal D1-30
            assert abs(rec["delta"] - k10["delta"]) < 1e-12 and abs(rec["p_holm"] - k10["p_holm"]) < 1e-12, (b, k)
            tl = L["benches"][TEST[b]]["contrasts"][k] if b in TEST else None
            tf = F["benches"][TEST[b]]["contrasts"][k] if b in TEST else None
            v = df8(ful, k10)
            facts.setdefault("df8", {})[(b, k)] = v
            out.append(f"| {b} | {p} | {cell(k10)} | {cell(ful)} | {v} | {cell(tl)} | {cell(tf)} | "
                       f"{neg(k10)} / {neg(ful)} / {neg(tl)} / {neg(tf)} |")
    return out


def llava_grid(S: dict) -> list[str]:
    D30, DF, L, F = S["d1_30"], S["d1_full"], S["r50_L"], S["r50_F"]
    out = ["| benchmark | point | 10k FIT, DEV / val (D1-30) | full FIT, DEV / val (D1-FULL) | 10k FIT, official test (R7-50 L) "
           "| full FIT, official test (R7-50 F) |", "|---|---|---|---|---|---|"]
    for b in BENCH:
        for p in PTS:
            k = f"S1-grid@{p}"
            cs = [D30["benches"][b]["contrasts"][k], DF["benches"][b]["full_scale"]["contrasts"][k],
                  L["benches"][TEST[b]]["contrasts"][k] if b in TEST else None, F["benches"][TEST[b]]["contrasts"][k] if b in TEST else None]
            for c in cs:
                assert c is None or (c["role"] == "secondary" and "p_holm" not in c), (b, k)
            out.append(f"| {b} | {p} | " + " | ".join(cell(c, holm=False) for c in cs) + " |")
    return out


def llava_grid_facts(S: dict, facts: dict) -> None:
    """T2 cells with S1 above the grid (point estimate / unadjusted p < 0.05), and cells where S1's point estimate exceeds both external rows
    (uniform grid at matched latency, secondary; adapted SparseVLM at matched budget, D30-1)."""
    D30, DF, L, F = S["d1_30"], S["d1_full"], S["r50_L"], S["r50_F"]
    srcs = (("10k DEV / val", lambda b: D30["benches"][b]["contrasts"]), ("full DEV / val", lambda b: DF["benches"][b]["full_scale"]["contrasts"]),
            ("phase L test", lambda b: L["benches"][TEST[b]]["contrasts"] if b in TEST else None),
            ("phase F test", lambda b: F["benches"][TEST[b]]["contrasts"] if b in TEST else None))
    pos, det, both, n = [], [], [], 0
    for b in BENCH:
        for p in PTS:
            for nm, get in srcs:
                C = get(b)
                if C is None:
                    continue
                n += 1
                g, sp = C[f"S1-grid@{p}"], C[f"S1-SparseVLM@{p}[budget]"]
                if g["delta"] > 0:
                    pos.append((b, p, nm, round(g["delta"], 2), round(g["p"], 3)))
                    if g["p"] < ALPHA:
                        det.append((b, p, nm))
                    if sp["delta"] > 0:
                        both.append((b, p, nm, round(g["delta"], 2), round(sp["delta"], 2), round(g["p"], 3), round(sp["p_holm"], 3)))
    facts["t2"] = (n, pos, det, both)


# ------------------------------------------------------------------------------------------------------------------------------ Qwen
def qwen_primary(name: str, S: dict, facts: dict) -> list[str]:
    E2, Q = S["e2"], S["r50_Q"]
    C = {(c["bench"], c["contrast"]): c for c in E2["claims"]}
    out = [f"| benchmark | fraction | 10k FIT, DEV / val (R7-40c) | full FIT, DEV / val (E2) | E2-8 | 10k FIT, official test (R7-50 Q) "
           f"| seeds < 0: 10k / full / Q |", "|---|---|---|---|---|---|---|"]
    for b in BENCH:
        for f in FRS:
            c = C[(b, f"{name}@{f}")]
            v = e2_8(c["full"], c["k10"])
            assert v == c["verdict"], (b, name, f, v, c["verdict"])
            q = Q["benches"][TEST[b]]["contrasts"][f"{name}@{f}"] if b in TEST else None
            assert q is None or (q["role"] == "primary" and q["holm_family_size"] == 6)
            out.append(f"| {b} | {f} | {cell(c['k10'])} | {cell(c['full'])} | {SHORT_E2[v]} | {cell(q)} | "
                       f"{neg(c['k10'])} / {neg(c['full'])} / {neg(q)} |")
    return out


def e2_counts(S: dict) -> list[str]:
    E2 = S["e2"]
    kinds = ("claim: S1 below", "claim: S1 above", "opposite signs", "10k only", "full only", "neither scale")
    out = ["| contrast | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
    tot = {k: 0 for k in kinds}
    for name in ("S1-S2", "S1-S3", "S1-grid"):
        n = {k: 0 for k in kinds}
        for c in E2["claims"]:
            if c["contrast"].split("@")[0] == name:
                n[SHORT_E2[c["verdict"]]] += 1
        for k in kinds:
            tot[k] += n[k]
        out.append(f"| {name} | " + " | ".join(str(n[k]) for k in kinds) + " |")
    out.append("| all 27 | " + " | ".join(str(tot[k]) for k in kinds) + " |")
    assert sum(tot.values()) == 27 and len(E2["claims"]) == 27
    return out


E1_F = ("0.125", "0.25", "0.5", "0.75")


def e1_counts(S: dict, facts: dict) -> list[str]:
    """E1 host column (R7-40c's 10k rows) plus E2's full-FIT seed means at the same fractions, all against the same uniform-grid column."""
    E1, E2, RE2 = S["e1"], S["e2"], S["e2_rows"]
    out = ["| benchmark | learned rows (S1, S2, S3) below the grid column: 10k FIT (R7-40c) | learned rows below the grid column: full FIT (E2) "
           "| MMTok / HiPrune above the grid column | MMTok / HiPrune − grid at f 0.75 | largest grid − S1, 10k FIT (f) |", "|---|---|---|---|---|---|"]
    ext_le05 = ext_all = 0
    full_above = []
    for b in BENCH:
        H = E1["benches"][b]["host_column_table"]
        FF = E2["scale_contrast"][b]["fixed_fraction"]
        below = below_f = above = 0
        worst = (0.0, "")
        for f in E1_F:
            g = H[f]["r7_40c"]["grid"]["quality"]
            assert abs(RE2["benches"][b]["fixed_fraction"][f"static:grid:{f}"]["quality"] - g) < 1e-9  # E2's grid row = the same column
            for s in ("S1", "S2", "S3"):
                assert abs(FF[f"{s}@f{f}"]["k10_quality"] - H[f]["r7_40c"][s]["quality"]) < 1e-9  # E2's 10k block = R7-40c's rows
                below += H[f]["r7_40c"][s]["quality"] < g
                q = FF[f"{s}@f{f}"]["full_quality"]
                below_f += q < g
                if q >= g:
                    full_above.append((b, f, s, q, g))
            n = sum(H[f]["e1"][m]["quality"] > g for m in ("mmtok", "hiprune"))
            above += n
            ext_all += n
            ext_le05 += n if f != "0.75" else 0
            if g - H[f]["r7_40c"]["S1"]["quality"] > worst[0]:
                worst = (g - H[f]["r7_40c"]["S1"]["quality"], f)
        d75 = [H["0.75"]["e1"][m]["quality"] - H["0.75"]["r7_40c"]["grid"]["quality"] for m in ("mmtok", "hiprune")]
        out.append(f"| {b} | {below} / 12 | {below_f} / 12 | {above} / 8 | {d75[0]:+.2f} / {d75[1]:+.2f} | {worst[0]:.2f} (f {worst[1]}) |")
    facts["e1_ext_above"] = (ext_le05, ext_all)
    facts["e2_full_above_grid"] = full_above
    return out


# ------------------------------------------------------------------------------------------------------------------------------ D1-10, D1-20, REV8
def d1_10(S: dict, facts: dict) -> list[str]:
    cells = S["d1_10"]["roles"]["CAL"]["cells"]
    out = ["| mechanism | kept | mean h | mean d | mean h − d [95 % CI] | abs(h − d) / h | per-item Spearman(h, d) [95 %] | calibrated-teacher band "
           "| rank rule |", "|---|---|---|---|---|---|---|---|---|"]
    ratios = {}
    for mech in ("grid", "random", "qsim"):
        for k in BUDGETS:
            c = cells[f"{mech}:{k}"]
            r, lo, hi = c["mean_residual"], *c["mean_residual_ci"]
            ratios[f"{mech}:{k}"] = abs(r) / c["mean_h"]
            rc = c["rank_ceiling"]
            fired = "fires" if rc["verdict"] != "not_below_calibrated_ceiling" else "does not fire"
            # frozen condition (protocol 359-360; scripts/r7_d1_analyze.py:65): fires iff the observed interval's upper end is below the band
            assert (rc["observed_boot_ci"][1] < rc["ceiling_band"][0]) == (fired == "fires"), (mech, k)
            assert abs(rc["margin"] - (rc["observed_boot_ci"][1] - rc["observed"])) < 1e-12
            out.append(f"| {mech} | {c['budget_fraction'] * 100:g} % | {c['mean_h']:.4f} | {c['mean_d']:.4f} | {r:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                       f"{ratios[f'{mech}:{k}']:.2f} | {rc['observed']:.3f} [{rc['observed_boot_ci'][0]:.3f}, {rc['observed_boot_ci'][1]:.3f}] | "
                       f"[{rc['ceiling_band'][0]:.3f}, {rc['ceiling_band'][1]:.3f}] | {fired} |")
    facts["d1_10_ratio_le_025"] = sum(v <= 0.25 for v in ratios.values())
    facts["d1_10_ratio_gt_025"] = {k: round(v, 3) for k, v in ratios.items() if v > 0.25}
    facts["d1_10_abs_residual_max"] = max(abs(c["mean_residual"]) for c in cells.values())
    facts["d1_10_ratio_max"] = max(ratios.items(), key=lambda kv: kv[1])
    facts["d1_10_rank_fired"] = sum(cells[k]["rank_ceiling"]["verdict"] != "not_below_calibrated_ceiling" for k in cells)
    facts["d1_10_residual_negative"] = {m: sum(cells[f"{m}:{k}"]["mean_residual"] < 0 for k in BUDGETS) for m in ("grid", "random", "qsim")}
    facts["d1_10_nonneg_residual"] = {k: (c["mean_residual"], c["mean_residual_ci"]) for k, c in cells.items() if c["mean_residual"] >= 0}
    facts["d1_10_below_band"] = {k: (c["rank_ceiling"]["observed"], c["rank_ceiling"]["ceiling_band"][0], c["rank_ceiling"]["observed_boot_ci"][1])
                                 for k, c in cells.items() if c["rank_ceiling"]["observed"] < c["rank_ceiling"]["ceiling_band"][0]}
    obs = [c["rank_ceiling"]["observed"] for c in cells.values()]
    facts["d1_10_rule05"] = (sum(c["rank_ceiling"]["rule_0p5_superseded"]["observed_below_0p5"] for c in cells.values()), min(obs), max(obs))
    fit = S["d1_10"]["roles"]["FIT"]["cells"]
    facts["d1_10_fit_fired"] = {k: (c["rank_ceiling"]["observed"], c["rank_ceiling"]["observed_boot_ci"], c["rank_ceiling"]["ceiling_band"])
                                for k, c in fit.items() if c["rank_ceiling"]["verdict"] != "not_below_calibrated_ceiling"}
    return out


REV8_UNITS = (("LLaVA 10k", "vcs", "llava10k__vcs_lr0.0003"), ("LLaVA 10k", "logistic", "llava10k__logistic_lr0.0001"),
              ("LLaVA 10k", "direct", "llava10k__direct_lr0.0001"), ("LLaVA FULL", "vcs", "llavafull__vcs_lr0.0001"),
              ("LLaVA FULL", "logistic", "llavafull__logistic_lr0.0001"), ("LLaVA FULL", "direct", "llavafull__direct_lr0.0001"),
              ("Qwen 10k", "vcs", "qwen10k__vcs_lr0.0003"), ("Qwen 10k", "logistic", "qwen10k__logistic_lr0.0001"),
              ("Qwen 10k", "direct", "qwen10k__direct_lr0.0003"))


def rev8(S: dict, facts: dict) -> list[str]:
    R8 = S["rev8"]["units"]
    out = ["| scale | objective (frozen lr) | pairwise accuracy within (q, k) | seeds with 95 % CI ∋ 0.5 | regret: critic's choice | "
           "regret: random choice | regret: teacher d choice (reference, uses host outputs) |", "|---|---|---|---|---|---|---|"]
    for scale, obj, pre in REV8_UNITS:
        W = [R8[f"{pre}_s{s}"]["A_R8_04"]["within_qk"] for s in range(3)]
        pa = [w["pairwise_accuracy"] for w in W]
        out.append(f"| {scale} | {obj} ({pre.split('_lr')[1]}) | {msd([p[0] for p in pa])} | {sum(p[1] <= 0.5 <= p[2] for p in pa)} / 3 | "
                   f"{msd([w['regret']['critic'][0] for w in W])} | {msd([w['regret']['random_choice_expected'][0] for w in W])} | "
                   f"{msd([w['regret']['teacher_d_choice'][0] for w in W])} |")
    return out


# ------------------------------------------------------------------------------------------------------------------------------ REV9 / REV10
CELLS = (  # scale, objective, phase-A lr, CAL-chosen, source, cell, role
    ("10k", "VCS", "3e-4", "yes", "rev9", "vcs_lr0.0003", "REV9"),
    ("10k", "logistic", "1e-4", "yes", "rev9", "logistic_lr0.0001", "REV9"),
    ("10k", "direct", "1e-4", "yes", "rev9", "direct_lr0.0001", "REV9"),
    ("10k", "VCS", "1e-4", "no", "rev10", "vcs_lr0.0001", "REV10"),
    ("10k", "logistic", "3e-4", "no", "rev10", "logistic_lr0.0003", "REV10"),
    ("full", "VCS", "1e-4", "yes", "rev9f", "vcs_lr0.0001", "REV9-FULL"),
    ("full", "logistic", "1e-4", "yes", "rev9f", "logistic_lr0.0001", "REV9-FULL"),
    ("full", "direct", "1e-4", "yes", "rev9h", "direct_lr0.0001", "REV9-FULL H100 secondary, seed 0 only"),
)


def counts(rows: list[dict]) -> dict:
    n = {"I1": [0, 0], "I2": [0, 0]}
    for r in rows:
        if r["p_holm"] < ALPHA:
            n[r["contrast"].split("-")[0]][0 if r["delta"] > 0 else 1] += 1
    return n


def pattern(n1: int, n2: int) -> str:
    if n2 >= 3 and abs(n1) <= 2:
        return "V"
    if n1 >= 3 and n2 <= -3:
        return "L"
    return "neither"


def phase_b(S: dict, facts: dict) -> list[str]:
    out = ["| scale | objective | phase-A lr | CAL-chosen | source | guard I0 (estimator / gate max abs diff) | I1 − frozen: Holm +, − | "
           "I2 − frozen: Holm +, − | net I1 | net I2 | pattern (REV10 definitions) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    pats, sig, deltas, n_contr = {}, 0, [], 0
    for scale, obj, lr, cal, src, c, role in CELLS:
        R = S[src]
        rows = R["primary"][c]
        assert len(rows) == 24, (src, c, len(rows))
        g = R["guard"][c]
        assert g["pass"] and g["max_abs_diff_estimator_vs_frozen"] == 0 and g["max_abs_diff_gate_vs_frozen"] <= 1e-3, (src, c)
        n = counts(rows)
        n1, n2 = n["I1"][0] - n["I1"][1], n["I2"][0] - n["I2"][1]
        pats[(src, c)] = pattern(n1, n2)
        if src != "rev9h":
            n_contr += len(rows)
            sig += sum(r["p_holm"] < ALPHA for r in rows)
            deltas += [r["delta"] for r in rows if r["p_holm"] < ALPHA]
        gtxt = f"{g['max_abs_diff_estimator_vs_frozen']} / {g['max_abs_diff_gate_vs_frozen']}" + (" (H100 rerun)" if src == "rev9h" else "")
        out.append(f"| {scale} | {obj} | {lr} | {cal} | {role} | {gtxt} | "
                   f"{n['I1'][0]}, {n['I1'][1]} | {n['I2'][0]}, {n['I2'][1]} | {n1:+d} | {n2:+d} | {pats[(src, c)]} |")
        if src == "rev9h":
            hs = [r["delta"] for r in rows if r["p_holm"] < ALPHA]
            facts["h100_sig"] = (len(hs), min(hs), max(hs))
    v, l = pats[("rev10", "vcs_lr0.0001")], pats[("rev10", "logistic_lr0.0003")]
    facts["rev10_rule"] = ("OBJECTIVE-DRIVEN" if (v, l) == ("V", "L") else "LR-DRIVEN" if (v, l) == ("L", "V") else "MIXED", v, l)
    facts["seeded_sig"] = (sig, n_contr, min(deltas), max(deltas))
    integ = []
    for src in ("rev9", "rev9f", "rev9h", "rev10"):
        I = S[src]["integrity"]
        integ.append((src, len(I), sum(x.get("dropped_lines", 0) + x.get("blank_lines", 0) + x.get("duplicate_qids", 0) for x in I.values())))
    facts["integrity"] = integ
    return out


def gap_cells(R: dict, c: str, arm: str, k: int) -> list[float]:
    return [R["gaps"][u][arm][str(k)]["gap"] for u in sorted(R["gaps"]) if u.rsplit("_s", 1)[0] == c and arm in R["gaps"][u]]


def gaps(S: dict, facts: dict) -> list[str]:
    out = ["| scale | objective | phase-A lr | frozen @72 | I1 @72 | I2 @72 | frozen @144 | I1 @144 | I2 @144 |", "|---|---|---|---|---|---|---|---|---|"]
    fr72 = {}
    for scale, obj, lr, cal, src, c, role in CELLS:
        R = S[src]
        cs = []
        for k in (72, 144):
            for arm in ("frozen", "I1", "I2"):
                v = gap_cells(R, c, arm, k)
                cs.append(f"{v[0]:+.4f} (n 1)" if len(v) == 1 else f"{st.mean(v):+.4f} [{min(v):+.4f}, {max(v):+.4f}]")
                if arm == "frozen" and k == 72:
                    fr72[(scale, obj, lr)] = v
        out.append(f"| {scale} | {obj} | {lr} | " + " | ".join(cs) + " |")
    facts["gap72_frozen"] = fr72
    return out


def grid_q(S: dict, b: str, k: int) -> float:
    return S["d1_30"]["benches"][b]["fixed_budget"][f"static:grid:{k}"]["quality"]


def acc(R: dict, b: str, c: str, k: int, arm: str) -> float:
    return st.mean(R["benches"][b][f"{c}@{k}"][arm])


def rev10_secondary(S: dict, facts: dict) -> tuple[list[str], list[str]]:
    corners = (("rev9", "vcs_lr0.0003", "VCS 3e-4 (CAL)"), ("rev10", "vcs_lr0.0001", "VCS 1e-4"),
               ("rev9", "logistic_lr0.0001", "logistic 1e-4 (CAL)"), ("rev10", "logistic_lr0.0003", "logistic 3e-4"))
    a = ["| benchmark | budget | uniform grid | " + " | ".join(f"{n}: frozen − grid" for _, _, n in corners) + " |", "|---|---|---|---|---|---|---|"]
    vcs_lr = []
    for b in BENCH:
        for k in BUDGETS:
            g = grid_q(S, b, k)
            fr = [acc(S[src], b, c, k, "frozen") for src, c, _ in corners]
            vcs_lr.append(((b, k), fr[1] - fr[0]))
            a.append(f"| {b} | {k} | {g:.2f} | " + " | ".join(f"{x - g:+.2f}" for x in fr) + " |")
    facts["vcs_1e4_minus_3e4"] = vcs_lr
    facts["vcs_1e4_minus_logistic_1e4"] = [((b, k), acc(S["rev10"], b, "vcs_lr0.0001", k, "frozen") - acc(S["rev9"], b, "logistic_lr0.0001", k, "frozen"))
                                           for b in BENCH for k in (72, 144)]
    bq = ["| benchmark | budget | VCS 1e-4: I2 − I1 [95 % CI], p (unadjusted) | logistic 3e-4: I2 − I1 [95 % CI], p (unadjusted) |", "|---|---|---|---|"]
    for b in BENCH:
        for k in BUDGETS:
            bq.append(f"| {b} | {k} | " + " | ".join(cell(S["rev10"]["benches"][b][f"{c}@{k}"]["I2-I1"], holm=False)
                                                      for c in ("vcs_lr0.0001", "logistic_lr0.0003")) + " |")
    return a, bq


def closing_rule(S: dict, facts: dict) -> tuple[list[str], list[str]]:
    """PROJECT_MAP_20261014.md:237 — S1 − grid narrowing at fixed budgets (grid row fixed, so the change equals S1's I − frozen)."""
    vc = (("10k", "rev9", "vcs_lr0.0003"), ("full", "rev9f", "vcs_lr0.0001"))
    t = ["| benchmark | budget | 10k VCS (3e-4): frozen − grid | I1 − frozen, Holm p | I2 − frozen, Holm p | full VCS (1e-4): frozen − grid "
         "| I1 − frozen, Holm p | I2 − frozen, Holm p |", "|---|---|---|---|---|---|---|---|"]
    summ = {}
    for b in BENCH:
        for k in BUDGETS:
            parts = []
            for scale, src, c in vc:
                R = S[src]
                by = {(r["bench"], r["cell"], r["contrast"]): r for r in R["primary"][c]}
                fg = acc(R, b, c, k, "frozen") - grid_q(S, b, k)
                parts.append(f"{fg:+.2f}")
                for iv in ("I1", "I2"):
                    r = by[(b, f"{c}@{k}", f"{iv}-frozen")]
                    parts.append(f"{r['delta']:+.2f}, {r['p_holm']:.3f}")
                    s = summ.setdefault((scale, iv), {"deficit": 0, "narrow": 0, "widen": 0})
                    if fg < 0:  # a cell where S1 is below the grid; a Holm-significant I − frozen narrows (> 0) or widens (< 0) the gap
                        s["deficit"] += 1
                        if r["p_holm"] < ALPHA:
                            s["narrow" if r["delta"] > 0 else "widen"] += 1
            t.append(f"| {b} | {k} | " + " | ".join(parts) + " |")
    facts["closing_rule"] = summ
    R = S["rev9f"]
    best = max((r["delta"], r["cell"], r["bench"]) for r in R["primary"]["vcs_lr0.0001"] if r["contrast"] == "I1-frozen" and r["p_holm"] < ALPHA
               and acc(R, r["bench"], "vcs_lr0.0001", int(r["cell"].split("@")[1]), "frozen") - grid_q(S, r["bench"], int(r["cell"].split("@")[1])) < 0)
    lg = max((r["delta"], r["cell"], r["bench"]) for r in R["primary"]["logistic_lr0.0001"] if r["contrast"] == "I1-frozen" and r["p_holm"] < ALPHA)
    facts["full_i1"] = (best, lg)
    pr = ["| scale | benchmark | budget | frozen S1 − frozen S2 | I1 S1 − I1 S2 | I2 S1 − I2 S2 | I2 S1 − I1 S2 (the arms that responded at 10k) |",
          "|---|---|---|---|---|---|---|"]
    rng = []
    for scale, src, s1, s2 in (("10k", "rev9", "vcs_lr0.0003", "logistic_lr0.0001"), ("full", "rev9f", "vcs_lr0.0001", "logistic_lr0.0001")):
        R = S[src]
        for b in BENCH:
            for k in BUDGETS:
                pairs = (("frozen", "frozen"), ("I1", "I1"), ("I2", "I2")) + ((("I2", "I1"),) if scale == "10k" else ())
                v = [acc(R, b, s1, k, x) - acc(R, b, s2, k, y) for x, y in pairs]
                if scale == "10k":
                    rng += v
                pr.append(f"| {scale} | {b} | {k} | " + " | ".join(f"{x:+.2f}" for x in v) + (" |" if scale == "10k" else f" | {DASH} |"))
    facts["pairing_range_10k"] = (min(rng), max(rng))
    return t, pr


def tight_budget(S: dict) -> list[str]:
    """25 % of the visual tokens (144 of 576 on LLaVA; f 0.25 on Qwen): seed means, fixed budget, no interval (descriptive)."""
    out = ["| host, FIT | benchmark | uniform grid | uniform random | external (LLaVA: SparseVLM adapted; Qwen: MMTok / HiPrune) | S1 VCS | "
           "S2 logistic | S3 direct risk | learned rows below the grid |", "|---|---|---|---|---|---|---|---|---|"]
    D30, DF, E1, E2 = S["d1_30"], S["d1_full"], S["e1"], S["e2"]
    for scale in ("10k", "full"):
        for b in BENCH:
            fb = D30["benches"][b]["fixed_budget"] if scale == "10k" else DF["benches"][b]["full_scale"]["fixed_budget"]
            if scale == "full":
                assert fb["static:grid:144"]["quality"] == D30["benches"][b]["fixed_budget"]["static:grid:144"]["quality"]
            units = (("vcs_lr0.0003", "logistic_lr0.0001", "direct_lr0.0001") if scale == "10k"
                     else ("F-vcs_lr0.0001", "F-logistic_lr0.0001", "F-direct_lr0.0001"))
            q = [st.mean(fb[f"sel:{u}_s{s}:144"]["quality"] for s in range(3)) for u in units]
            g = fb["static:grid:144"]["quality"]
            out.append(f"| LLaVA, {scale} | {b} | {g:.2f} | {fb['static:random:144']['quality']:.2f} | {fb['sparsevlm:144:adapted']['quality']:.2f} | "
                       + " | ".join(f"{x:.2f}" for x in q) + f" | {sum(x < g for x in q)} / 3 |")
    C = E2["scale_contrast"]
    for scale in ("10k", "full"):
        for b in BENCH:
            H = E1["benches"][b]["host_column_table"]["0.25"]
            g = H["r7_40c"]["grid"]["quality"]
            if scale == "10k":
                q = [H["r7_40c"][f"S{i}"]["quality"] for i in (1, 2, 3)]
            else:
                q = [C[b]["fixed_fraction"][f"S{i}@f0.25"]["full_quality"] for i in (1, 2, 3)]
                for i in (1, 2, 3):  # E2's 10k block is R7-40c's rows
                    assert abs(C[b]["fixed_fraction"][f"S{i}@f0.25"]["k10_quality"] - H["r7_40c"][f"S{i}"]["quality"]) < 1e-9
            ext = f"{H['e1']['mmtok']['quality']:.2f} / {H['e1']['hiprune']['quality']:.2f}"
            out.append(f"| Qwen, {scale} | {b} | {g:.2f} | {DASH} | {ext} | " + " | ".join(f"{x:.2f}" for x in q) + f" | {sum(x < g for x in q)} / 3 |")
    return out


def tight_budget_seeds(S: dict, facts: dict) -> None:
    """Per-seed version of T18 (25 %): each trained seed against the grid and against the lowest external row of its host (q above it = the
    seed is not below every external row, the per-seed negation of T18's last count)."""
    D30, DF, E1, E2, R40, RE2 = S["d1_30"], S["d1_full"], S["e1"], S["e2"], S["r40c"], S["e2_rows"]
    n, above = {}, []
    for scale in ("10k", "full"):
        for b in BENCH:
            fb = D30["benches"][b]["fixed_budget"] if scale == "10k" else DF["benches"][b]["full_scale"]["fixed_budget"]
            units = (("vcs_lr0.0003", "logistic_lr0.0001", "direct_lr0.0001") if scale == "10k"
                     else ("F-vcs_lr0.0001", "F-logistic_lr0.0001", "F-direct_lr0.0001"))
            g, ext = fb["static:grid:144"]["quality"], fb["sparsevlm:144:adapted"]["quality"]
            for i, u in enumerate(units):
                key = ("LLaVA", scale, f"S{i + 1}")
                for s in range(3):
                    q = fb[f"sel:{u}_s{s}:144"]["quality"]
                    c = n.setdefault(key, [0, 0, 0])
                    c[0] += q > g; c[1] += q > ext; c[2] += 1
                    if q > g:
                        above.append(("LLaVA", scale, b, f"S{i + 1}", s, round(q, 2), round(g, 2), round(ext, 2)))
    for scale, R, U in (("10k", R40, E2["units"]["k10"]), ("full", RE2, E2["units"]["full"])):
        for b in BENCH:
            H = E1["benches"][b]["host_column_table"]["0.25"]
            fb = R["benches"][b]["fixed_fraction"]
            g, ext = fb["static:grid:0.25"]["quality"], min(H["e1"]["mmtok"]["quality"], H["e1"]["hiprune"]["quality"])
            assert abs(g - H["r7_40c"]["grid"]["quality"]) < 1e-9
            for i, obj in enumerate(("vcs", "logistic", "direct")):
                qs = [fb[f"sel:{u}:0.25"]["quality"] for u in U[obj]]
                ref = (H["r7_40c"][f"S{i + 1}"]["quality"] if scale == "10k"
                       else E2["scale_contrast"][b]["fixed_fraction"][f"S{i + 1}@f0.25"]["full_quality"])
                assert abs(st.mean(qs) - ref) < 1e-9, (scale, b, obj)  # seed means = the T18 numbers
                key = ("Qwen", scale, f"S{i + 1}")
                for s, q in enumerate(qs):
                    c = n.setdefault(key, [0, 0, 0])
                    c[0] += q > g; c[1] += q > ext; c[2] += 1
                    if q > g:
                        above.append(("Qwen", scale, b, f"S{i + 1}", s, round(q, 2), round(g, 2), round(ext, 2)))
    facts["t18_seeds"] = (n, above)


def tight_budget_counts(rows: list[str]) -> dict:
    """Counts over the 12 (host, FIT, benchmark) rows of T18: learned rows below the grid / below every external row."""
    n = {"S1": [0, 0], "S2": [0, 0], "S3": [0, 0]}
    for r in rows[2:]:
        f = [x.strip() for x in r.strip("|").split("|")]
        g, ext = float(f[2]), [float(x) for x in f[4].split(" / ")]
        for i, sname in enumerate(("S1", "S2", "S3")):
            q = float(f[5 + i])
            n[sname][0] += q < g
            n[sname][1] += q < min(ext)
    return n


# ------------------------------------------------------------------------------------------------------------------------------ main
def sources_table() -> list[str]:
    out = ["| key | file | results-only commit | sha256 (first 16) |", "|---|---|---|---|"]
    for k, (p, c) in SOURCES.items():
        out.append(f"| {k} | `{p}` | {c} | {hashlib.sha256((ROOT / p).read_bytes()).hexdigest()[:16]} |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="report file that must contain every generated table verbatim")
    a = ap.parse_args()
    S = {k: load(k) for k in SOURCES}
    facts: dict = {}
    tables = {
        "T0 sources": sources_table(),
        "T1 S1 − SparseVLM (LLaVA)": llava_primary("S1-SparseVLM", S, facts),
        "T2 S1 − uniform grid (LLaVA, secondary)": llava_grid(S),
        "T3 S1 − uniform grid (Qwen)": qwen_primary("S1-grid", S, facts),
        "T4 S1 − S3 (LLaVA)": llava_primary("S1-S3", S, facts),
        "T5 S1 − S3 (Qwen)": qwen_primary("S1-S3", S, facts),
        "T6 D1-10 CAL residual": d1_10(S, facts),
        "T7 S1 − S2 (LLaVA)": llava_primary("S1-S2", S, facts),
        "T8 S1 − S2 (Qwen)": qwen_primary("S1-S2", S, facts),
        "T9 E2-8 verdict counts": e2_counts(S),
        "T10 E1 descriptive counts": e1_counts(S, facts),
        "T11 REV8 within-question choice": rev8(S, facts),
    }
    t12, t13 = closing_rule(S, facts)
    tables["T12 closing rule: S1 − grid under I1 / I2"] = t12
    tables["T13 S1 − S2 by arm pairing (descriptive)"] = t13
    tables["T14 phase-B responses"] = phase_b(S, facts)
    tables["T15 soft − hard gap"] = gaps(S, facts)
    t16, t17 = rev10_secondary(S, facts)
    tables["T16 REV10 secondary: frozen − grid"] = t16
    tables["T17 REV10 secondary: I2 − I1"] = t17
    tables["T18 learned vs training-free rows at 25 %"] = tight_budget(S)
    facts["t18"] = tight_budget_counts(tables["T18 learned vs training-free rows at 25 %"])
    tight_budget_seeds(S, facts)
    llava_grid_facts(S, facts)
    for name, rows in tables.items():
        print(f"### {name}\n")
        print("\n".join(rows) + "\n")
    # ---------------------------------------------------------------- numbers quoted in the text
    d8 = facts["df8"]
    n8 = {v: sum(1 for x in d8.values() if x == v) for v in sorted(set(d8.values()))}
    print("### numbers quoted in the text\n")
    print(f"- DF-8 over the 27 LLaVA primary contrasts: {n8}")
    cal = S["d1_10"]["roles"]["CAL"]
    print(f"- D1-10 CAL set: {cal['n_questions']} questions on {cal['n_images']} images")
    ch = S["d1_20"]["choice"]
    v = ch["vcs"]["cal_spearman"]
    print(f"- D1-20 lr choice margin for vcs (seed 0): {v['0.0003'] - v['0.0001']:.4f}")
    smoke = ROOT.parent / "outputs/R7_D1/smoke/timing/timing_report.json"  # D1-10 timing smoke, outside the repository
    if smoke.exists():
        tot = {(r["batch"], r["visual_tokens"]): r["total_ms_per_item"] for r in json.loads(smoke.read_text())["rows"]}
        print("- D1-10 timing smoke (regime rule input; outside the repository): uniform grid 144 vs 576 tokens saves " + ", ".join(
            f"{100 * (tot[(b, 576)] - tot[(b, 144)]) / tot[(b, 576)]:.1f} % at B = {b}" for b in sorted({b for b, _ in tot}) if (b, 576) in tot and (b, 144) in tot))
    else:
        print("- D1-10 timing smoke file not present here; the regime-rule figures are quoted from the frozen protocol text")
    print("- D1-20 CAL lr choice (seed 0, CAL Spearman(r, h)): " + "; ".join(
        f"{o} chose {v['chosen_lr']:g} ({', '.join(f'{lr}: {s:.4f}' for lr, s in v['cal_spearman'].items())})" for o, v in ch.items()))
    print("- D1-20 CAL Spearman(r, h), mean ± sd over 3 seeds: " + "; ".join(f"{g} {msd(v, 4)}" for g, v in S["d1_20"]["groups"].items()))
    r = facts["d1_10_ratio_max"]
    print(f"- D1-10 CAL: abs(h − d) / h <= 0.25 in {facts['d1_10_ratio_le_025']} / 12 cells; largest {r[0]} = {r[1]:.3f}; "
          f"rank rule fires in {facts['d1_10_rank_fired']} / 12 CAL cells; negative residual cells per mechanism {facts['d1_10_residual_negative']}")
    print(f"- D1-10 CAL cells with abs(h − d) / h > 0.25: {facts['d1_10_ratio_gt_025']}; largest abs(h − d) {facts['d1_10_abs_residual_max']:.4f}")
    print("- D1-10 CAL cells with a non-negative residual: " + "; ".join(f"{k} {r:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]" for k, (r, c) in facts["d1_10_nonneg_residual"].items()))
    print("- D1-10 CAL cells with the observed Spearman below the band's lower end (observed / band lower end / interval upper end): " + "; ".join(
        f"{k} {o:.3f} / {lo:.3f} / {hi:.3f}" for k, (o, lo, hi) in facts["d1_10_below_band"].items()))
    nb, omin, omax = facts["d1_10_rule05"]
    print(f"- D1-10 CAL superseded 0.5 line: observed Spearman below 0.5 in {nb} / 12 cells (range {omin:.3f}–{omax:.3f})")
    print("- D1-10 FIT (descriptive) cells where the rank rule fires: " + "; ".join(
        f"{k} {o:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] against [{bd[0]:.3f}, {bd[1]:.3f}]" for k, (o, ci, bd) in facts["d1_10_fit_fired"].items()))
    sav = []
    for b in BENCH:
        rl = S["d1_30"]["benches"][b]["row_latency_ms"]
        f4, g4 = rl["full@B4"]["median"], rl["static:grid:144@B4"]["median"]
        sav.append(f"{b} full {f4:.2f} ms, grid 144 {g4:.2f} ms, saving {100 * (f4 - g4) / f4:.1f} %")
    print("- D1-30 timing at B* = 4 (median per-item latency, profiling workload): " + "; ".join(sav))
    T = S["r40a_t"]
    sv = T["saving_vs_full"]
    assert T["B_star"] is None and all(sv[f"B{b}_f0.25"] < 0.25 for b in (1, 4, 16, 32))  # no batch size meets the 25 % rule at 25 % kept
    best = max(sv.items(), key=lambda kv: kv[1])
    b1 = [v for k, v in sv.items() if k.startswith("B1_")]
    print(f"- R7-40a Qwen regime rule ({T['gpu']}, {T['host']}): B* = {T['B_star']}; largest end-to-end saving {100 * best[1]:.1f} % ({best[0]}); "
          f"at 25 % kept: " + ", ".join(f"B {b} {100 * sv[f'B{b}_f0.25']:.1f} %" for b in (1, 4, 16, 32)) + f"; B 1 range {100 * min(b1):.1f}–{100 * max(b1):.1f} %")
    n, pos, det, both = facts["t2"]
    print(f"- T2 (S1 − grid, LLaVA, {n} cells): point estimate > 0 in {len(pos)}: {pos}; of these unadjusted p < 0.05: {det}")
    print(f"- T1 + T2: S1 point estimate above both the grid and SparseVLM (grid delta, SparseVLM delta, grid p, SparseVLM Holm p): {both}")
    print("- T18 (25 %, 12 host x FIT x benchmark rows): rows below the grid / below every external row: " + "; ".join(
        f"{k} {v[0]} / {v[1]}" for k, v in facts["t18"].items()))
    sn, sa = facts["t18_seeds"]
    print("- T18 per seed (3 benchmarks x 3 seeds per host x FIT x selector): seeds above the grid / above at least one external row: " + "; ".join(
        f"{h} {sc} {sel} {v[0]} / {v[1]} of {v[2]}" for (h, sc, sel), v in sn.items()))
    print(f"- T18 per seed, seeds above the grid (host, FIT, benchmark, selector, seed, quality, grid, lowest external row): {sa}")
    ea, eb = facts["e1_ext_above"]
    print(f"- E1: MMTok / HiPrune above the grid column in {ea} / 18 cells at f <= 0.5 and {eb} / 24 at f 0.125–0.75")
    print("- E2 full-FIT learned rows at or above the grid column (benchmark, f, selector, quality, grid): " + str(
        [(b, f, s, round(q, 2), round(g, 2)) for b, f, s, q, g in facts["e2_full_above_grid"]]))
    fl = S["rev8"]["units"]
    print("- REV8 flag R8_04_pooled_mostly_between_budgets set in " + str(sum(bool(u["flags"]["R8_04_pooled_mostly_between_budgets"]) for u in fl.values()))
          + f" of {len(fl)} units")
    sig, n, lo, hi = facts["seeded_sig"]
    print(f"- REV9 / REV9-FULL / REV10 seeded primary cells: {sig} of {n} contrasts Holm-significant; Holm-significant deltas from {lo:+.2f} to {hi:+.2f}")
    hn, hlo, hhi = facts["h100_sig"]
    print(f"- REV9-FULL H100 direct secondary (seed 0 only): {hn} of 24 contrasts Holm-significant, from {hlo:+.2f} to {hhi:+.2f}")
    print(f"- REV10 frozen decision rule: {facts['rev10_rule'][0]} (vcs_lr0.0001 pattern {facts['rev10_rule'][1]}, "
          f"logistic_lr0.0003 pattern {facts['rev10_rule'][2]})")
    print("- prediction-file integrity (files, dropped + blank + duplicate lines): " + "; ".join(f"{s} {nf} files, {bad}" for s, nf, bad in facts["integrity"]))
    g = facts["gap72_frozen"]
    print("- frozen soft − hard gap at 72 per unit: " + "; ".join(f"{k[0]} {k[1]} {k[2]}: " + ", ".join(f"{x:+.4f}" for x in v) for k, v in g.items()))
    lr3 = [abs(x) for k, v in g.items() if k[0] == "10k" and k[2] == "3e-4" for x in v]
    lr1 = [abs(x) for k, v in g.items() if k[2] == "1e-4" and k[1] != "direct" for x in v]
    print(f"- |frozen gap| at 72: lr 3e-4 units min {min(lr3):.4f}; lr 1e-4 VCS / logistic units max {max(lr1):.4f}")
    r8 = [(u, k, b["relax_det"][0]) for u, v in S["rev8"]["units"].items() if "_lr0.0001_" in u for k, b in v["B_R8_02"]["by_budget"].items()]
    grp: dict = {}
    for u, k, x in r8:
        grp.setdefault((u.rsplit("_s", 1)[0], k), []).append(x)
    gm = max(((abs(st.mean(v)), g) for g, v in grp.items()))
    over = sorted((x for x in r8 if abs(x[2]) > 0.004), key=lambda x: x[2])
    print(f"- REV8 SD − HD (relax_det) at phase-A lr 1e-4: {len({u for u, _, _ in r8})} units; largest |seed mean| {gm[0]:.4f} ({gm[1][0]} at {gm[1][1]}); "
          f"per unit, {len(over)} values in {len({u for u, _, _ in over})} units exceed 0.004: " + "; ".join(f"{u} at {k} {x:+.4f}" for u, k, x in over))
    cr = facts["closing_rule"]
    print("- closing rule (S1 − grid deficit cells at fixed budgets, Holm narrowing / widening): " + "; ".join(
        f"{s} {iv}: deficit {v['deficit']}, narrowed {v['narrow']}, widened {v['widen']}" for (s, iv), v in sorted(cr.items())))
    (b1, c1, n1), (b2, c2, n2) = facts["full_i1"]
    print(f"- full scale, I1: largest Holm-significant change of a VCS deficit cell {b1:+.2f} ({n1} {c1}); largest logistic I1 gain {b2:+.2f} ({n2} {c2})")
    R9 = S["rev9"]
    c = R9["benches"]["textvqa_val"]
    print("- 10k logistic 1e-4 TextVQA frozen accuracy per seed: " + "; ".join(
        f"@{k} " + " / ".join(f"{x:.2f}" for x in c[f"logistic_lr0.0001@{k}"]["frozen"]) for k in (72, 144))
          + "; I2 − frozen per seed: " + "; ".join(f"@{k} " + " / ".join(f"{x:+.2f}" for x in c[f"logistic_lr0.0001@{k}"]["I2-frozen"]["per_seed"])
                                                    for k in (72, 144)))
    lo, hi = facts["pairing_range_10k"]
    print(f"- 10k S1 − S2 at fixed budgets across the four arm pairings of T13: {lo:+.2f} to {hi:+.2f}")
    pos = [(bk, round(x, 2)) for bk, x in facts["vcs_1e4_minus_3e4"]]
    print(f"- frozen VCS 1e-4 − frozen VCS 3e-4 at 10k (seed means; descriptive): {pos}")
    print("- frozen VCS 1e-4 − frozen logistic 1e-4 at 10k (seed means; descriptive): "
          + str([(bk, round(x, 2)) for bk, x in facts["vcs_1e4_minus_logistic_1e4"]]))
    nr = tot = 0
    for R, get in ((S["d1_30"], lambda b: S["d1_30"]["benches"][b]["contrasts"]), (S["d1_full"], lambda b: S["d1_full"]["benches"][b]["full_scale"]["contrasts"]),
                   (S["r50_L"], lambda b: S["r50_L"]["benches"][TEST[b]]["contrasts"]), (S["r50_F"], lambda b: S["r50_F"]["benches"][TEST[b]]["contrasts"])):
        for b in (BENCH if R in (S["d1_30"], S["d1_full"]) else tuple(TEST)):
            for pnt in PTS:
                tot += 1
                nr += bool(get(b)[f"S1-SparseVLM@{pnt}[latency_b1]"].get("not_reached"))
    print(f"- S1 − SparseVLM latency-matched at B = 1 (secondary): not reached in {nr} of {tot} cells (D1-30, D1-FULL, R7-50 L, F)")
    L, F, Q = S["r50_L"], S["r50_F"], S["r50_Q"]
    for nm, R in (("L", L), ("F", F)):
        for b in ("gqa_testdev", "chartqa_test"):
            c = R["benches"][b]["contrasts"]["S1-S2@L25"]
            print(f"- R7-50 {nm} {b} S1-S2@L25: {c['delta']:+.4f} {ci(c['ci95'])} p_holm {c['p_holm']:.4f} seeds "
                  + ", ".join(f"{x:+.2f}" for x in c["seed_deltas"]))
    rows = F["benches"]["gqa_testdev"]["rows"]
    fb = {"uniform grid": rows["static:grid:144"]["quality"], "SparseVLM (adapted)": rows["sparsevlm:144:adapted"]["quality"]}
    for o, nm in (("logistic", "S2"), ("vcs", "S1"), ("direct", "S3")):
        fb[nm] = st.mean(rows[f"sel:{o}_lr0.0001_s{s}:144"]["quality"] for s in range(3))
    print(f"- R7-50 F GQA testdev, 144 tokens (25 %), full {rows['full']['quality']:.2f}: " + " > ".join(
        f"{k} {v:.2f}" for k, v in sorted(fb.items(), key=lambda kv: -kv[1])))
    print("- R7-50 Q S1-S2: " + "; ".join(f"{b} {f} {Q['benches'][b]['contrasts'][f'S1-S2@{f}']['delta']:+.2f} (p_holm "
                                           f"{Q['benches'][b]['contrasts'][f'S1-S2@{f}']['p_holm']:.3f})" for b in ("gqa_testdev", "chartqa_test") for f in FRS))
    if a.check:
        text = Path(a.check).read_text()
        missing = [name for name, rows in tables.items() if "\n".join(rows) not in text]
        print(f"\ncheck {a.check}: {len(tables) - len(missing)} / {len(tables)} tables found verbatim" + (f"; MISSING: {missing}" if missing else ""))
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
