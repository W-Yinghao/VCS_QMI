"""r7 R7-30 D1-30 smoke summary (CPU): collects the LLaVA-side checks, the SparseVLM counter checks and reproduction of the R7-20 official
anchors (GQA fixture; TextVQA fixture with the OCR prompt), the timing rows and the scorer's output; writes <smoke>/smoke_summary.json and
reports/R7/R7_D1_30_SMOKE.md.  Smoke numbers are pipeline checks on 64-item fixture subsets with D1-20 smoke selectors — not results.
    python scripts/r7_d1_30_smoke_report.py --smoke S
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_run as RUN  # noqa: E402
import r7_vl_common as V  # noqa: E402

R720 = Path("/home/infres/yinwang/CS_QMI/outputs/R7_fixtures/d1")
REPORT = Path(__file__).resolve().parents[1] / "reports" / "R7" / "R7_D1_30_SMOKE.md"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", required=True); a = ap.parse_args(); sm = Path(a.smoke)
    llava = json.loads((sm / "smoke_report_llava.json").read_text()); S = {"llava": {k: v for k, v in llava.items() if k.startswith(("check_", "gpu", "load", "sets", "seconds"))}}
    sv_counters = []
    for f in sorted((sm / "pred").glob("*/sparsevlm_run_*.json")) + sorted((sm / "pred_repro").glob("*/sparsevlm_run_*.json")):
        for r in json.loads(f.read_text())["rows"]:
            sv_counters.append({k: r.get(k) for k in ("bench", "config", "layer_list", "predicted_equal_tokens", "equal_visual_tokens", "counter_check", "items")})
    S["sparsevlm_counters"] = sv_counters; repro = {}
    for T in (192, 64):
        ref = {}
        for x in (R720 / f"pred_sparsevlm{T}.jsonl").read_text().splitlines():
            j = json.loads(x); ref[(j["fixture"], j["qid"])] = j["text"]
        for b, fx, ocr in (("gqa_fixture", "gqa", False), ("textvqa_fixture", "textvqa", True)):
            mine = {r["qid"]: r["pred"] for r in V.read_jsonl(sm / "pred_repro" / b / f"{RUN.cfg_file(f'sparsevlm:{T}:official' + (':ocr' if ocr else ''))}.jsonl")}
            n = sum((fx, q) in ref for q in mine); same = sum(mine[q] == ref[(fx, q)] for q in mine if (fx, q) in ref)
            repro[f"sparsevlm{T}_{b}{'_ocr' if ocr else ''}"] = f"{same}/{n} identical to R7-20"
    S["sparsevlm_repro"] = repro
    S["timing"] = {}
    for f in sorted((sm / "timing").glob("*/timing_*.json")) + sorted((sm / "timing").glob("*/sparsevlm_timing_*.json")):
        j = json.loads(f.read_text()); S["timing"][f.name] = {"gpu": j.get("gpu"), "rows": [{k: r.get(k) for k in ("config", "B", "per_item_ms_median", "ttft_ms_median", "mean_visual_tokens")}
                                                                                         for r in j["rows"] if r["B"] in (1, 4) and (r["config"] in ("full",) or ":144" in r["config"] or ":0.5" in r["config"] or r["config"].startswith("sparsevlm"))]}
    res = json.loads((sm / "results" / "d1_30_results.json").read_text()); S["scorer"] = {}
    for b, r in res["benches"].items():
        S["scorer"][b] = {"n_items": r["n_items"], "full": r["fixed_budget"]["full"]["quality"], "primary": {k: v for k, v in r["contrasts"].items() if v.get("role") == "primary"},
                          "latency_points_ms_B4": r["latency_points_ms"].get("4", r["latency_points_ms"].get(4))}
    dr = json.loads((sm / "devrisk" / "d1_30_devrisk_smoke.json").read_text()); S["devrisk"] = {u: {k: v[k] for k in ("n_pairs", "spearman_r_h", "slope_h_on_r")} for u, v in dr["units"].items()}
    (sm / "smoke_summary.json").write_text(V.jdump(S, indent=1))
    L = ["# R7-30 D1-30 smoke (R7-20 fixtures only)", "", f"Job output: {sm}.  GPU {llava.get('gpu')}; pipeline checks on fixture subsets with the D1-20 *smoke* "
         "selectors (seeds 1-2 = seed-0 weights relabelled; dynamic tau calibrated on the fixture) — numbers are not results.", "", "## Checks", ""]
    for k, v in S["llava"].items():
        if k.startswith("check_"):
            L.append(f"- {k[6:]}: {json.dumps(v)}")
    for k, v in repro.items():
        L.append(f"- SparseVLM reproduction {k}: {v}")
    bad = [c for c in sv_counters if c["counter_check"] is not True]
    L.append(f"- SparseVLM official counter == predicted for {len(sv_counters) - len(bad)}/{len(sv_counters)} rows" + (f" (mismatch: {bad})" if bad else ""))
    L += ["", "## SparseVLM rows (official 'Equal Tokens' counter)", "", "| bench | row | layer list | predicted | official |", "|---|---|---|---|---|"]
    seen = set()
    for c in sv_counters:
        if (c["bench"], c["config"]) not in seen:
            seen.add((c["bench"], c["config"])); L.append(f"| {c['bench']} | {c['config']} | {c['layer_list']} | {c['predicted_equal_tokens']:.3f} | {c['equal_visual_tokens']} |")
    L += ["", "## Scorer (smoke subsets)", ""]
    for b, r in S["scorer"].items():
        L.append(f"- {b}: n = {r['n_items']}, full = {r['full']:.1f}; L25/L50/L75 at B = 4 (ms): {r['latency_points_ms_B4']}")
        for k, v in r["primary"].items():
            L.append(f"  - {k}: " + ("not reached" if v.get("not_reached") else f"Δ {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}], p_holm {v.get('p_holm')}"))
    L += ["", "## DEV-risk code path (probe records, fixture binary questions)", ""] + [f"- {u}: {json.dumps(v)}" for u, v in S["devrisk"].items()]
    REPORT.write_text("\n".join(L) + "\n"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
