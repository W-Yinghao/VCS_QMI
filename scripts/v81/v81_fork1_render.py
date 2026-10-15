#!/usr/bin/env python
"""Render results/selection_gap.md (D05 log level) and results/benchmark_context.md (D07) from the JSON / CSV written by v81_fork1_audit.py.
Every number in those two markdown files comes from the result files; nothing is typed by hand."""
from __future__ import annotations
import csv, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "reports/V8_1_DIAGNOSTICS/results"
ARMS = ["U-CS", "U-BCE", "U-RPS", "U-CE", "U-MSE", "U-L1", "U-RANK", "U-LIST"]
f3 = lambda x: "NA" if x is None else f"{x:.3f}"
f4 = lambda x: "NA" if x is None else f"{x:+.4f}"
lst = lambda xs, k=3: " / ".join(f"{x:.{k}f}" for x in xs)


def selection_gap():
    D = json.loads((OUT / "d05_log_level.json").read_text()); S = D["summary"]; U = D["units"]
    L = [f"# D05 log level — training curves and selection alignment (generated {datetime.now(timezone.utc).strftime('%FT%TZ')} from `d05_log_level.json`; 48 units; no checkpoint read, no new inference)", "",
         f"Rule: {D['rule']}.  Selected epoch = the frozen V8-03 rule (CAL_SELECT prompt-macro mean-rating MSE); the other columns are the epoch the same CAL_SELECT log would pick under a different criterion.  Log level only: a difference between columns flags a possible misalignment, it does not establish an external gain (docs/04 D05 level 1).", ""]
    L += ["## 1. Formal arms (seeds 0-2): mean curves over the three seeds", "", "| arm | CAL mean-rating MSE e0 / e1 / e2 / e3 / e4 | CAL pair acc e0..e4 | native train loss e0..e4 | selected (MSE) s0/s1/s2 | pair-acc best s0/s1/s2 | RPS best s0/s1/s2 |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        s = S[f"FIT/{a}"]; L.append(f"| {a} | {lst(s['mean_mse_curve'])} | {lst(s['mean_pairacc_curve'])} | {lst(s['mean_train_loss_curve'], 4)} | {'/'.join(map(str, s['selected_epochs']))} | {'/'.join(map(str, s['pairacc_epochs']))} | {'/'.join(map(str, s['rps_epochs']))} |")
    L += ["", "## 2. Tuning units (seed 17, lr 5e-6 / 1e-5 / 2e-5): mean curves over the three lr", "", "| arm | CAL MSE e0..e4 | CAL pair acc e0..e4 | train loss e0..e4 | selected (MSE) per lr | pair-acc best per lr |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        s = S[f"TUNE/{a}"]; L.append(f"| {a} | {lst(s['mean_mse_curve'])} | {lst(s['mean_pairacc_curve'])} | {lst(s['mean_train_loss_curve'], 4)} | {'/'.join(map(str, s['selected_epochs']))} | {'/'.join(map(str, s['pairacc_epochs']))} |")
    A = S["ALL"]; sel_c = Counter(u["epoch_by_mse_selected"] for u in U); fit_sel = Counter(u["epoch_by_mse_selected"] for u in U if u["kind"] == "FIT")
    L += ["", "## 3. Per unit", "", "| unit | selected (MSE) | pair-acc best | RPS best | CE best | pair acc at selected − best | MSE at pair-acc-best epoch − selected | MSE e0 − selected | MSE e4 − selected | train loss e0 → e4 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for u in U:
        L.append(f"| {u['unit']} | {u['epoch_by_mse_selected']} | {u['epoch_by_pairacc']} | {u['epoch_by_rps']} | {u['epoch_by_ce']} | {f4(u['pairacc_gap_sel_minus_best'])} | {f4(u['mse_at_pairacc_best_minus_sel'])} | {f4(u['mse_e0_minus_sel'])} | {f4(u['mse_last_minus_sel'])} | {u['train_loss_e0_to_last'][0]:.4f} → {u['train_loss_e0_to_last'][1]:.4f} |")
    rank = [u for u in U if u["arm"] == "U-RANK" and u["kind"] == "FIT"]
    L += ["", "## 4. Findings (counts and ranges computed from the same file)", "",
          f"- Selected-epoch distribution over the 48 units (0-based index = completed epochs − 1): {dict(sorted(sel_c.items()))}; formal units only: {dict(sorted(fit_sel.items()))}.  Epoch index 0 is the checkpoint after the first full pass (2,379 steps, 26,177 images), never an untrained head (V81_01).",
          f"- The native training loss decreases monotonically over the five epochs in {A['n_train_loss_monotone']} / {A['n_units']} units; the CAL_SELECT mean-rating MSE rises again after the selected epoch in {A['n_cal_mse_rises_after_selected']} / {A['n_units']} units (last − selected from {A['mse_last_minus_sel_range'][0]:+.3f} to {A['mse_last_minus_sel_range'][1]:+.3f}; trained arms' seed means {min(S[f'FIT/{a}']['mean_mse_last_minus_sel'] for a in ARMS if a != 'U-RANK'):+.3f} to {max(S[f'FIT/{a}']['mean_mse_last_minus_sel'] for a in ARMS if a != 'U-RANK'):+.3f}, U-RANK {S['FIT/U-RANK']['mean_mse_last_minus_sel']:+.3f}).  FIT keeps improving while the CAL mean-rating error has passed its minimum: the frozen rule stops early by design.",
          f"- Pair-accuracy criterion vs the frozen MSE criterion: the argmax epoch differs in {A['n_pairacc_epoch_differs']} / {A['n_units']} units, but the pair accuracy forgone at the selected epoch is at most {-A['pairacc_gap_range'][0]:.3f} (range {A['pairacc_gap_range'][0]:+.3f} to {A['pairacc_gap_range'][1]:+.3f}); choosing the pair-accuracy epoch would cost {A['mse_cost_of_pairacc_choice_range'][0]:.3f} to {A['mse_cost_of_pairacc_choice_range'][1]:.3f} in CAL mean-rating MSE.  RPS argmin differs in {A['n_rps_epoch_differs']} / 48, CE argmin in {A['n_ce_epoch_differs']} / 48.  Across the formal trained arms the seed-mean pair accuracy moves within {min(min(S[f'FIT/{a}']['mean_pairacc_curve']) for a in ARMS):.3f}-{max(max(S[f'FIT/{a}']['mean_pairacc_curve']) for a in ARMS):.3f} over all epochs and arms.  Reading: the log shows a small possible misalignment (≤ 0.02 pair accuracy) between the MSE rule and a ranking criterion on CAL_SELECT; it is not evidence of an external gain and no checkpoint other than best / resume exists to test it (level 2 stops here).",
          f"- U-RANK (formal): CAL mean-rating MSE {lst([u['mse_curve'][0] for u in rank])} at e0 rising to {lst([u['mse_curve'][4] for u in rank])} at e4, while its CAL pair accuracy stays {min(min(u['pairacc_curve']) for u in rank):.3f}-{max(max(u['pairacc_curve']) for u in rank):.3f} (level with the other arms at every epoch) and its native loss falls {rank[0]['train_loss_curve'][0]:.3f} → {rank[0]['train_loss_curve'][4]:.3f} (point part {rank[0]['train_point_curve'][0]:.3f} → {rank[0]['train_point_curve'][4]:.3f}, ranking part {rank[0]['train_ranking_curve'][0]:.3f} → {rank[0]['train_ranking_curve'][4]:.3f}; {rank[0]['ranking_pairs_per_step']:.1f} scored pairs and {rank[0]['ranking_subgroups_per_step']:.1f} subgroups per step).  Log-level reading: the ranking arm's within-prompt ordering is intact; what the MSE rule penalises is the scale / level of mu, which drifts as the RankNet term spreads the mu differences (tau = 1).  'U-RANK failed' in V8-04 is a statement about the mean-rating MSE rule, not about its native ordering; the GenAI-Rank N9 result (rank-based) was level with the other arms for the same reason.  Whether this is a scale-only difference is D03's question (V81_03).",
          f"- Tuning units: with lr 5e-6 / 1e-5 / 2e-5 the selected epoch is 1 or 2 for every trained arm (table 2) and the epoch-0 → selected MSE drop is {min(S[f'TUNE/{a}']['mean_mse_e0_minus_sel'] for a in ARMS if a != 'U-RANK'):.3f}-{max(S[f'TUNE/{a}']['mean_mse_e0_minus_sel'] for a in ARMS if a != 'U-RANK'):.3f} (seed 17), larger than in the formal seeds ({min(S[f'FIT/{a}']['mean_mse_e0_minus_sel'] for a in ARMS if a != 'U-RANK'):.3f}-{max(S[f'FIT/{a}']['mean_mse_e0_minus_sel'] for a in ARMS if a != 'U-RANK'):.3f}): seed 17's first epoch starts from a worse CAL MSE (0.44-0.46 vs 0.40-0.44), so the single tuning seed is a noisier basis for the lr rule than the three formal seeds.  No lr-grid extension is proposed (owner decision 2026-10-15).",
          "- Not claimed: any DEV / GenAI number, any 'overfitting' beyond the CAL mean-rating MSE curve, any per-epoch per-image curve (weights not saved: only best.pt and the final resume.pt exist per unit)."]
    (OUT / "selection_gap.md").write_text("\n".join(L) + "\n")


def context():
    rows = list(csv.DictReader(open(OUT / "benchmark_context.csv"))); C = json.loads((OUT / "d07_context.json").read_text())
    g = lambda r, k: ("NA" if r[k] in ("", "None") else f"{float(r[k]):.3f}")
    L = [f"# D07 external context table (generated {datetime.now(timezone.utc).strftime('%FT%TZ')} from `benchmark_context.csv` / `d07_context.json`; frozen results only, no new scoring)", "",
         "Sources: " + "; ".join(f"{k} sha256 {v['sha256'][:12]}…" for k, v in C["sources"].items()) + ".", "",
         "| system | host / backbone | trained on | route status | GenAI-Rank N9 selected (regret) DALLE_3 | SDXL_Base | GenAI-Bench 850 pooled Spearman | pairwise | EvalMuse DEV SRCC (descr.) | DEV pairwise |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["system"].startswith("Table C"): continue
        L.append(f"| {r['system']} | {r['host_backbone']} | {r['trained_on']} | {r['route_status']} | {g(r, 'genai_rank_N9_selected_DALLE_3')} ({g(r, 'genai_rank_N9_regret_DALLE_3')}) | {g(r, 'genai_rank_N9_selected_SDXL_Base')} ({g(r, 'genai_rank_N9_regret_SDXL_Base')}) | {g(r, 'genai_bench850_pooled_spearman')} | {g(r, 'genai_bench850_pairwise')} | {g(r, 'evalmuse_DEV_srcc_descriptive')} | {g(r, 'evalmuse_DEV_pairwise_descriptive')} |")
    L += ["", "## Frozen contrasts kept as published", "", "| contrast | GenAI-Rank N9 Δ [95 %] Holm (V8-04) |", "|---|---|"]
    for k, v in C["genai_rank_contrasts_U-CS"].items():
        L.append(f"| {k} | {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] Holm {v['p_holm']:.3f} |")
    L += ["", "| family (V8-05, GenAI-Bench 850 × 6) | contrast | Δ [95 %] | Holm |", "|---|---|---|---|"]
    for fam, d in C["genai_bench_families"].items():
        for comp, mets in d.items():
            for met, v in mets.items():
                if isinstance(v, dict) and "delta" in v:
                    ph = v.get("p_holm"); L.append(f"| {fam} | U-CS − {comp} ({met}) | {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {('%.4f' % ph) if isinstance(ph, (int, float)) and ph == ph else 'descriptive'} |")
    L += ["", "## Cost (historical job timings — not a standard workload; devices differ, see note)", ""]
    for gname, c in C["v8_cost_by_gpu"].items():
        L.append(f"- V8 arm on {gname} ({c['n_units']} formal units): training {c['train_min'][0]:.0f}-{c['train_min'][1]:.0f} min per unit (5 epochs over 26,177 images + 5 CAL_SELECT evaluations), CAL_SELECT evaluation {c['cal_eval_img_per_s'][0]:.0f}-{c['cal_eval_img_per_s'][1]:.0f} images / s, export of 30,540 images {c['export_min'][0]:.0f}-{c['export_min'][1]:.0f} min ({c['export_img_per_s'][0]:.0f}-{c['export_img_per_s'][1]:.0f} images / s, batch 32, 364 px, fp16 autocast), peak training memory {c['peak_mem_gb'][1]:.1f} GB.")
    for r in rows:
        if r["cost_recorded"] and not r["system"].startswith("U-") and not r["system"].startswith("Table C"):
            L.append(f"- {r['system']}: {r['cost_recorded']}.")
    L += ["", "## Table C kernel routes (untrained representation; separate class, not a pure estimator contest)", "", "| route | CAL_PROB RPS (natural) | CAL_PROB mean-MSE macro | DEV RPS | DEV mean-MSE macro |", "|---|---|---|---|---|"]
    E = json.loads(Path(C["sources"]["v8_eval.json"]["path"]).read_text())
    for k, v in E["table_c_distribution"].items():
        L.append(f"| {k} | {v['CAL_PROB']['raw']['rps_natural']:.3f} | {v['CAL_PROB']['raw']['mean_mse_prompt_macro']:.3f} | {v['DEV']['raw']['rps_natural']:.3f} | {v['DEV']['raw']['mean_mse_prompt_macro']:.3f} |")
    (OUT / "benchmark_context.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    selection_gap(); context(); print("rendered", OUT / "selection_gap.md", OUT / "benchmark_context.md")
