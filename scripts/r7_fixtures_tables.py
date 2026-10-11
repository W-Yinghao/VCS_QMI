"""r7 R7-20 — assemble the fixture results into report tables and the two package CSVs (measured numbers only).
Reads outputs/R7_fixtures/d1/d1_summary.json, outputs/R7_fixtures/d2/{pope,mhal}_summary.json, outputs/R7_D3/d3_finite.json (each optional).
Writes outputs/R7_fixtures/r7_fixture_tables.md, reports/R7/baseline_reproduction.csv, reports/R7/compute_budget.csv.
Paper numbers were read from the papers (sources inline); they are on the full official splits, our fixtures are small subsets.
    python scripts/r7_fixtures_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/infres/yinwang/CS_QMI"); OUT = ROOT / "outputs" / "R7_fixtures"; REP = ROOT / "ssl_pilot" / "reports" / "R7"
SVLM = "arXiv 2410.04417 (SparseVLM)"
# (source, GQA testdev-bal acc, TextVQA val soft acc, POPE F1 over the 3 COCO splits)
PAPER = {"full": [("LLaVA MODEL_ZOO (llava-v1.5-7b)", 62.0, 58.2, 85.9), (SVLM + " v1 Tab.1 vanilla", 61.9, 58.2, 85.9), (SVLM + " v4 Tab.1 vanilla", 61.9, 58.3, 85.9)],
         "sparsevlm192": [(SVLM + " v1 Tab.1", 57.6, 56.1, 83.6), (SVLM + " v4 (ICML 2025) Tab.1", 59.5, 57.8, 85.3)],
         "sparsevlm64": [(SVLM + " v1 Tab.1", 52.7, 51.8, 75.1), (SVLM + " v4 (ICML 2025) Tab.1", 53.8, 53.4, 77.5)],
         "qwen25vl7b": [("arXiv 2502.13923 v1 (Qwen2.5-VL report) Tab.; GQA / POPE not reported", None, 84.9, None)]}
CODE = {"qwen": ("transformers 4.57.6 Qwen2_5_VLForConditionalGeneration (Qwen/Qwen2.5-VL-7B-Instruct@cc594898)", "transformers==4.57.6"),
        "full": ("https://github.com/haotian-liu/LLaVA (as shipped in Gumpest/SparseVLMs)", "a9e71427220c4cacb6afb5a96d23cfb43142e0c3"),
        "sparsevlm": ("https://github.com/Gumpest/SparseVLMs", "a9e71427220c4cacb6afb5a96d23cfb43142e0c3")}
HOST_REV = "liuhaotian/llava-v1.5-7b@4481d270; vision openai/clip-vit-large-patch14-336@ce19dc91"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def f1(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(c, nd=1):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def d1_tables(S, md, rows, budget):
    md.append("## D1 — host and SparseVLM on the fixed fixtures\n")
    md.append("| config | GQA val-bal acc (n=200) | TextVQA val soft acc (n=200) | POPE-adv acc / F1 / yes-ratio (n=300) | equal visual tokens | GPU |")
    md.append("|---|---|---|---|---|---|")
    for t in S["configs"]:
        m, run = S["metrics"].get(t, {}), S["runs"][t]
        g, tv, p = m.get("gqa"), m.get("textvqa"), m.get("pope")
        eq = ("576" if t.startswith("full") else f"{f1(S['timing'][t]['gqa']['visual_tokens']['mean'], 0)} (GQA mean, dynamic)" if "visual_tokens" in S["timing"][t]["gqa"]
              else f1(run["fixtures"].get("gqa", {}).get("equal_visual_tokens")))
        md.append(f"| {t} | {f1(g['value'])} {ci(g['ci95_image_bootstrap'])} | {f1(tv['value'])} {ci(tv['ci95_image_bootstrap'])} | "
                  f"{f1(p['accuracy'])} / {f1(p['f1'])} / {f1(p['yes_ratio'])} | {eq} | {run['env'].get('gpu', '?')} |")
    md.append("\nPaper references (full official splits: GQA testdev-balanced 12,578 q, TextVQA val 5,000 q, POPE F1 over random / popular / adversarial):\n")
    md.append("| config | source | GQA | TextVQA | POPE F1 |"); md.append("|---|---|---|---|---|")
    for t in S["configs"]:
        for src, g, tv, p in PAPER.get(t, []):
            md.append(f"| {t} | {src} | {g if g is not None else '—'} | {tv if tv is not None else '—'} | {p if p is not None else '—'} |")
    md.append("\nPaired change vs the full host on the same items (points, image-cluster bootstrap 95 %):\n")
    md.append("| config | GQA | TextVQA | POPE-adv acc | answers changed (GQA / TextVQA / POPE) |"); md.append("|---|---|---|---|---|")
    for t, v in S["paired_vs_full"].items():
        md.append(f"| {t} | {f1(v['gqa']['delta_points'])} {ci(v['gqa']['ci95_image_bootstrap'])} | {f1(v['textvqa']['delta_points'])} "
                  f"{ci(v['textvqa']['ci95_image_bootstrap'])} | {f1(v['pope']['delta_points'])} {ci(v['pope']['ci95_image_bootstrap'])} | "
                  f"{v['gqa']['answer_changed_fraction']:.2f} / {v['textvqa']['answer_changed_fraction']:.2f} / {v['pope']['answer_changed_fraction']:.2f} |")
    md.append("\nTiming split, ms per item (mean; CUDA-synchronised; batch 1, fp16, greedy):\n")
    md.append("| config | fixture | vision | selector | prefill (excl. selector) | decode (steps) | total wall | prompt tokens |"); md.append("|---|---|---|---|---|---|---|---|")
    for t in S["configs"]:
        for name, tm in S["timing"][t].items():
            md.append(f"| {t} | {name} | {f1(tm['vision_ms']['mean'])} | {f1(tm['selector_ms']['mean'], 2)} | {f1(tm['prefill_ms']['mean'])} | "
                      f"{f1(tm['decode_ms']['mean'])} ({f1(tm['decode_steps']['mean'])}) | {f1(tm['wall_ms']['mean'])} | {f1(tm['prompt_tokens']['mean'], 0)} |")
    pairs = [("full", "full_h100nvl", "full host, L40S vs H100 NVL"), ("sparsevlm192", None, "SparseVLM-192, L40S (untimed first run) vs H100 NVL")]
    for x, y, label in pairs:
        fa = OUT / "d1" / f"pred_{x}.jsonl"; fb = OUT / "d1" / (f"pred_{y}.jsonl" if y else "run1_untimed_pred_sparsevlm192.jsonl")
        if fa.exists() and fb.exists():
            A = {(r["fixture"], r["qid"]): r for r in map(json.loads, fa.read_text().splitlines())}
            B = [r for r in map(json.loads, fb.read_text().splitlines()) if (r["fixture"], r["qid"]) in A]
            md.append(f"\nRepeatability ({label}): identical answers on {np.mean([A[(r['fixture'], r['qid'])]['text'] == r['text'] for r in B]):.3f} of {len(B)} items; "
                      f"max |ΔP(yes)| = {max(abs(A[(r['fixture'], r['qid'])]['p_yes'] - r['p_yes']) for r in B):.3f}.")
    md.append("\nFinite-model check on binary items (C = label, b = P_full(yes), a = P_pruned(yes); h = (C−a)² − (C−b)², d = (b−a)², h − d = 2(C−b)(b−a)):\n")
    md.append("| config | fixture | n | mean h [95 %] | mean d [95 %] | mean residual [95 %] | P(h<0) | P(h>d) | corr(h,d) | Brier full → pruned | identity err |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, v in S["finite_risk"].items():
        for name, r in v.items():
            md.append(f"| {t} | {name} | {r['n']} | {r['mean_h']:.4f} {ci(r['mean_h_ci95'], 4)} | {r['mean_d']:.4f} {ci(r['mean_d_ci95'], 4)} | "
                      f"{r['mean_residual']:.4f} {ci(r['mean_residual_ci95'], 4)} | {r['frac_h_negative']:.2f} | {r['frac_h_exceeds_d']:.2f} | "
                      f"{f1(r['corr_h_d'], 2)} | {r['brier_full']:.4f} → {r['brier_pruned']:.4f} | {r['identity_max_abs_error']:.1e} |")
    md.append("")
    fx = {"gqa": 1, "textvqa": 2, "pope": 3}
    for t in S["configs"]:
        m, run = S["metrics"][t], S["runs"][t]; kind = {"full": "full", "full_h100nvl": "full", "qwen25vl7b": "qwen"}.get(t, "sparsevlm"); papers = PAPER.get(t, [])
        for name, col in fx.items():
            if name not in m or t not in PAPER:                                        # timing-only reruns are not reproduction rows
                continue
            ours = m[name]; val = ours["f1"] if name == "pope" else ours["value"]
            rows.append({"method_id": f"D1-{t}-{name}", "paper_id": "; ".join(p[0] for p in papers),
                         "code_url": CODE[kind][0], "commit": CODE[kind][1],
                         "host_revision": HOST_REV if kind != "qwen" else "Qwen/Qwen2.5-VL-7B-Instruct@cc594898 (bf16, SDPA, min/max pixels 256·28² / 1280·28²)",
                         "paper_training_data": "LLaVA-1.5 mix665k (no retraining; training-free pruning)" if kind != "qwen" else "Qwen2.5-VL pre-training + SFT (not public)",
                         "our_training_data": "none (inference only)",
                         "official_protocol": {"gqa": "LLaVA eval: testdev-balanced, short-answer prompt, convert_gqa_for_eval + GQA eval.py exact match",
                                               "textvqa": "LLaVA eval: val v051_ocr prompt, m4c TextVQAAccuracyEvaluator",
                                               "pope": "LLaVA eval: llava_pope_test, eval_pope.py parsing, F1 on random/popular/adversarial"}[name],
                         "our_changes": {"gqa": "200 val-balanced q (100 images, hash-fixed) instead of testdev; lmms-lab parquet",
                                         "textvqa": "200 val q (hash-fixed, grouped by image); OCR tokens from lmms-lab ocr_tokens (', '-joined)",
                                         "pope": "300 adversarial q (50 images × 6, hash-fixed); F1 on adversarial only"}[name]
                                        + ("; batch 1 fp16 greedy max_new_tokens 128; CLIP tower loaded from local copy" if kind != "qwen" else
                                           "; Qwen: question + short-answer suffix, no OCR tokens; bf16 greedy (shipped repetition penalty 1.05) max_new_tokens 128"),
                         "evaluator_commit": f"SparseVLMs a9e7142 ({'convert_gqa_for_eval.py' if name == 'gqa' else 'm4c_evaluator.py' if name == 'textvqa' else 'eval_pope.py'})",
                         "fixture_pass": "yes" if run["fixtures"].get(name, {}).get("items") else "no",
                         "paper_metric": " / ".join(f"{p[col]}" if p[col] is not None else "not reported" for p in papers) or "none",
                         "our_reproduction_metric": f"{val:.1f} (95% {ci(ours['ci95_image_bootstrap'])} acc)" if name != "pope" else f"F1 {val:.1f}; acc {ours['accuracy']:.1f} (95% {ci(ours['ci95_image_bootstrap'])})",
                         "difference_explanation": {"gqa": ("val-balanced vs testdev, n=200 (SE ≈ 3 pts); GQA val images come from Visual Genome, which LLaVA-1.5 "
                                                            "trained on (GQA train + VG), so val is likely exposed (atlas DS01 note): not a like-for-like reproduction")
                                                           if kind != "qwen" else "val-balanced, n=200; GQA not reported for Qwen2.5-VL; exact match penalises phrase-level answers",
                                                    "textvqa": "same split, n=200 subset (SE ≈ 3.3 pts)",
                                                    "pope": "adversarial split only (hardest), n=300; paper pools three splits"}[name],
                         "status": "FIXTURE_REPRODUCED_SUBSET"})
        sec = sum(run["fixtures"][n]["wall_seconds"] for n in run["fixtures"]); items = sum(run["fixtures"][n]["items"] for n in run["fixtures"])
        mem = max(run["fixtures"][n]["peak_mem_gb"] for n in run["fixtures"]); per = sec / items
        if kind == "qwen":
            vt = S["timing"][t]; isz = ("dynamic resolution (256–1280 × 28² px); mean visual tokens GQA %.0f / TextVQA %.0f / POPE %.0f"
                                        % tuple(vt[n]["visual_tokens"]["mean"] for n in ("gqa", "textvqa", "pope")))
        else:
            isz = "336 px, 576 visual tokens" + (f" -> SparseVLM {run['retain']}" if run.get("retain") else "")
        load = f"{run['load_seconds']:.0f} s model load" + (" (staged to node-local disk first)" if any("staged" in x for x in run.get("stage_notes") or []) else " (direct NFS mmap)")
        budget.append({"direction": "D1", "stage": f"D1-00 fixture inference ({t})", "hardware": run["env"].get("gpu", "?"), "precision": "bf16" if kind == "qwen" else "fp16",
                       "batch": 1, "input_size": isz,
                       "observed_seconds": f"{sec:.0f} for {items} items ({per:.3f} s/item) + {load}",
                       "observed_peak_memory": f"{mem:.1f} GiB", "scope": "200 GQA val-bal + 200 TextVQA val + 300 POPE-adv",
                       "proposed_units": "one configuration on GQA testdev-bal (12,578) + TextVQA val (5,000)",
                       "estimated_total_from_measurement": f"{per * 17578 / 3600:.2f} GPU-h per configuration (inference only)",
                       "authorization": "R7-20 fixture (coordinator directive)", "status": "MEASURED"})


def d2_tables(P, M, md, rows, budget):
    md.append("## D2 — POPE host fixture, text-only control, M-HalDetect\n")
    if P:
        md.append("| condition | acc [95 %] | precision | recall | F1 | yes-ratio | AUROC of P(yes) | ms/item |"); md.append("|---|---|---|---|---|---|---|---|")
        for c, v in P["conditions"].items():
            md.append(f"| {c} | {f1(v['accuracy'])} {ci(v['accuracy_ci95_image_bootstrap'])} | {f1(v['precision'])} | {f1(v['recall'])} | {f1(v['f1'])} | "
                      f"{f1(v['yes_ratio'])} | {f1(v['auroc_p_yes'], 3)} | {f1(v['timing_ms_mean']['wall_ms'])} |")
        f1p, f2p = OUT / "d1" / "pred_full.jsonl", OUT / "d2" / "pope_preds.jsonl"
        if f1p.exists() and f2p.exists():                                              # same host, same items, two processes / jobs
            a1 = {r["qid"]: r for r in map(json.loads, f1p.read_text().splitlines()) if r["fixture"] == "pope"}
            a2 = [r for r in map(json.loads, f2p.read_text().splitlines()) if r["condition"] == "image" and r["qid"] in a1]
            if a2:
                md.append(f"\nRepeatability: the D1 full-host job and the D2 job gave identical POPE answers on {np.mean([a1[r['qid']]['text'] == r['text'] for r in a2]):.3f} "
                          f"of {len(a2)} items (max |ΔP(yes)| = {max(abs(a1[r['qid']]['p_yes'] - r['p_yes']) for r in a2):.2e}).")
        md.append(f"\nCOCO-image overlap of the POPE fixture with M-HalDetect (train ∪ val): {P['mhal_image_overlap']['also_in_mhal_detect']} of {P['mhal_image_overlap']['pope_fixture_images']} images.\n")
        im = P["conditions"]["image"]
        rows.append({"method_id": "D2-host-pope-adversarial", "paper_id": "LLaVA-1.5 (arXiv 2310.03744) POPE F1 on all three splits",
                     "code_url": CODE["full"][0], "commit": CODE["full"][1], "host_revision": HOST_REV, "paper_training_data": "LLaVA-1.5 mix665k",
                     "our_training_data": "none (inference only)", "official_protocol": "llava_pope_test + eval_pope.py; random / popular / adversarial",
                     "our_changes": "300 adversarial q (50 images × 6, hash-fixed); + text-only control (no image token)",
                     "evaluator_commit": "SparseVLMs a9e7142 (eval_pope.py logic)", "fixture_pass": "yes", "paper_metric": "85.9 (F1, 3 splits)",
                     "our_reproduction_metric": f"F1 {im['f1']:.1f}; acc {im['accuracy']:.1f}; text-only acc {P['conditions']['text_only']['accuracy']:.1f}",
                     "difference_explanation": "adversarial only, n=300", "status": "FIXTURE_REPRODUCED_SUBSET"})
        budget.append({"direction": "D2", "stage": "D2 POPE fixture (image + text-only)", "hardware": P["run"]["env"].get("gpu", "?"), "precision": "fp16", "batch": 1,
                       "input_size": "336 px, 576 visual tokens", "observed_seconds": f"{P['run']['wall_seconds_image']:.0f} (image) + {P['run']['wall_seconds_text_only']:.0f} (text-only) for 300 items each",
                       "observed_peak_memory": f"{P['run']['peak_mem_gb']:.1f} GiB", "scope": "300 POPE-adversarial items × 2 conditions",
                       "proposed_units": "POPE 3 configurations × 3,000 q, host + VCD + verifier arms",
                       "estimated_total_from_measurement": f"{P['run']['wall_seconds_image'] / 300 * 9000 / 3600:.2f} GPU-h per arm (inference only)",
                       "authorization": "R7-20 fixture (coordinator directive)", "status": "MEASURED"})
    if M:
        st = M["shortcut_train_only"]; md.append("M-HalDetect schema (official splits): " + "; ".join(
            f"{k}: {v['entries']} responses, {v['segments']} segments, {v['images']} images, labels {v['labels']}" for k, v in M["schema"].items()))
        md.append(f"\nOfficial train/val image overlap: {M['official_split_image_overlap']['shared']} shared images.\n")
        md.append("Pure-text shortcut check (official train only, 5 image-grouped folds; out-of-fold AUROC, fold sd):\n")
        md.append("| task | n | positive rate | n-gram logistic | position-only | length-only |"); md.append("|---|---|---|---|---|---|")
        for task, v in st.items():
            md.append(f"| {task} | {v['n']} | {v['positive_rate']:.3f} | {v['ngram_logistic']['auroc_oof']:.3f} ± {v['ngram_logistic']['auroc_fold_sd']:.3f} | "
                      f"{v['position_only']['auroc_oof']:.3f} | {v['length_only']['auroc_oof']:.3f} |")
        md.append(f"\nTop n-gram cues for INACCURATE: {', '.join(st['inaccurate_vs_accurate']['top_cues_positive'][:12])}\n")
        budget.append({"direction": "D2", "stage": "M-HalDetect loader + text-only shortcut CV", "hardware": "CPU (8 cores)", "precision": "float64", "batch": "full",
                       "input_size": "segment text", "observed_seconds": f"{M['seconds']:.0f}", "observed_peak_memory": "n/a", "scope": "official train (grouped CV) + schema of val",
                       "proposed_units": "verifier training needs the COCO val2014 images (steward: extraction in progress)", "estimated_total_from_measurement": "n/a (CPU only)",
                       "authorization": "R7-20 fixture", "status": "MEASURED"})


def d3_tables(D, md, budget):
    md.append("## D3 — finite conditional distributions (exact S; 16 conditions; 3 draws)\n")
    cells = D["cells"]; kinds = [(k, l) for k, ls in D["design"]["levels"].items() for l in ls]; ns = D["design"]["n_per_condition"]
    def agg(k, l, n, est, key):
        v = [c["estimators"][est][key] for c in cells if c["kind"] == k and abs(c["lambda"] - l) < 1e-9 and c["n_per_condition"] == n]
        return float(np.mean(v)), float(np.std(v))
    md.append("Posterior MSE S − J(T) = E_M(T − η)² (population, exact), mean over draws; lower is better; the null critic scores S:\n")
    md.append("| distribution | S | z-marginal L1 shift | " + " | ".join(f"n={n} VCS / logistic / kernel" for n in ns) + " |")
    md.append("|---|---|---|" + "---|" * len(ns))
    for k, l in kinds:
        key = f"{k}:{l:.3f}"; S = D["design"]["S_true"][key]
        md.append(f"| {k} λ={l:.2f} | {S:.4f} | {D['design']['z_marginal_L1_shift'][key]:.3f} | " +
                  " | ".join(" / ".join(f"{agg(k, l, n, e, 'posterior_mse')[0]:.4f}" for e in ("vcs", "logistic", "kernel")) for n in ns) + " |")
    md.append("\nHeld-out estimate error Ĵ_EVAL(T) − S (mean ± sd over draws; 512 EVAL samples per condition per side):\n")
    md.append("| distribution | " + " | ".join(f"n={n} VCS / logistic / kernel" for n in ns) + " |"); md.append("|---|" + "---|" * len(ns))
    for k, l in kinds:
        S = D["design"]["S_true"][f"{k}:{l:.3f}"]
        md.append(f"| {k} λ={l:.2f} | " + " | ".join(" / ".join(f"{agg(k, l, n, e, 'J_eval')[0] - S:+.3f}±{agg(k, l, n, e, 'J_eval')[1]:.3f}" for e in ("vcs", "logistic", "kernel")) for n in ns) + " |")
    md.append("\nPer-condition view (16 conditions): mean over conditions of the population posterior MSE S_c − J_c(T), mean over draws:\n")
    md.append("| distribution | " + " | ".join(f"n={n} VCS / logistic / kernel" for n in ns) + " |"); md.append("|---|" + "---|" * len(ns))
    for k, l in kinds:
        md.append(f"| {k} λ={l:.2f} | " + " | ".join(" / ".join(f"{agg(k, l, n, e, 'per_condition_posterior_mse_mean')[0]:.4f}" for e in ("vcs", "logistic", "kernel")) for n in ns) + " |")
    md.append("\nDetection statistics (not S estimates; mean over draws; clean row = null level):\n")
    md.append("| distribution | " + " | ".join(f"n={n} grouped cond. MMD² / uncond. MMD²" for n in ns) + " |"); md.append("|---|" + "---|" * len(ns))
    for k, l in kinds:
        vals = []
        for n in ns:
            cs = [c["discrimination"] for c in cells if c["kind"] == k and abs(c["lambda"] - l) < 1e-9 and c["n_per_condition"] == n]
            vals.append(f"{np.mean([c['grouped_conditional_mmd2'] for c in cs]):.4f} / {np.mean([c['unconditional_mmd2'] for c in cs]):.4f}")
        md.append(f"| {k} λ={l:.2f} | " + " | ".join(vals) + " |")
    md.append("\nCells where the learned critic is worse than the null critic T ≡ 0 (posterior MSE > S), out of 12 shifted distributions × draws;"
              " and mean relative posterior MSE (S − J)/S over the shifted distributions:\n")
    md.append("| n | VCS | logistic | kernel |"); md.append("|---|---|---|---|")
    for n in ns:
        sh = [c for c in cells if c["kind"] != "clean" and c["n_per_condition"] == n]
        md.append(f"| {n} | " + " | ".join(f"{sum(c['estimators'][e]['posterior_mse'] > c['S_true'] for c in sh)}/{len(sh)}; "
                                           f"{np.mean([c['estimators'][e]['posterior_mse'] / c['S_true'] for c in sh]):.2f}" for e in ("vcs", "logistic", "kernel")) + " |")
    secs = {e: float(np.mean([c["estimators"][e]["seconds"] for c in cells])) for e in ("vcs", "logistic", "kernel")}
    budget.append({"direction": "D3", "stage": "D3 finite conditional check", "hardware": "CPU (8 workers)", "precision": "float64", "batch": "full",
                   "input_size": "16 conditions × 96 states", "observed_seconds": f"{D['seconds']:.0f} for {len(cells)} cells (per fit: VCS {secs['vcs']:.1f} s, logistic {secs['logistic']:.1f} s, kernel {secs['kernel']:.1f} s)",
                   "observed_peak_memory": "n/a", "scope": "13 distributions × n {8,32,128} × 3 draws", "proposed_units": "natural-image D3 needs generators + encoders (not measured)",
                   "estimated_total_from_measurement": "finite grid is CPU-minutes", "authorization": "R7-20 fixture", "status": "MEASURED"})


def main() -> int:
    md = ["# R7-20 fixture tables (generated by scripts/r7_fixtures_tables.py)\n"]; rows, budget = [], []
    S = load(OUT / "d1" / "d1_summary.json"); P = load(OUT / "d2" / "pope_summary.json"); M = load(OUT / "d2" / "mhal_summary.json")
    D = load(ROOT / "outputs" / "R7_D3" / "d3_finite.json")
    if S:
        d1_tables(S, md, rows, budget)
    if P or M:
        d2_tables(P, M, md, rows, budget)
    if D:
        d3_tables(D, md, budget)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "r7_fixture_tables.md").write_text("\n".join(md) + "\n")
    for name, data in (("baseline_reproduction.csv", rows), ("compute_budget.csv", budget)):
        hdr = (ROOT / "ssl_pilot" / "r7" / "package_r7_20261011" / "templates" / name).read_text(encoding="utf-8-sig").splitlines()[0].split(",")
        with open(REP / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader(); [w.writerow({k: r.get(k, "") for k in hdr}) for r in data]
    print("\n".join(md)); print(f"[r7_tables] {len(rows)} reproduction rows, {len(budget)} budget rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
