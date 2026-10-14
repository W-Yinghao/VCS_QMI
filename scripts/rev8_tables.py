"""REV8 report tables (formatting only): every number comes from reports/REV8/REV8_DIAG_RESULTS/rev8_diag_results.json and the unit lists of
the frozen protocol.  Main units: mean ± sd over seeds 0-2 per (scale, objective); learning-rate comparison: the seed-0 units at both rates.
    python scripts/rev8_tables.py RESULTS_JSON > tables.md"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

SCALES = (("llava10k", "LLaVA 10k"), ("llavafull", "LLaVA FULL"), ("qwen10k", "Qwen 10k"))
OBJ = ("vcs", "logistic", "direct")


def ms(v, nd=3):
    v = [x for x in v if x is not None]
    return "—" if not v else (f"{v[0]:.{nd}f}" if len(v) == 1 else f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}")


def main():
    R = json.loads(Path(sys.argv[1]).read_text()); U = R["units"]; P = yaml.safe_load(Path(R["protocol"]).read_text())["rev8_diag"]["units"]
    def units(scale, obj, which):
        tags = [t for t in P[scale][which] if t.startswith(obj + "_")]
        return [U[f"{scale}__{t}"] for t in tags if f"{scale}__{t}" in U]
    out = ["## A (R8-04): what the CAL score measures, and the within-question mask choice\n",
           "| scale | objective | pooled ρ | within-budget ρ | within (budget, mech.) ρ | within-question ρ | pairwise acc. (q, k) | top-1 hit | regret: critic | random choice | always grid | teacher d | seeds with acc. CI ∋ 0.5 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, name in SCALES:
        for o in OBJ:
            us = units(sc, o, "main")
            if not us:
                continue
            A = [u["A_R8_04"] for u in us]; q = [a["within_qk"] for a in A]
            out.append(f"| {name} | {o} | {ms([a['pooled_spearman_avg'] for a in A])} | {ms([np.mean([v for v in a['within_budget_spearman'].values() if v is not None]) for a in A])} | "
                       f"{ms([a['within_budget_mechanism_spearman']['mean'] for a in A])} | {ms([a['within_question_spearman_mean']['mean'] for a in A])} | "
                       f"{ms([x['pairwise_accuracy'][0] for x in q])} | {ms([x['top1_hit_rate'][0] for x in q])} | {ms([x['regret']['critic'][0] for x in q])} | "
                       f"{ms([x['regret']['random_choice_expected'][0] for x in q])} | {ms([x['regret']['always_grid'][0] for x in q])} | {ms([x['regret']['teacher_d_choice'][0] for x in q])} | "
                       f"{sum(u['flags']['R8_04_within_qk_not_better_than_chance'] for u in us)} / {len(us)} |")
    out += ["\n## B (R8-02): the soft–hard gap decomposed (critic in eval mode, τ_end 0.1, smallest and 25 % budgets)\n",
            "| scale | objective | units | budget | HD | SD − HD (relaxation) | HN − HD (noise, hard) | SN − HN (relaxation, noisy) | SN − HD (legacy) | logged | soft mass outside hard set (SN) | Σm − k (SN) | dominant |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, name in SCALES:
        for o in OBJ:
            for which, lab in (("main", "main (3 seeds)"), ("lr_alt_seed0", "other lr, seed 0")):
                us = units(sc, o, which)
                if not us:
                    continue
                keys = list(us[0]["B_R8_02"]["by_budget"])[:2]; lr = us[0]["meta"]["lr"]
                for k in keys:
                    e = [u["B_R8_02"]["by_budget"][k] for u in us]
                    out.append(f"| {name} | {o} | {lab}, lr {lr:g} | {k} | {ms([x['means']['HD'] for x in e])} | {ms([x['relax_det'][0] for x in e])} | "
                               f"{ms([x['noise_hard'][0] for x in e])} | {ms([x['relax_noise'][0] for x in e])} | {ms([x['legacy_SN_minus_HD'][0] for x in e])} | "
                               f"{ms([x.get('logged_gap') for x in e])} | {ms([x['mask']['outside_frac_end_SN'] for x in e])} | {ms([x['mask']['mass_minus_k_end_SN'] for x in e], 2)} | "
                               f"{', '.join(sorted(set(u['flags']['R8_02_dominant'] for u in us)))} |")
    out += ["\n## C (R8-01): dropout in phase B (R = 8 train-mode draws; 25 % budget)\n",
            "| scale | objective | eval readout | train readout | distortion ratio | rank agreement R 8 / R 4 | gate top-k Jaccard | phase-B loss: train mode | eval mode (SN) | var T_A | self-gap 2 var T_B |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, name in SCALES:
        for o in OBJ:
            us = units(sc, o, "main")
            if not us:
                continue
            k = list(us[0]["C_R8_01"]["by_budget"])[1]; e = [u["C_R8_01"]["by_budget"][k] for u in us]
            out.append(f"| {name} | {o} | {ms([x['mean_eval_readout'] for x in e])} | {ms([x['mean_train_readout'] for x in e])} | {ms([x['distortion_ratio'] for x in e])} | "
                       f"{ms([x['rank_agreement_R'] for x in e])} / {ms([x['rank_agreement_half'] for x in e])} | {ms([x['gate_topk_jaccard_train_vs_eval'] for x in e])} | "
                       f"{ms([x['phaseB_loss_train_mode'] for x in e])} | {ms([x['phaseB_loss_eval_mode_SN'] for x in e])} | {ms([x.get('var_TA_mean') for x in e], 4)} | {ms([x.get('self_gap_2varTB') for x in e], 4)} |")
    fl = {}
    for key, u in U.items():
        for f, v in u["flags"].items():
            fl.setdefault(f, {}).setdefault(str(v), []).append(key)
    out += ["\n## Flags (pre-registered, descriptive)\n"] + [f"- {f}: " + "; ".join(f"{v} × {len(ks)}" for v, ks in sorted(d.items())) for f, d in fl.items()]
    g = [u["meta"]["guard"] for u in U.values()]
    out.append(f"\nGuard: {len(g)} units; max |recomputed − logged| pooled CAL Spearman = {max(abs(x['rho'] - x['logged']) for x in g):.2e}.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
