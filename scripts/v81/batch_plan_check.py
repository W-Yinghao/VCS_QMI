"""V8.1 D06 supplement (added 2026-10-16 after the V8.1 report check) — batch-plan checks for the ranking arms (package docs/03 §5): does one prompt span several subgroups in a batch, and does the ranking
normaliser match the frozen V8-03 rule.  Uses only the FIT image -> prompt map (no new labels); plan = v8_core.batch_manifest (epoch 0)."""
import json, sys
from collections import defaultdict
from statistics import mean
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/ssl_pilot_v8/scripts/v8'); sys.path.insert(0, '/home/infres/yinwang/CS_QMI/ssl_pilot_v8/scripts/v7')
import v8_core as C, v8_labels as LB
rows = LB.load('/home/infres/yinwang/CS_QMI/outputs/V7/data'); R = LB.records(rows)
fit = R['FIT']; prompt_of = [r['prompt_id'] for r in fit]
pi = defaultdict(list)
for k, r in enumerate(fit): pi[r['prompt_id']].append(k)
nsub = {p: len(C.split_sizes(len(v))) for p, v in pi.items()}
out = {'n_fit_images': len(fit), 'n_fit_prompts': len(pi), 'prompts_with_more_than_one_subgroup': sum(1 for v in nsub.values() if v > 1),
       'max_images_per_prompt': max(len(v) for v in pi.values()), 'seeds': {}}
for seed in (0, 1, 2):
    plan = C.batch_manifest(dict(pi), seed, 0)
    def stats(batches):
        S = []
        for b in batches:
            subs = defaultdict(list)
            for i, g, ns in b: subs[g].append(i)
            scored = {g: v for g, v in subs.items() if len(v) >= 2}
            prom = defaultdict(set)
            for g, v in subs.items():
                ps = {prompt_of[i] for i in v}; assert len(ps) == 1, 'a subgroup mixes prompts'
                prom[ps.pop()].add(g)
            den = sum(1.0 / next(ns for i, gg, ns in b if gg == g) for g in scored)
            S.append({'images': len(b), 'subgroups_scored': len(scored), 'pairs': sum(len(v) * (len(v) - 1) // 2 for v in scored.values()),
                      'prompts': len(prom), 'prompts_with_2plus_subgroups_in_batch': sum(1 for s in prom.values() if len(s) > 1),
                      'scored_subgroups_from_split_prompts': sum(1 for g in scored if next(ns for i, gg, ns in b if gg == g) > 1), 'normaliser_den': den})
        return S
    first, allb = stats(plan[:20]), stats(plan)
    out['seeds'][seed] = {
        'first20': {k: mean(s[k] for s in first) for k in first[0]},
        'first20_batches_with_a_prompt_in_2plus_subgroups': sum(1 for s in first if s['prompts_with_2plus_subgroups_in_batch'] > 0),
        'epoch0_batches': len(allb), 'epoch0_batches_with_a_prompt_in_2plus_subgroups': sum(1 for s in allb if s['prompts_with_2plus_subgroups_in_batch'] > 0),
        'epoch0_mean_pairs': mean(s['pairs'] for s in allb), 'epoch0_mean_scored_subgroups': mean(s['subgroups_scored'] for s in allb),
        'epoch0_share_scored_subgroups_from_split_prompts': sum(s['scored_subgroups_from_split_prompts'] for s in allb) / sum(s['subgroups_scored'] for s in allb)}
from pathlib import Path
RES = Path(__file__).resolve().parents[2] / 'reports' / 'V8_1_DIAGNOSTICS' / 'results'
G = json.load(open(RES / 'gradient_checks.json'))
out['gradient_checks_logged'] = {r['tag']: {'ranking_pairs': r['ranking_pairs'], 'ranking_subgroups': r['ranking_subgroups']} for r in G['rows'] if r['tag'].endswith('__best') and r.get('ranking_pairs') is not None}
out['script'] = 'scripts/v81/batch_plan_check.py'; out['status'] = 'development_descriptive_no_labels'
out['images_per_prompt'] = sorted(__import__('collections').Counter(len(v) for v in pi.values()).items())
dst = RES / 'batch_plan_check.json'
if dst.exists():
    raise SystemExit(f'refusing to overwrite {dst}')
dst.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8'); print(json.dumps(out['seeds'][0], indent=1))
