#!/usr/bin/env python3
"""Write the 36 NR2 unit configs from the package MATRIX.csv. CONFIRM = the NR1 seed-1001 locked G config with only
seed/unit_id changed. REFERENCE = NR1 G recipe fields + NR2 query/reference sampler fields."""
import csv, json, sys
from pathlib import Path
N = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104'); NR1 = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/locked_configs')
rows = list(csv.DictReader(open(N/'package/configs/MATRIX.csv')))
assert len(rows) == 36 and len({r['unit_id'] for r in rows}) == 36
out = N/'locked_configs'; out.mkdir(exist_ok=True)
for r in rows:
    base = json.loads((NR1/f"NR1_{r['condition']}_G_{r['objective']}_s1001.json").read_text())
    assert base['architecture'] == 'G' and r['architecture'] == 'G' and int(r['steps']) == base['steps'] == 8000
    if r['block'] == 'CONFIRM':
        c = dict(base); c['seed'] = int(r['seed']); c['unit_id'] = r['unit_id']
        assert r['protocol'] == base['protocol_id']
        diff = {k for k in set(c) | set(base) if c.get(k) != base.get(k)}
        assert diff == {'seed', 'unit_id'}, diff
    else:
        c = {k: v for k, v in base.items() if k not in ('batch_size', 'pair_chunk', 'sampler', 'q_estimator')}
        c.update(protocol_id=r['protocol'], unit_id=r['unit_id'], seed=int(r['seed']), reference=r['reference'],
                 alpha=float(r['alpha']), hard_k=int(r['hard_k']), queries_per_direction=32, reference_draws_per_query=63,
                 positive_records_per_step=int(r['positive_records_per_step']), negative_pairs_per_step=int(r['negative_pairs_per_step']),
                 sampler='two_direction_iid_observed_query_plus_iid_reference_v1', directions_weight=[.5, .5],
                 proposal='alpha*p_marginal + (1-alpha)*Uniform(base FIT cosine Top100(query)); exact mixture probability',
                 inference_score='s (never s-log w)', importance_self_normalization=False, importance_clipping=False,
                 nce_multipositive='observed-only (complete observed map)')
        assert c['positive_records_per_step'] == 64 and c['negative_pairs_per_step'] == 4032 == 64*63
    (out/f"{r['unit_id']}.json").write_text(json.dumps(c, indent=2) + '\n')
print('wrote', len(rows))
