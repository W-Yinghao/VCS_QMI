"""Per student seed (1001 CR4 PC, 1002/1003 CR5 PC, 2002/2003 CR5P1105 PC): J_KEEP_TOP1 minus OBS / J_MASS_OBS / BASE / BCE / NCE (rSum, PC consumer),
per dataset, no cross-dataset averaging; actual exposure by replaying the frozen sampler (CPU, no model) and checking the training receipts' edge hashes."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/site'); sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/code/src')
import cr1_prod as C1                                                       # noqa: E402
TREE = Path('/home/infres/yinwang/CS_QMI/handoff/cr5p_20261105'); rj = lambda p: json.loads(Path(p).read_text())
CR4 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104'); CR5 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5/CR5_20261105'); P = Path('/projects/EEG-foundation-model/yinghao/outputs_cr5p/CR5P1105')
CONDS = ('Flickr60', 'CC152K_RECOVERED_20261104'); RELS = ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'J_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1')
SEEDS = {1001: ('CR4', 20261104201), 1002: ('CR5', 20261104202), 1003: ('CR5', 20261104203), 2002: ('CR5P1105', 202611052002), 2003: ('CR5P1105', 202611052003)}
fz4 = rj('/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/PROTOCOL_FROZEN_CR4.json')
gp = lambda c, r: next(i['graph_path'] for i in fz4['units'].values() if i['condition'] == c and i['relation'] == r)
def cdir(c, r, s):
    d = {1001: CR4/'students'/c/f'CR4_{c}_{r}_COMPLETE_PC_s1001', 1002: CR5/'students'/c/f'CR5_{c}_{r}_COMPLETE_PC_s1002', 1003: CR5/'students'/c/f'CR5_{c}_{r}_COMPLETE_PC_s1003',
         2002: P/'students'/c/f'CR5P1105_{c}_{r}_COMPLETE_PC_t2002', 2003: P/'students'/c/f'CR5P1105_{c}_{r}_COMPLETE_PC_t2003'}[s]
    uc = d/'UNIT_COMPLETE.json'
    return Path(rj(uc)['reused_dir']) if uc.exists() and rj(uc).get('reused_dir') else d
def rsum(c, r, s):
    p = cdir(c, r, s)/'eval_008000'/'METRICS.json'; return rj(p)['metrics']['rSum'] if p.exists() else None
rows, exp = [], []
for c in CONDS:
    for s, (src, ss) in SEEDS.items():
        v = {r: rsum(c, r, s) for r in RELS}; row = dict(condition=c, student_seed=s, source_round=src, J_KEEP_TOP1=v['J_KEEP_TOP1'])
        for r in ('OBS', 'J_MASS_OBS', 'BASE_KEEP_TOP1', 'BCE_KEEP_TOP1', 'NCE_KEEP_TOP1'):
            row[f'J_minus_{r}'] = (v['J_KEEP_TOP1']-v[r]) if (v['J_KEEP_TOP1'] is not None and v[r] is not None) else None
        row.update({f'rSum_{r}': v[r] for r in RELS}); rows.append(row)
        for r in RELS:
            d = cdir(c, r, s)
            if not (d/'DONE.json').exists(): continue
            D = rj(d/'DONE.json'); rec_hash = D.get('same_draw_receipt', {}).get('edge_hits_sha256')
            rel = C1.ExposedRelation(load_npz(gp(c, r)).tocsr()); rng = np.random.default_rng(ss); hits = np.zeros(len(rel.edge_mass), np.int64)
            iq = np.zeros(rel.matrix.shape[1], np.int64); tq = np.zeros(rel.matrix.shape[0], np.int64)
            for _ in range(8000):
                b = rel.batch_edges(rng, 64); np.add.at(hits, b['edges'], 1); iq[b['images']] += 1; tq[b['texts']] += 1
            h = hashlib.sha256(hits.astype(np.int64).tobytes()).hexdigest(); pi = iq/iq.sum()
            exp.append(dict(condition=c, relation=r, student_seed=s, sampler_seed=ss, replay_matches_training_receipt=(h == rec_hash) if rec_hash else 'no_receipt_hash',
                            distinct_edges=int((hits > 0).sum()), graph_edges=int(len(hits)), distinct_edge_fraction=float((hits > 0).mean()), max_edge_repeats=int(hits.max()),
                            distinct_images_queried=int((iq > 0).sum()), distinct_texts_queried=int((tq > 0).sum()),
                            image_query_entropy_nats=float(-(pi[pi > 0]*np.log(pi[pi > 0])).sum()), max_image_query_count=int(iq.max())))
for n, rr in (('J_DIFFS_PER_STUDENT_SEED.csv', rows), ('EXPOSURE_PER_CELL.csv', exp)):
    k = []; [k.append(x) for r in rr for x in r if x not in k]
    with open(TREE/n, 'w', newline='') as f: w = csv.DictWriter(f, k); w.writeheader(); w.writerows(rr)
s2 = lambda x: '' if x is None else f'{x:+.2f}'; f2 = lambda x: '' if x is None else f'{x:.2f}'
L = ['# CR5_P1105 + CR4/CR5: J_KEEP_TOP1 increments per student seed (COMPLETE_PC consumer; numbers only, no cross-dataset averaging)', '',
     '| condition | student seed | round | J rSum | J−OBS | J−J_MASS_OBS | J−BASE | J−BCE | J−NCE |', '|---|---|---|---|---|---|---|---|---|']
for r in rows: L.append(f"| {r['condition']} | {r['student_seed']} | {r['source_round']} | {f2(r['J_KEEP_TOP1'])} | {s2(r['J_minus_OBS'])} | {s2(r['J_minus_J_MASS_OBS'])} | {s2(r['J_minus_BASE_KEEP_TOP1'])} | {s2(r['J_minus_BCE_KEEP_TOP1'])} | {s2(r['J_minus_NCE_KEEP_TOP1'])} |")
L += ['', '## Actual exposure (replayed frozen sampler; checked against training edge-hash receipts)', '', '| condition | graph | seed | receipt match | distinct edges / graph edges | max repeats | images queried | image-query entropy |', '|---|---|---|---|---|---|---|---|']
for e in exp: L.append(f"| {e['condition']} | {e['relation']} | {e['student_seed']} | {e['replay_matches_training_receipt']} | {e['distinct_edges']}/{e['graph_edges']} | {e['max_edge_repeats']} | {e['distinct_images_queried']} | {e['image_query_entropy_nats']:.3f} |")
(TREE/'RESULTS_CR5P.md').write_text('\n'.join(L)+'\n'); print('\n'.join(L))
