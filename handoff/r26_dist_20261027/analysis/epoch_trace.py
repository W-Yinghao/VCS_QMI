"""Review 20261029 §6.1/6.2: compact per-epoch table for the 48 completed Flickr runs (R25 H0/V1/B1 + R26 five arms) and
the seed-divergence timeline. Reads existing EPOCHS.json only (no retraining, no TEST). Writes EPOCH_TRACE.csv and SEED_DIVERGENCE.json."""
import json, csv, os
import numpy as np
R25 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/groups'
R26 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs'
OUT = os.path.dirname(os.path.abspath(__file__))
cells = [f'f{n}_s{s}' for n in (40, 60) for s in (1001, 1002, 1003)]
runs = {}
for c in cells:
    for a in ('H0', 'V1', 'B1'):
        runs[(c, 'R25_' + a)] = f'{R25}/{c}/{a}/EPOCHS.json'
    for a in ('d_vcs', 'd_bce', 'm_vcs', 'n_vcs', 'd_snce'):
        runs[(c, 'R26_' + a)] = f'{R26}/{c}/{a}/EPOCHS.json'
rows = []
prefix = {}
for (c, a), p in runs.items():
    d = json.load(open(p))
    if c not in prefix:
        prefix[c] = {'warmup_rsum': [e['dev']['rsum'] for e in d['warmup']],
                     'cotrain0': {k: d['cotrain'][0][k] for k in ('split_A', 'split_B')} | {'rsum': d['cotrain'][0]['dev']['rsum']}}
    for e in d['cotrain']:
        r = {'cell': c, 'arm': a, 'epoch': e['epoch'], 'rsum': e['dev']['rsum'], 'peer0': e['dev']['peer0_rsum'], 'peer1': e['dev']['peer1_rsum'],
             'i2t_r1': e['dev']['i2t_r1'], 't2i_r1': e['dev']['t2i_r1'], 'lr': e.get('lr')}
        for p_ in ('A', 'B'):
            s = e.get('split_' + p_, {})
            r[f'split{p_}_n'] = s.get('n_pred_clean'); r[f'split{p_}_prec'] = s.get('precision'); r[f'split{p_}_rec'] = s.get('recall')
        rows.append(r)
with open(os.path.join(OUT, 'EPOCH_TRACE.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# seed timeline per noise level: mean over the 8 arms of each seed, by epoch
tl = {}
for n in (40, 60):
    for s in (1001, 1002, 1003):
        c = f'f{n}_s{s}'
        sub = [r for r in rows if r['cell'] == c]
        tl[c] = {k: [float(np.mean([r[k] for r in sub if r['epoch'] == ep])) for ep in range(40)] for k in ('rsum', 'splitA_prec', 'splitB_prec', 'splitA_rec', 'splitB_rec', 'splitA_n', 'splitB_n')}
        tl[c]['arm_spread_best'] = [min(max(r['rsum'] for r in sub if r['arm'] == a) for a in {r['arm'] for r in sub}),
                                    max(max(r['rsum'] for r in sub if r['arm'] == a) for a in {r['arm'] for r in sub})]
json.dump({'note': 'means over 8 arms per seed; epoch 0 is the shared native prefix epoch; injected-noise truth used only for split precision/recall diagnostics already logged by the host',
           'prefix': prefix, 'timeline': tl}, open(os.path.join(OUT, 'SEED_DIVERGENCE.json'), 'w'), indent=1)
for n in (40, 60):
    print(f'== {n}%')
    for s in (1001, 1002, 1003):
        c = f'f{n}_s{s}'; p = prefix[c]; t = tl[c]
        print(c, 'warmup rsum', [round(x, 1) for x in p['warmup_rsum']], 'ep0', round(p['cotrain0']['rsum'], 1),
              'ep0 splitA prec/rec %.3f/%.3f' % (p['cotrain0']['split_A']['precision'], p['cotrain0']['split_A']['recall']), 'best range', [round(x, 1) for x in t['arm_spread_best']])
        print('   mean rsum ep 0,1,2,5,10,20,39:', [round(t['rsum'][e], 1) for e in (0, 1, 2, 5, 10, 20, 39)])
        print('   splitA prec ep 0,1,2,5,10,20,39:', [round(t['splitA_prec'][e], 3) for e in (0, 1, 2, 5, 10, 20, 39)], ' rec', [round(t['splitA_rec'][e], 3) for e in (0, 5, 39)], ' n', [int(t['splitA_n'][e]) for e in (0, 5, 39)])
