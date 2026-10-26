"""PMI readout analysis (PROTOCOL.json): 24 fixed 30k checkpoints, same-job 2x2 COS/PMI coarse/local readouts on DEV."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent; SEEDS = (1001, 1002, 1003); ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
POL = ('COS__COS', 'PMI__COS', 'COS__PMI', 'PMI__PMI'); S = {}; F = {}
for ds in ('bf', 'epfl'):
    for a in ARMS:
        for s in SEEDS:
            d = HERE / 'runs' / f'{ds}_{a}_{s}'; c = json.loads((d / 'COMPLETE.json').read_text())
            assert c['status'] == 'COMPLETE' and not c['smoke'] and c['weights_unchanged']
            S[(ds, a, s)] = c['summaries']
            rows = [json.loads(l) for l in open(d / 'ROWS.jsonl')]
            F[(ds, a, s)] = {p: dict(fail9=sum(r['policies'][p]['err_full'] >= 9 for r in rows),
                                     energy_prefers_wrong=sum(r['policies'][p]['err_full'] >= 9 and r['policies'][p]['F_hat'] >= r['policies'][p]['F_true'] for r in rows)) for p in POL}
def d(ds, a, p, q='COS__COS', k='success9', b=None):
    return [(S[(ds, a, s)][p][k] - S[(ds, b or a, s)][q][k]) * 100 for s in SEEDS]
res = {}
for ds in ('bf', 'epfl'):
    r = {}
    for a in ARMS:
        x9, x42 = d(ds, a, 'PMI__PMI'), d(ds, a, 'PMI__PMI', k='success42')
        r[a] = dict(pmi_minus_cos_strict=[round(v, 3) for v in x9], strict_mean=round(st.mean(x9), 3), positive=sum(v > 0 for v in x9),
                    loose_mean=round(st.mean(x42), 3), signal=st.mean(x9) >= 1 and sum(v > 0 for v in x9) >= 2 and st.mean(x42) >= -.5,
                    coarse_only_PMI__COS=round(st.mean(d(ds, a, 'PMI__COS')), 3), local_only_COS__PMI=round(st.mean(d(ds, a, 'COS__PMI')), 3),
                    means={p: round(st.mean(S[(ds, a, s)][p]['success9'] for s in SEEDS) * 100, 3) for p in POL},
                    fail9={p: sum(F[(ds, a, s)][p]['fail9'] for s in SEEDS) for p in POL},
                    energy_prefers_wrong={p: sum(F[(ds, a, s)][p]['energy_prefers_wrong'] for s in SEEDS) for p in POL})
    calibrated = [a for a in ('vcs_exact', 'vcs_perm', 'bce') if r[a]['signal']]
    r['theory_pattern'] = dict(calibrated_arms_with_signal=calibrated,
                               nce_gain=r['nce']['strict_mean'], gap_vs_nce={a: round(r[a]['strict_mean'] - r['nce']['strict_mean'], 3) for a in ('vcs_exact', 'vcs_perm', 'bce')})
    r['vcs_exact_PMI_minus_bce_PMI'] = [round(v, 3) for v in d(ds, 'vcs_exact', 'PMI__PMI', 'PMI__PMI', b='bce')]
    r['vcs_exact_PMI_minus_nce_COS'] = [round(v, 3) for v in d(ds, 'vcs_exact', 'PMI__PMI', 'COS__COS', b='nce')]
    res[ds] = r
cost = sum(json.loads((HERE / 'runs' / f'{ds}_{a}_{s}' / 'COMPLETE.json').read_text())['seconds'] for ds in ('bf', 'epfl') for a in ARMS for s in SEEDS) / 3600
(HERE / 'ANALYSIS.json').write_text(json.dumps(dict(status='COMPLETE', results=res, receipt_gpu_hours=round(cost, 3)), indent=1))
for ds, r in res.items():
    print('==', ds)
    for a in ARMS: print(f"  {a:9s} PMI-COS strict {r[a]['strict_mean']:+6.2f} ({r[a]['positive']}/3) {r[a]['pmi_minus_cos_strict']} loose {r[a]['loose_mean']:+5.2f} | coarse-only {r[a]['coarse_only_PMI__COS']:+5.2f} local-only {r[a]['local_only_COS__PMI']:+5.2f} | means {r[a]['means']} {'SIGNAL' if r[a]['signal'] else ''}")
    print('  theory', r['theory_pattern']); print('  vcsPMI-bcePMI', r['vcs_exact_PMI_minus_bce_PMI'], ' vcsPMI-nceCOS', r['vcs_exact_PMI_minus_nce_COS'])
    print('  fail/energy-prefers-wrong vcs_exact', r['vcs_exact']['fail9'], r['vcs_exact']['energy_prefers_wrong'])
print('GPUh', round(cost, 2))
