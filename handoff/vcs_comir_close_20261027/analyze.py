"""R24 CoMIR closing analysis: cross-modal-only NCE vs full NCE / VCS / BCE, readout table, FIT diagnostics.
Prints WAITING for any block whose units are not all COMPLETE; writes ANALYSIS.json."""
import importlib.util
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('r24_base_analyze', BASE / 'handoff/vcs_comir_20261026/analyze.py')
BA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(BA)
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/vcs_comir_close_20261027')
SEEDS = BA.SEEDS
ARMS = ('nce_mse_t2.0_lr0.01', 'nce_cos_t0.1_lr0.01', 'vcs_exp_t0.5_lr0.01', 'vcs_cos_lr0.003', 'bce_exp_t0.5_lr0.01', 'bce_cos_lr0.003')
X = {'nce_mse_x_t2.0_lr0.01': 'nce_mse_t2.0_lr0.01', 'nce_cos_x_t0.1_lr0.01': 'nce_cos_t0.1_lr0.01'}
KEYS = ('success9', 'success42', 'median_err', 'p95_err')


def complete(d):
    p = d / 'COMPLETE.json'
    return json.loads(p.read_text()) if p.exists() else None


def mean_of(by_seed):
    return {k: statistics.fmean(by_seed[s][k] for s in SEEDS) for k in KEYS}


def pair(ds, a, b):
    """a, b: (metrics by seed, records by seed)."""
    return dict(strict=BA.contrast(a[0], b[0], 'success9'), loose=BA.contrast(a[0], b[0], 'success42'),
                median=BA.contrast(a[0], b[0], 'median_err'), p95=BA.contrast(a[0], b[0], 'p95_err'),
                strict_scene_bootstrap=BA.core_bootstrap(ds, a[1], b[1], 9), loose_scene_bootstrap=BA.core_bootstrap(ds, a[1], b[1], 42))


def base_block(ds, name):
    fn = BA.scene_fn(ds); m, r = {}, {}
    for s in SEEDS:
        _, recs = BA.load(ds, name, s); r[s] = recs; m[s] = BA.metrics(recs, fn)
    return m, r


def exp1(ds):
    fn = BA.scene_fn(ds); missing = [f'{n}_s{s}' for n in X for s in SEEDS if not complete(OUT / 'train' / f'{ds}_{n}_s{s}')]
    if missing:
        return dict(status='WAITING', missing=missing)
    blocks, cost, init_same = {}, 0.0, []
    for n in X:
        m, r = {}, {}
        for s in SEEDS:
            d = OUT / 'train' / f'{ds}_{n}_s{s}'; c = complete(d); cost += c['seconds']; init_same.append(c['initial_equals_full_nce'])
            r[s] = BA.rows(d / 'DEV.jsonl'); m[s] = BA.metrics(r[s], fn)
        blocks[n] = (m, r)
    for n in ARMS:
        blocks[n] = base_block(ds, n)
    pairs = {}
    for xn, full in X.items():
        pairs[f'{xn} - {full} (cross-modal only - full NCE)'] = pair(ds, blocks[xn], blocks[full])
        for other in ('vcs_exp_t0.5_lr0.01', 'vcs_cos_lr0.003', 'bce_exp_t0.5_lr0.01', 'bce_cos_lr0.003'):
            pairs[f'{other} - {xn}'] = pair(ds, blocks[other], blocks[xn])
    return dict(status='COMPLETE', per_seed={n: blocks[n][0] for n in blocks}, means={n: mean_of(blocks[n][0]) for n in blocks},
                pairs=pairs, initial_models_equal_full_nce=all(init_same), gpu_hours=cost / 3600)


def exp2(ds):
    fn = BA.scene_fn(ds); missing = [f'{n}_s{s}' for n in ARMS for s in SEEDS if not complete(OUT / 'readout' / f'{ds}_{n}_s{s}')]
    if missing:
        return dict(status='WAITING', missing=missing)
    table, recs, cost, reeval = {}, {}, 0.0, {}
    for n in ARMS:
        bm, br = base_block(ds, n)
        table[n] = {'mse': bm}; recs[n] = {'mse': br}
        for ro in ('ncc', 'cos'):
            table[n][ro], recs[n][ro] = {}, {}
        for s in SEEDS:
            d = OUT / 'readout' / f'{ds}_{n}_s{s}'; c = complete(d); cost += c['seconds']
            for ro in ('ncc', 'cos'):
                r = BA.rows(d / f'DEV_{ro}.jsonl'); recs[n][ro][s] = r; table[n][ro][s] = BA.metrics(r, fn)
            if (d / 'DEV_mse.jsonl').exists():
                again = BA.metrics(BA.rows(d / 'DEV_mse.jsonl'), fn)
                reeval[f'{n}_s{s}'] = {k: again[k] - bm[s][k] for k in KEYS}
    means = {n: {ro: mean_of(table[n][ro]) for ro in table[n]} for n in ARMS}
    best_ro = {n: max(('ncc', 'cos'), key=lambda ro: means[n][ro]['success9']) for n in ARMS}
    pairs = {}
    for n in ARMS:
        for ro in ('ncc', 'cos'):
            pairs[f'{n}: {ro} - mse'] = pair(ds, (table[n][ro], recs[n][ro]), (table[n]['mse'], recs[n]['mse']))
    nce = ('nce_mse_t2.0_lr0.01', 'nce_cos_t0.1_lr0.01')
    best_nce_mse = max(nce, key=lambda n: means[n]['mse']['success9'])
    best_nce_any = max(((n, ro) for n in nce for ro in table[n]), key=lambda x: means[x[0]][x[1]]['success9'])
    for fam in ('vcs', 'bce'):
        arms = [n for n in ARMS if n.startswith(fam)]
        best = max(((n, ro) for n in arms for ro in table[n]), key=lambda x: means[x[0]][x[1]]['success9'])
        pairs[f'best {fam.upper()} any readout ({best[0]}:{best[1]}) - best NCE mse ({best_nce_mse})'] = pair(
            ds, (table[best[0]][best[1]], recs[best[0]][best[1]]), (table[best_nce_mse]['mse'], recs[best_nce_mse]['mse']))
        pairs[f'best {fam.upper()} any readout ({best[0]}:{best[1]}) - best NCE any readout ({best_nce_any[0]}:{best_nce_any[1]})'] = pair(
            ds, (table[best[0]][best[1]], recs[best[0]][best[1]]), (table[best_nce_any[0]][best_nce_any[1]], recs[best_nce_any[0]][best_nce_any[1]]))
    out = dict(status='COMPLETE', per_seed=table, means=means, best_non_mse_readout=best_ro, pairs=pairs, gpu_hours=cost / 3600,
               mse_reused_from='base DEV.jsonl of the same checkpoint (separate search)')
    if reeval:
        out['mse_reevaluation_minus_base'] = dict(per_unit=reeval, mean={k: statistics.fmean(v[k] for v in reeval.values()) for k in KEYS},
                                                  mean_abs={k: statistics.fmean(abs(v[k]) for v in reeval.values()) for k in KEYS})
    return out


def exp3(ds):
    p = OUT / 'diag' / f'DIAG_{ds}.json'
    if not p.exists():
        return dict(status='WAITING')
    d = json.loads(p.read_text()); rows = {}
    for n in ARMS:
        units = [d['results'][f'{n}_s{s}'] for s in SEEDS]
        row = {}
        for phase in ('step0', 'final'):
            row[phase] = dict(ratio_main_over_reg_median=statistics.fmean(u[phase]['ratio_main_over_reg']['q50'] for u in units),
                              main_grad_norm_median=statistics.fmean(u[phase]['main_grad_norm']['q50'] for u in units),
                              reg_grad_norm_median=statistics.fmean(u[phase]['reg_grad_norm']['q50'] for u in units),
                              cos_main_reg_median=statistics.fmean(u[phase]['cos_main_reg']['q50'] for u in units))
            for k in ('pos_x', 'neg_x', 'vcs_pos_factor', 'vcs_neg_factor', 'T_pos', 'T_neg'):
                if k in units[0][phase]:
                    row[phase][k] = {q: statistics.fmean(u[phase][k][q] for u in units) for q in ('q05', 'q50', 'q95', 'mean')}
            if 'pos_x_fraction_below' in units[0][phase]:
                row[phase]['pos_x_fraction_below'] = {q: statistics.fmean(u[phase]['pos_x_fraction_below'][q] for u in units)
                                                      for q in units[0][phase]['pos_x_fraction_below']}
        rows[n] = row
    return dict(status='COMPLETE', seed_mean=rows, gpu_hours=d['seconds'] / 3600, gpu=d['gpu'])


def main():
    out = {ds: dict(cross_modal_nce=exp1(ds), readout=exp2(ds), diagnostics=exp3(ds)) for ds in ('bf', 'epfl')}
    out['gpu_hours_total'] = sum(out[ds][k].get('gpu_hours', 0) for ds in ('bf', 'epfl') for k in ('cross_modal_nce', 'readout', 'diagnostics'))
    (HERE / 'ANALYSIS.json').write_text(json.dumps(out, indent=1) + '\n')
    for ds in ('bf', 'epfl'):
        print('==', ds)
        for k, v in out[ds].items():
            print(' ', k, v['status'])
            if v['status'] != 'COMPLETE':
                continue
            if k == 'cross_modal_nce':
                for n, m in v['means'].items():
                    print(f"    {n:28s} strict {m['success9']:6.2f} loose {m['success42']:6.2f} median {m['median_err']:6.2f} p95 {m['p95_err']:7.2f}")
                for n, p in v['pairs'].items():
                    print(f"    {n:70s} strict {p['strict']['mean']:+6.2f} ({p['strict']['positive']}/3) loose {p['loose']['mean']:+6.2f} CI {p['strict_scene_bootstrap']['ci95']}")
            elif k == 'readout':
                for n, r in v['means'].items():
                    print(f"    {n:22s} " + '  '.join(f"{ro} {m['success9']:5.1f}/{m['success42']:5.1f}/{m['median_err']:5.2f}/{m['p95_err']:6.1f}" for ro, m in r.items()))
                for n, p in v['pairs'].items():
                    print(f"    {n:90s} strict {p['strict']['mean']:+6.2f} ({p['strict']['positive']}/3) loose {p['loose']['mean']:+6.2f}")
                if 'mse_reevaluation_minus_base' in v:
                    print('    mse re-eval - base (mean abs)', v['mse_reevaluation_minus_base']['mean_abs'])
            else:
                for n, r in v['seed_mean'].items():
                    print(f"    {n:22s} ratio step0 {r['step0']['ratio_main_over_reg_median']:9.3g} final {r['final']['ratio_main_over_reg_median']:9.3g}"
                          + (f"  pos d/tau q50 {r['final']['pos_x']['q50']:.3g} vcs_pos_factor q50 {r['final']['vcs_pos_factor']['q50']:.3g}" if 'pos_x' in r['final'] else ''))
    print('GPU h', out['gpu_hours_total'])


if __name__ == '__main__':
    main()
