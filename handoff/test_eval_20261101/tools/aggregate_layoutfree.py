"""Build TEST_RESULTS_LAYOUTFREE.json and TEST_TABLES_LAYOUTFREE.md from TEST_RESULTS.json (native, existing) and
results_layoutfree/*.json (tools/test_eval_layoutfree.py). Record arithmetic only (no model / data access), numbers only.
Hosts 'sps' and 'nrcs' have no batch-layout dependence (see NOT_AFFECTED) and are listed with layout-free = native.
"""
import os, json, datetime, statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
SEEDS = (1001, 1002, 1003)
VARS = ('native', 'global', 'decoupled')
GATE_OK = ('PASS', 'PASS_LOGGING_RULE')
LINE_ORDER = ['R25', 'R25-SPS', 'R26', 'R27', 'R27-D', 'R27-O', 'R27-O3', 'R27-CS', 'R27-Q', 'R27-A1X', 'R28', 'R29', 'R30',
              'R31-FID', 'R32', 'NRCS-v1', 'R33-BASE', 'R37S-PARENT']
NOT_AFFECTED = {
    'sps': 'SPS/SGRAF eval_code_sps/model.py SGRAF.forward_emb returns img_enc(images), txt_enc(captions) with no batch-mean '
           'subtraction; val_start puts img_enc/txt_enc/sim_enc (all BatchNorm1d) in eval mode -> per-item embeddings and scores, '
           'no dependence on batch composition. Layout-free = native; TEST not re-read.',
    'nrcs': 'NRCS-v1 r32_native_ratio_20261101/nrcs/test_entry.py Nrcs._scores -> gpo_adapter.encode_pool (eval mode, frozen BN, '
            '"no batch centering"; calls EncoderImage/EncoderText directly, not VSE.forward_emb) -> critic.retrieval_embeddings '
            '(pointwise heads). Layout-free = native; TEST not re-read.',
}
CONDS = [('f30k', 0), ('f30k', 20), ('f30k', 40), ('f30k', 60), ('f30k', 80), ('coco', 20), ('coco', 40), ('coco', 60), ('coco', 80),
         ('cc152k', 'real')]
COND_NAME = {('f30k', n): f'Flickr30K, {n}% noise' for n in (0, 20, 40, 60, 80)}
COND_NAME.update({('coco', n): f'MS-COCO, {n}% noise' for n in (20, 40, 60, 80)}); COND_NAME[('cc152k', 'real')] = 'CC152K (real noise)'


def f2(x):
    return 'NA' if x is None else f'{x:.2f}'


def tmain(v, which='main'):
    if v is None:
        return None
    if 'test' in v:
        return v['test']
    return v['test_5fold1k']['mean'] if which == 'main' else v['test_5k']


def host_of(u):
    return 'nrcs' if u['line'] == 'NRCS-v1' else 'sps' if u['line'] == 'R25-SPS' else 'in2r'


def main():
    tr = json.load(open(os.path.join(HERE, 'TEST_RESULTS.json')))
    rows = []
    for u in tr['units']:
        host = host_of(u)
        row = dict(unit_id=u['unit_id'], line=u['line'], arm_label=u['arm_label'], dataset=u['dataset'], noise=u['noise'], seed=u['seed'],
                   oracle_diagnostic=u['oracle_diagnostic'], native_in2r_reproduction=u['native_in2r_reproduction'], host=host,
                   checkpoint_sha256=u['checkpoint_sha256'], dev_selected_epoch=u['dev_selected_epoch'],
                   recorded_dev_gate=u.get('dev_gate'), recorded_eval_gpu=(u.get('eval_env') or {}).get('gpu'))
        nat = dict(dev=u.get('dev_recomputed'))
        if 'test' in u:
            nat['test'] = u['test']
        elif 'test_5k' in u:
            nat.update(test_5fold1k=u['test_5fold1k'], test_5k=u['test_5k'])
        row['recorded_native'] = nat
        if host != 'in2r':
            row.update(layout='NOT_AFFECTED', not_affected_evidence=NOT_AFFECTED[host], status='NOT_AFFECTED',
                       readouts={k: nat for k in VARS}, dev_gate=u.get('dev_gate'))
        else:
            p = os.path.join(HERE, 'results_layoutfree', u['checkpoint_sha256'][:16] + '.json')
            r = json.load(open(p)) if os.path.exists(p) else None
            if r is None or r.get('status') != 'COMPLETE':
                row.update(layout='AFFECTED', status='PENDING' if r is None else r['status'])
            else:
                nn = r['native']
                natnow = dict(dev=nn['dev_recomputed'])
                natnow.update({k: nn[k] for k in ('test', 'test_5fold1k', 'test_5k') if k in nn})
                row.update(layout='AFFECTED', status='COMPLETE', dev_gate=r['dev_gate'], dev_max_abs_recall_diff_pp=r['dev_max_abs_recall_diff_pp'],
                           test_native_match=r['test_native_match'], test_native_vs_recorded_max_abs=r['test_native_vs_recorded_max_abs'],
                           native_rebuilt_minus_native_max_abs=r['native_rebuilt_minus_native_max_abs'],
                           readouts={'native': natnow, 'global': r['global'], 'decoupled': r['decoupled']}, native_rebuilt=r['native_rebuilt'],
                           eval_env=r['env'], result_file=os.path.relpath(p, HERE))
                # the recorded gate verdict (TEST_RESULTS.json, re-derived with the 1e-9 epsilon) is kept; a unit enters the
                # tables only if both the recorded and the re-run gates pass
                row['in_tables'] = r['dev_gate'] == 'PASS' and u.get('dev_gate') in GATE_OK
        if host != 'in2r':
            row['in_tables'] = u.get('dev_gate') in GATE_OK
        rows.append(row)
    lit = json.load(open(os.path.join(HERE, 'LITERATURE_TABLE.json')))
    out = dict(generated_utc=datetime.datetime.utcnow().isoformat() + 'Z', source_native='TEST_RESULTS.json', variants={
        'native': 'author IN2R evaluation: per sequential loader batch (128 rows; row r = image r//5, caption r) mean subtraction + l2norm',
        'global': 'no per-batch centring; one mean per modality over the whole split (COCO 5-fold: each fold), then l2norm',
        'decoupled': 'images and captions in independent fixed-seed random batches (numpy default_rng 44 / 4444, same batch size), '
                  'per-batch centring as native'}, not_affected=NOT_AFFECTED,
        n_units=len(rows), n_in2r=sum(r['host'] == 'in2r' for r in rows), n_in2r_complete=sum(r['host'] == 'in2r' and r['status'] == 'COMPLETE' for r in rows),
        n_test_native_match=sum(bool(r.get('test_native_match')) for r in rows), units=rows)
    json.dump(out, open(os.path.join(HERE, 'TEST_RESULTS_LAYOUTFREE.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'TEST_TABLES_LAYOUTFREE.md'), 'w').write(tables(rows, lit, out))
    print(f"{out['n_units']} units; in2r {out['n_in2r']} ({out['n_in2r_complete']} re-read); native TEST match {out['n_test_native_match']}")


def msd(v):
    v = [x for x in v if x is not None]
    if len(v) < 3:
        return None
    return statistics.mean(v), statistics.stdev(v)


def ms(t):
    return 'NA' if t is None else f'{t[0]:.2f} ± {t[1]:.2f}'


def lkey(r):
    return (LINE_ORDER.index(r['line']) if r['line'] in LINE_ORDER else 99, r['line'], r['arm_label'])


def tables(rows, lit, out):
    L = []; A = L.append
    A('# Layout-free TEST re-read of DEV-selected retrieval checkpoints (numbers only)\n')
    A(f"Generated {out['generated_utc']} by tools/aggregate_layoutfree.py from TEST_RESULTS.json and results_layoutfree/ "
      '(tools/test_eval_layoutfree.py). Same frozen manifests and checkpoints as TEST_TABLES.md; no new selection.\n')
    for k, v in out['variants'].items():
        A(f'- **{k}**: {v}')
    A('- Each re-read unit: native DEV recomputed and gated (0.1 pp vs logged DEV, test_eval.gate); native TEST recomputed and compared '
      'with the recorded TEST; same job, same GPU type as the recorded evaluation. DEV = full DEV recompute (not the logged value).')
    for h, e in NOT_AFFECTED.items():
        A(f'- host {h}: not affected. {e}')
    A('')
    inr = [r for r in rows if r['host'] == 'in2r']
    comp = [r for r in inr if r['status'] == 'COMPLETE']
    A('## Coverage and reproduction checks\n')
    A(f"Units: {len(rows)}; IN2R-host units: {len(inr)}; re-read: {len(comp)}; pending: {len(inr) - len(comp)}; "
      f"re-run DEV gate PASS: {sum(r['dev_gate'] == 'PASS' for r in comp)}; native TEST == recorded: {sum(r['test_native_match'] for r in comp)}; "
      f"not affected (SPS / NRCS): {sum(r['host'] != 'in2r' for r in rows)}.\n")
    if comp:
        A('| line | re-read | DEV gate PASS | max abs DEV recall diff (pp) | native TEST == recorded | max abs native TEST diff vs recorded | '
          'max abs native_rebuilt − native (DEV / TEST rSum-level) |')
        A('|---|---|---|---|---|---|---|')
        for ln in sorted({r['line'] for r in comp}, key=lambda x: LINE_ORDER.index(x) if x in LINE_ORDER else 99):
            cc = [r for r in comp if r['line'] == ln]
            A(f"| {ln} | {len(cc)} | {sum(r['dev_gate'] == 'PASS' for r in cc)} | {max(r['dev_max_abs_recall_diff_pp'] for r in cc):.4f} | "
              f"{sum(r['test_native_match'] for r in cc)} | {max(r['test_native_vs_recorded_max_abs'] for r in cc):.4f} | "
              f"{max(r['native_rebuilt_minus_native_max_abs']['dev'] for r in cc):.4f} / {max(r['native_rebuilt_minus_native_max_abs']['test'] for r in cc):.4f} |")
        A('')
        bad = [r for r in comp if not r['test_native_match'] or r['dev_gate'] != 'PASS' or r['recorded_dev_gate'] not in GATE_OK]
        if bad:
            A('### Units with a gate failure or a native TEST mismatch (kept out of the condition tables if a gate fails)\n')
            A('| unit | recorded gate | re-run gate | max DEV diff (pp) | native TEST − recorded max abs | recorded TEST rSum | native TEST rSum now |')
            A('|---|---|---|---|---|---|---|')
            for r in bad:
                A(f"| {r['unit_id']} | {r['recorded_dev_gate']} | {r['dev_gate']} | {r['dev_max_abs_recall_diff_pp']:.4f} | "
                  f"{r['test_native_vs_recorded_max_abs']:.4f} | {f2(tmain(r['recorded_native'])['rsum'])} | {f2(tmain(r['readouts']['native'])['rsum'])} |")
            A('')
    pend = [r for r in inr if r['status'] != 'COMPLETE']
    if pend:
        A(f"Pending: {', '.join(r['unit_id'] for r in pend)}\n")
    ok = [r for r in rows if r.get('readouts') and r['in_tables']]
    for cond in CONDS:
        ds, nz = cond
        cr = [r for r in ok if r['dataset'] == ds and r['noise'] == nz]
        if not cr:
            continue
        coco = ds == 'coco'
        A(f'## {COND_NAME[cond]}\n')
        lr = []
        if ds in ('f30k', 'coco') and str(nz) in lit['noise_tables']:
            kk = 'f30k' if ds == 'f30k' else 'coco_5fold1k'
            lr = [(m, v[kk]) for m, v in lit['noise_tables'][str(nz)].items() if kk in v]
        elif ds == 'cc152k':
            lr = list(lit['cc152k'].items())
        if lr:
            A('### Literature (paper-reported, not reproduced here) -- IN2R paper arXiv:2606.04061v1, ' + ('Table 2 (CC152K)' if ds == 'cc152k' else 'Table 1')
              + (' -- MS-COCO 5-fold 1K' if coco else (' -- Flickr30K 1K test' if ds == 'f30k' else '')) + '\n')
            A('| method | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | rSum |')
            A('|---|---|---|---|---|---|---|---|')
            for m, v in lr:
                A(f"| {m} (paper) | " + ' | '.join(f'{v[k]:.1f}' for k in KEYS) + f" | {v['rsum']:.1f} |")
            A('')
        tl = 'TEST 5-fold 1K' if coco else 'TEST'
        groups = {}
        for r in cr:
            groups.setdefault((r['line'], r['arm_label']), []).append(r)
        A(f'### rSum per group, 3-seed mean ± std (NA unless seeds 1001/1002/1003 all present); n = seeds present\n')
        h = '| line | arm | n | DEV native | DEV global | DEV decoupled | ' + f'{tl} native | {tl} global | {tl} decoupled | {tl} global − native |'
        s = '|---|---|---|---|---|---|---|---|---|---|'
        if coco:
            h += ' 5K native | 5K global | 5K decoupled |'; s += '---|---|---|'
        A(h); A(s)
        for key in sorted(groups, key=lambda k: lkey(groups[k][0])):
            g = groups[key]; bys = {r['seed']: r for r in g}
            tag = ' (oracle)' if g[0]['oracle_diagnostic'] else ''
            tag += ' (native IN2R repro)' if g[0]['native_in2r_reproduction'] else ''
            tag += ' (not affected)' if g[0]['host'] != 'in2r' else ''
            dv = {v: msd([bys[x]['readouts'][v]['dev']['rsum'] if x in bys else None for x in SEEDS]) for v in VARS}
            tv = {v: msd([tmain(bys[x]['readouts'][v])['rsum'] if x in bys else None for x in SEEDS]) for v in VARS}
            dl = msd([tmain(bys[x]['readouts']['global'])['rsum'] - tmain(bys[x]['readouts']['native'])['rsum'] if x in bys else None for x in SEEDS])
            ln = f"| {key[0]} | {key[1]}{tag} | {len(g)} | " + ' | '.join(ms(dv[v]) for v in VARS) + ' | ' + ' | '.join(ms(tv[v]) for v in VARS) + f' | {ms(dl)} |'
            if coco:
                t5 = {v: msd([tmain(bys[x]['readouts'][v], '5k')['rsum'] if x in bys else None for x in SEEDS]) for v in VARS}
                ln += ' ' + ' | '.join(ms(t5[v]) for v in VARS) + ' |'
            A(ln)
        A('')
        A('### Six TEST recalls per variant, 3-seed mean (groups with all three seeds)' + (' -- MS-COCO 5-fold 1K' if coco else '') + '\n')
        A('| line | arm | variant | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | rSum |')
        A('|---|---|---|---|---|---|---|---|---|---|')
        for key in sorted(groups, key=lambda k: lkey(groups[k][0])):
            g = groups[key]; bys = {r['seed']: r for r in g}
            if not all(x in bys for x in SEEDS) or g[0]['host'] != 'in2r':
                continue
            for v in VARS:
                tt = [tmain(bys[x]['readouts'][v]) for x in SEEDS]
                A(f"| {key[0]} | {key[1]} | {v} | " + ' | '.join(f"{statistics.mean(t[k] for t in tt):.2f}" for k in KEYS) + f" | {statistics.mean(t['rsum'] for t in tt):.2f} |")
        A('')
        A('### Per unit (rSum)\n')
        h = '| unit | seed | epoch | DEV native | DEV global | DEV decoupled | ' + f'{tl} recorded | {tl} native now | {tl} global | {tl} decoupled |'
        s = '|---|---|---|---|---|---|---|---|---|---|'
        if coco:
            h += ' 5K native | 5K global | 5K decoupled |'; s += '---|---|---|'
        A(h); A(s)
        for r in sorted(cr, key=lambda r: (lkey(r), r['seed'])):
            R = r['readouts']
            ln = (f"| {r['unit_id']}{' (not affected)' if r['host'] != 'in2r' else ''} | {r['seed']} | {r['dev_selected_epoch']} | " +
                  ' | '.join(f2(R[v]['dev']['rsum']) for v in VARS) + f" | {f2(tmain(r['recorded_native'])['rsum'])} | " +
                  ' | '.join(f2(tmain(R[v])['rsum']) for v in VARS) + ' |')
            if coco:
                ln += ' ' + ' | '.join(f2(tmain(R[v], '5k')['rsum']) for v in VARS) + ' |'
            A(ln)
        A('')
    A('## Limitations\n')
    A('- The layout-free variants change the input statistics of the similarity for models trained with native per-batch centring; '
      'global − native mixes removal of the layout effect with this distribution shift (R44_AUDIT.md caveat).')
    A('- decoupled uses one fixed random batch assignment per split / fold (seeds 44 / 4444); no average over assignments.')
    A('- Gate-failing units of TEST_TABLES.md stay out of these tables; NRCS-v1 and SPS rows are copied from TEST_RESULTS.json (not re-read).')
    A('- This is a second TEST read of the same checkpoints (user authorisation 2026-11-04); nothing is selected on TEST.')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
