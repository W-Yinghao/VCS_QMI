"""Rebuild TEST_RESULTS.json and TEST_TABLES.md from every frozen manifest (TEST_ACCESS_LOG.jsonl) and results/*.json.
Record arithmetic only (no model / data access). Numbers only, no interpretation.
"""
import os, json, glob, hashlib, datetime, statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(HERE, 'TEST_ACCESS_LOG.jsonl')
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10')
SEEDS = (1001, 1002, 1003)
NATIVE = {('R25', 'H0'), ('R26', 'H0')}
LINE_ORDER = ['R25', 'R25-SPS', 'R26', 'R27', 'R27-D', 'R27-O', 'R27-O3', 'R27-CS', 'R27-Q', 'R27-A1X', 'R28', 'R29', 'R30']
ARM_NOTE = {
    ('R25', 'H0'): 'native IN2R (SCE) -- our native IN2R reproduction', ('R25', 'V0'): 'VCS, critic C0', ('R25', 'V1'): 'VCS, critic C1',
    ('R25', 'B0'): 'balanced BCE, critic C0', ('R25', 'B1'): 'balanced BCE, critic C1', ('R25-SPS', 'S0_sps'): 'SPS native code (external reference)',
    ('R26', 'H0'): 'native IN2R (SCE) -- our native IN2R reproduction',
}
CONDS = [('f30k', 0), ('f30k', 40), ('f30k', 60), ('coco', 40), ('cc152k', 'real')]
COND_NAME = {('f30k', 0): 'Flickr30K, 0% noise', ('f30k', 40): 'Flickr30K, 40% noise', ('f30k', 60): 'Flickr30K, 60% noise',
             ('coco', 40): 'MS-COCO, 40% noise', ('cc152k', 'real'): 'CC152K (real noise)'}


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def f2(x):
    return 'NA' if x is None else f'{x:.2f}'


def msd(v):
    if len(v) < 2:
        return None, None
    return statistics.mean(v), statistics.stdev(v)


GATE_OK = ('PASS', 'PASS_LOGGING_RULE')
_XC = os.path.join(HERE, 'diag', 'NRCS_RANK_RULE_CROSSCHECK.json')
LOGGING_RULE = {'NRCS-v1': {x['view']: x for x in json.load(open(_XC))['rows']}} if os.path.exists(_XC) else {}


def main():
    events = [json.loads(l) for l in open(LOG)]
    mans = [(e['manifest'], e['sha256']) for e in events if e.get('event') == 'manifest_frozen']
    units, skipped = [], []
    for mp, sha in mans:
        assert sha_file(mp) == sha, mp
        m = json.load(open(mp))
        for u in m['units']:
            units.append(dict(u, manifest=mp))
        skipped += m.get('skipped', [])
    # drop skipped entries for dirs that later became units
    unit_dirs = {u['run_dir'] for u in units}
    skipped = [s for s in skipped if s['path'] not in unit_dirs]
    seen_sk, sk2 = set(), []
    for s in skipped:
        if s['path'] not in seen_sk:
            seen_sk.add(s['path']); sk2.append(s)
    skipped = sk2
    rows = []
    for u in units:
        p = os.path.join(HERE, 'results', u['checkpoint_sha256'][:16] + '.json')
        r = json.load(open(p)) if os.path.exists(p) else None
        row = dict(unit_id=u['unit_id'], line=u['line'], cell=u['cell'], arm=u['arm'], arm_label=u.get('arm_label') or u['arm'],
                   dataset=u['dataset'], noise=u['noise'], seed=u['seed'], oracle_diagnostic=u['oracle_diagnostic'],
                   native_in2r_reproduction=(u['line'], u['arm']) in NATIVE, source=u['source'], run_dir=u['run_dir'],
                   checkpoint=u['checkpoint'], checkpoint_sha256=u['checkpoint_sha256'], manifest=u['manifest'],
                   dev_selected_epoch=u['checkpoint_epoch'], epochs_done=u.get('epochs_done'), logged_dev=u['logged_dev'],
                   manifest_checks_pass=u['all_checks_pass'], training_gpus=sorted({str(t.get('gpu')) for t in u.get('training_runs', [])}))
        if r is None:
            row['status'] = 'PENDING'
        else:
            row['status'] = r['status']
            if r['status'] == 'COMPLETE':
                # gate re-derived here with a 1e-9 float epsilon on the 0.1 pp tolerance (one CC152K query = exactly 0.1 pp);
                # the job-time verdict is kept as dev_gate_jobtime
                ok = (r['dev_max_abs_recall_diff_pp'] <= 0.1 + 1e-9 and int(r['dev_recomputed']['n_images']) == int(u['logged_dev']['n_images'])
                      and int(r['dev_recomputed']['n_captions']) == int(u['logged_dev']['n_captions']))
                row.update(dev_recomputed=r['dev_recomputed'], dev_diff_recomputed_minus_logged=r['dev_diff_recomputed_minus_logged'],
                           dev_max_abs_recall_diff_pp=r['dev_max_abs_recall_diff_pp'], dev_gate='PASS' if ok else 'FAIL',
                           dev_gate_jobtime=r['dev_gate'], eval_env=r['env'], eval_job=r['env']['job'])
                if not ok and u['line'] in LOGGING_RULE:
                    # GATE_AMENDMENT_NRCS_EVAL_RULE_20261102.md: the line logged DEV under its own rank rule; the gate is checked
                    # under that rule on the same score matrix (DEV-only cross-check); table values stay the upstream program's
                    xc = LOGGING_RULE[u['line']].get(u['run_dir'].split('test_view/')[-1])
                    if (xc and xc['package_minus_logged_max_pp'] <= 0.1 + 1e-9
                            and abs(xc['upstream']['rsum'] - r['dev_recomputed']['rsum']) < 1e-6):
                        row.update(dev_gate='PASS_LOGGING_RULE', dev_logging_rule_check=dict(
                            source='diag/NRCS_RANK_RULE_CROSSCHECK.json', package_rule_rsum=xc['package']['rsum'],
                            upstream_rsum=xc['upstream']['rsum'], tie_worst_rsum=xc['tie_worst']['rsum'],
                            tie_best_rsum=xc['tie_best']['rsum'], package_minus_logged_max_pp=xc['package_minus_logged_max_pp']))
                if u['line'] in LOGGING_RULE:
                    # unified table rule: every unit of a line that logged DEV under another rank rule shows the upstream-program
                    # DEV recompute in the tables; the logged value (logging rule) is kept alongside
                    row['logged_dev_logging_rule'] = row['logged_dev']
                    row['logged_dev'] = dict(r['dev_recomputed'], source='upstream evaluation.py recompute (unified table rule)')
                if 'test' in r:
                    row['test'] = r['test']
                else:
                    row['test_5fold1k'] = r['test_5fold1k']; row['test_5k'] = r['test_5k']
        rows.append(row)
    lit = json.load(open(os.path.join(HERE, 'LITERATURE_TABLE.json')))
    out = dict(generated_utc=datetime.datetime.utcnow().isoformat() + 'Z', manifests=[dict(file=a, sha256=b) for a, b in mans],
               n_units=len(rows), n_complete=sum(r['status'] == 'COMPLETE' for r in rows),
               n_gate_pass=sum(r.get('dev_gate') in GATE_OK for r in rows), units=rows, skipped=skipped,
               test_read_events=[e for e in events if e.get('event') == 'test_read'])
    json.dump(out, open(os.path.join(HERE, 'TEST_RESULTS.json'), 'w'), indent=1)
    md = tables(rows, skipped, mans, lit, events)
    open(os.path.join(HERE, 'TEST_TABLES.md'), 'w').write(md)
    print(f"{out['n_units']} units, {out['n_complete']} complete, {out['n_gate_pass']} gate PASS")


def tsum(r, which='main'):
    if 'test' in r:
        return r['test']
    if 'test_5fold1k' in r:
        return r['test_5fold1k']['mean'] if which == 'main' else r['test_5k']
    return None


def lkey(r):
    return (LINE_ORDER.index(r['line']) if r['line'] in LINE_ORDER else 99, r['line'], r['arm_label'])


def tables(rows, skipped, mans, lit, events):
    L = []
    A = L.append
    A('# TEST evaluation of DEV-selected retrieval checkpoints (numbers only)\n')
    A(f"Generated {datetime.datetime.utcnow().isoformat()}Z by tools/aggregate_test.py. Frozen manifests: " +
      ', '.join(f"`{os.path.basename(a)}` (sha256 {b})" for a, b in mans) + '.')
    A('Checkpoint per unit = existing model_best.pth.tar (best full-DEV epoch during training; SPS S0: full-DEV best checkpoint_37 '
      'from FULLDEV.json). TEST protocol = native IN2R evalrank: both peers, peer-averaged similarity. Flickr30K: test split '
      '1,000 images / 5,000 captions. MS-COCO: testall, 5-fold 1K mean (main) and 5K. CC152K: test 1,000 / 1,000. '
      'DEV gate: DEV recomputed with the same evaluator (r25_co_train.validate path) must match the logged best-epoch DEV within '
      '0.1 pp on each of the six recalls; units failing it are listed separately and are not in the tables. '
      'mean ± std = over seeds 1001/1002/1003 (sample std, n−1), given only when all three seeds exist; otherwise NA. '
      'Oracle rows (R27-O / R27-O3) use ground-truth rescue during training: oracle diagnostic, not a method.\n')
    comp = [r for r in rows if r['status'] == 'COMPLETE']
    gp = [r for r in comp if r['dev_gate'] in GATE_OK]
    gf = [r for r in comp if r['dev_gate'] not in GATE_OK]
    A('## Coverage\n')
    A(f"Units in manifests: {len(rows)}; evaluated: {len(comp)}; DEV gate PASS: {len(gp)}; FAIL: {len(gf)}; "
      f"pending/not evaluated: {len(rows) - len(comp)}; skipped dirs: {len(skipped)}.\n")
    A('| line | units | evaluated | gate PASS | gate FAIL | max abs DEV recall diff (pp) |')
    A('|---|---|---|---|---|---|')
    for ln in sorted({r['line'] for r in rows}, key=lambda x: LINE_ORDER.index(x) if x in LINE_ORDER else 99):
        rr = [r for r in rows if r['line'] == ln]
        cc = [r for r in rr if r['status'] == 'COMPLETE']
        mx = max([r['dev_max_abs_recall_diff_pp'] for r in cc], default=None)
        A(f"| {ln} | {len(rr)} | {len(cc)} | {sum(r['dev_gate'] in GATE_OK for r in cc)} | {sum(r['dev_gate'] not in GATE_OK for r in cc)} | "
          f"{'NA' if mx is None else f'{mx:.4f}'} |")
    A('')
    if gf:
        A('### Units failing the DEV-recompute gate (TEST numbers not used in the tables)\n')
        A('| unit | logged DEV rSum | recomputed DEV rSum | max abs recall diff (pp) | TEST rSum |')
        A('|---|---|---|---|---|')
        for r in gf:
            A(f"| {r['unit_id']} | {f2(r['logged_dev']['rsum'])} | {f2(r['dev_recomputed']['rsum'])} | {r['dev_max_abs_recall_diff_pp']:.4f} | {f2(tsum(r)['rsum'])} |")
        A('')
    lr = [r for r in comp if r['dev_gate'] == 'PASS_LOGGING_RULE']
    if lr:
        A('### Units passing the DEV gate under their logging rank rule (GATE_AMENDMENT_NRCS_EVAL_RULE_20261102.md)\n')
        A('These lines logged DEV with a different rank rule (R32 NRCS-v1: package retrieval_metrics, ties by increasing candidate index). '
          'The gate is checked under that rule on the same full-DEV score matrix (diag/NRCS_RANK_RULE_CROSSCHECK.json, DEV only); '
          'all table values are the upstream IN2R evaluation.py values. The job-time gate FAIL records are kept. '
          'Tie range = rSum with every tied candidate ranked ahead / none ranked ahead.\n')
        A('| unit | logged DEV rSum (logging rule) | DEV rSum, logging rule recomputed | DEV rSum, upstream | DEV tie range | TEST rSum (upstream) |')
        A('|---|---|---|---|---|---|')
        for r in lr:
            c = r['dev_logging_rule_check']
            A(f"| {r['unit_id']} | {f2(r['logged_dev_logging_rule']['rsum'])} | {f2(c['package_rule_rsum'])} | {f2(c['upstream_rsum'])} | "
              f"{f2(c['tie_worst_rsum'])}–{f2(c['tie_best_rsum'])} | {f2(tsum(r)['rsum'])} |")
        A('')
    pend = [r for r in rows if r['status'] != 'COMPLETE']
    if pend:
        A('### Units not evaluated yet\n')
        A(', '.join(f"{r['unit_id']} ({r['status']})" for r in pend) + '\n')

    for cond in CONDS:
        ds, nz = cond
        cr = [r for r in gp if r['dataset'] == ds and r['noise'] == nz]
        if not cr and not (ds == 'f30k' and nz in (40, 60)) and cond != ('coco', 40) and ds != 'cc152k':
            continue
        A(f'## {COND_NAME[cond]}\n')
        coco = ds == 'coco'
        # literature
        litrows = []
        if ds == 'f30k' and str(nz) in lit['noise_tables']:
            litrows = [(m, v['f30k']) for m, v in lit['noise_tables'][str(nz)].items() if 'f30k' in v]
        elif coco and str(nz) in lit['noise_tables']:
            litrows = [(m, v['coco_5fold1k']) for m, v in lit['noise_tables'][str(nz)].items() if 'coco_5fold1k' in v]
        elif ds == 'cc152k':
            litrows = list(lit['cc152k'].items())
        if litrows:
            A('### Literature (paper-reported, not reproduced here) -- IN2R paper arXiv:2606.04061v1, ' + ('Table 2 (CC152K)' if ds == 'cc152k' else 'Table 1') + (' -- MS-COCO 5-fold 1K' if coco else (' -- Flickr30K 1K test' if ds == 'f30k' else '')) + '\n')
            A('| method | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | rSum |')
            A('|---|---|---|---|---|---|---|---|')
            for m, v in litrows:
                A(f"| {m} (paper) | " + ' | '.join(f'{v[k]:.1f}' for k in KEYS) + f" | {v['rsum']:.1f} |")
            A('')
        elif ds == 'f30k' and nz == 0:
            A('No paper value at 0% noise in the literature table.\n')
        if not cr:
            A('No evaluated unit.\n'); continue
        groups = {}
        for r in cr:
            groups.setdefault((r['line'], r['arm_label']), []).append(r)
        A('### rSum per group (DEV = logged best full-DEV rSum of the selected epoch; TEST = one-shot TEST rSum)\n')
        tl = 'TEST 5-fold 1K' if coco else 'TEST'
        hdr = '| line | arm | note | n | DEV s1001 | DEV s1002 | DEV s1003 | DEV mean ± std | ' + f'{tl} s1001 | {tl} s1002 | {tl} s1003 | {tl} mean ± std |'
        sep = '|---|---|---|---|---|---|---|---|---|---|---|---|'
        if coco:
            hdr += ' TEST 5K s1001 | 5K s1002 | 5K s1003 | 5K mean ± std |'; sep += '---|---|---|---|'
        A(hdr); A(sep)
        ranking = []
        for key in sorted(groups, key=lambda k: lkey(groups[k][0])):
            g = groups[key]
            bys = {r['seed']: r for r in g}
            note = []
            if g[0]['native_in2r_reproduction']:
                note.append('**native IN2R reproduction**')
            if g[0]['oracle_diagnostic']:
                note.append('oracle diagnostic, not a method')
            an = ARM_NOTE.get((g[0]['line'], g[0]['arm']))
            if an and not g[0]['native_in2r_reproduction']:
                note.append(an)
            ep = sorted({r.get('epochs_done') for r in g if r.get('epochs_done') is not None})
            if g[0]['line'] == 'R30' or ep != [40]:
                note.append('epochs ' + '/'.join(map(str, ep)))
            dv = [bys[s]['logged_dev']['rsum'] if s in bys else None for s in SEEDS]
            tv = [tsum(bys[s])['rsum'] if s in bys else None for s in SEEDS]
            full = all(x is not None for x in dv)
            dm, dsd = msd(dv) if full else (None, None)
            tm, tsd = msd(tv) if full else (None, None)
            line = (f"| {key[0]} | {key[1]} | {'; '.join(note)} | {len(g)} | " + ' | '.join(f2(x) for x in dv) +
                    f" | {'NA' if dm is None else f'{dm:.2f} ± {dsd:.2f}'} | " + ' | '.join(f2(x) for x in tv) +
                    f" | {'NA' if tm is None else f'{tm:.2f} ± {tsd:.2f}'} |")
            if coco:
                t5 = [tsum(bys[s], '5k')['rsum'] if s in bys else None for s in SEEDS]
                t5m, t5s = msd(t5) if full else (None, None)
                line += ' ' + ' | '.join(f2(x) for x in t5) + f" | {'NA' if t5m is None else f'{t5m:.2f} ± {t5s:.2f}'} |"
            A(line)
            avail = [s for s in SEEDS if s in bys]
            ranking.append(dict(key=key, n=len(avail), seeds=avail, dev=statistics.mean(bys[s]['logged_dev']['rsum'] for s in avail),
                                test=statistics.mean(tsum(bys[s])['rsum'] for s in avail), oracle=g[0]['oracle_diagnostic'],
                                native=g[0]['native_in2r_reproduction'], g=g, full=full))
        A('')
        A('### Six TEST recalls, 3-seed mean (groups with all three seeds)' + (' -- MS-COCO 5-fold 1K' if coco else '') + '\n')
        A('| line | arm | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | rSum |')
        A('|---|---|---|---|---|---|---|---|---|')
        for rk in ranking:
            if rk['full']:
                rr = [tsum(r) for r in rk['g']]
                A(f"| {rk['key'][0]} | {rk['key'][1]}{' (oracle)' if rk['oracle'] else ''}{' (native IN2R repro)' if rk['native'] else ''} | " +
                  ' | '.join(f"{statistics.mean(x[k] for x in rr):.2f}" for k in KEYS) + f" | {statistics.mean(x['rsum'] for x in rr):.2f} |")
        A('')
        A('### Ranking by mean DEV rSum (seeds available per group; n differs between groups), with mean TEST rSum of the same seeds\n')
        A('| rank | line | arm | n seeds | mean DEV rSum | mean ' + tl + ' rSum |')
        A('|---|---|---|---|---|---|')
        for i, rk in enumerate(sorted(ranking, key=lambda x: -x['dev']), 1):
            A(f"| {i} | {rk['key'][0]} | {rk['key'][1]}{' (oracle)' if rk['oracle'] else ''}{' (native IN2R repro)' if rk['native'] else ''} | "
              f"{rk['n']} | {rk['dev']:.2f} | {rk['test']:.2f} |")
        A('')
        s1 = [rk for rk in ranking if 1001 in rk['seeds']]
        if s1:
            A('### Seed-matched ranking: seed 1001 only (every group that has seed 1001), DEV rSum with TEST rSum\n')
            A('| rank | line | arm | DEV rSum s1001 | ' + tl + ' rSum s1001 |')
            A('|---|---|---|---|---|')
            b1 = {rk['key']: [r for r in rk['g'] if r['seed'] == 1001][0] for rk in s1}
            for i, rk in enumerate(sorted(s1, key=lambda x: -b1[x['key']]['logged_dev']['rsum']), 1):
                r = b1[rk['key']]
                A(f"| {i} | {rk['key'][0]} | {rk['key'][1]}{' (oracle)' if rk['oracle'] else ''}{' (native IN2R repro)' if rk['native'] else ''} | "
                  f"{r['logged_dev']['rsum']:.2f} | {tsum(r)['rsum']:.2f} |")
            A('')
        full3 = [rk for rk in ranking if rk['full'] and not rk['oracle']]
        if full3:
            b = max(full3, key=lambda x: x['dev'])
            A(f"DEV-selected group among non-oracle groups with all three seeds: {b['key'][0]} {b['key'][1]} -- mean DEV {b['dev']:.2f}, mean {tl} {b['test']:.2f}.")
        mx = max(cr, key=lambda r: tsum(r)['rsum'])
        A(f"Post-hoc note: the single highest TEST row in this condition is {mx['unit_id']} ({tl} rSum {tsum(mx)['rsum']:.2f}, DEV {mx['logged_dev']['rsum']:.2f}); "
          'picking the maximum TEST row (or group) across methods uses TEST for selection and is optimistic, not a DEV-selected result.\n')
        A('### Per unit\n')
        h = '| unit | seed | epoch | GPU (train) | DEV logged | DEV recomputed | max diff (pp) | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | ' + tl + ' rSum |'
        s = '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
        if coco:
            h += ' 5K rSum |'; s += '---|'
        A(h); A(s)
        for r in sorted(cr, key=lambda r: (lkey(r), r['seed'])):
            t = tsum(r)
            ln = (f"| {r['unit_id']} | {r['seed']} | {r['dev_selected_epoch']} | {', '.join(x.replace('NVIDIA ', '').replace(' Blackwell Server Edition', '') for x in r['training_gpus'])} | "
                  f"{r['logged_dev']['rsum']:.2f} | {r['dev_recomputed']['rsum']:.2f} | {r['dev_max_abs_recall_diff_pp']:.4f} | " +
                  ' | '.join(f'{t[k]:.2f}' for k in KEYS) + f" | {t['rsum']:.2f} |")
            if coco:
                ln += f" {tsum(r, '5k')['rsum']:.2f} |"
            A(ln)
        A('')
    A('## Skipped directories\n')
    A('| path | reason |')
    A('|---|---|')
    for sk in skipped:
        A(f"| {sk['path'].replace('/projects/EEG-foundation-model/yinghao/outputs_v9/', '')} | {sk['reason']} |")
    A('')
    A('## Limitations\n')
    A('- DEV (full DEV) was used heavily during development of all these lines (method design, critic selection, matrix decisions); '
      'only the checkpoint choice within each run is the pre-registered DEV rule. This is the first one-shot TEST read of these '
      'DEV-selected checkpoints. Flickr30K TEST was read once before in earlier V9-era work (e.g. job 1042195, full Flickr TEST; '
      '1043800 a 100-image subset), on different models.')
    A('- Training GPU types differ between runs and lines (RTX PRO 6000 / L40S / H100 / runfill V100/A100/A40, see the per-unit table); '
      'the evaluation jobs ran on runfill GPUs (V100/L40S/A100/A40); the DEV gate bounds evaluation-side differences only.')
    A('- Seeds: many groups have one seed; means/stds over three seeds are descriptive, not significance tests. Rankings mix groups with '
      'different numbers of seeds.')
    A('- Literature rows are copied from the IN2R paper (arXiv:2606.04061v1, Tables 1-2); they use their own noise draws and runs and were not '
      'reproduced here. Our noise maps are the locked R25 maps (40%/60%/0%); CC152K noise is real.')
    A('- R27-O / R27-O3 use ground-truth oracle rescue/targets during training (diagnostic, not a method). R30 stage paths stop at 10 '
      'co-train epochs by design (marked "epochs 10").')
    n_reads = sum(1 for e in events if e.get('event') == 'test_read')
    A(f'- TEST access log: {n_reads} test_read entries (one per evaluation job start, including re-starts after preemption).')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
