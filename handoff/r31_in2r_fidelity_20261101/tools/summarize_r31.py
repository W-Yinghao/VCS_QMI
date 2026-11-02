"""R31 IN2R reproduction-fidelity table (numbers only, no interpretation).

For each R31 set (a: official NCR 20% index + R25 code config; b: author run.py defaults; b': defaults + --i2t --t2i) and
cell, reports the line's own best full-DEV rSum and the one-shot TEST (Flickr30K 1K / CC152K test) six Recalls and rSum from
the standing TEST tool, the 3-seed mean, the IN2R paper value (LITERATURE_TABLE.json, arXiv:2606.04061v1) and our earlier
native reproductions under the same condition (R25 H0 Flickr 40%, R26 H0 CC152K). Only units passing the DEV-recompute gate
enter the TEST columns. Writes R31_FIDELITY.md / R31_FIDELITY.json next to DESIGN.json.
Usage: python3 -B tools/summarize_r31.py"""
import json, os, statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TE = os.path.join(os.path.dirname(HERE), 'test_eval_20261101')
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101/runs'
KEYS = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
D = json.load(open(os.path.join(HERE, 'DESIGN.json')))
U = json.load(open(os.path.join(TE, 'TEST_RESULTS.json')))['units']
U = list(U.values()) if isinstance(U, dict) else U
LITJ = json.load(open(os.path.join(TE, 'LITERATURE_TABLE.json'))); LIT = LITJ['noise_tables']


def test_of(u):
    t = u.get('test') or {}
    return {k: t.get(k) for k in KEYS} if t.get('rsum') is not None else None


def by_line(line, arm, ds, noise):
    out = {}
    for u in U:
        if u['line'] == line and u['arm'] == arm and u['dataset'] == ds and str(u['noise']) == str(noise) and u.get('dev_gate') in ('PASS', 'PASS_LOGGING_RULE'):
            out[u['cell']] = u
    return out


def mean(v):
    v = [x for x in v if x is not None]
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else None, len(v)) if v else (None, None, 0)


COND = {'f20': ('f30k', 20, 'f30k', '20'), 'f40': ('f30k', 40, 'f30k', '40'), 'cc': ('cc152k', 'real', 'cc152k', 'cc152k')}
REF = {'f40': ('R25', 'H0', 'f30k', 40), 'cc': ('R26', 'H0', 'cc152k', 'real')}


def paper(cond):
    key = COND[cond][3]
    if key == 'cc152k':
        return (LITJ['cc152k'].get('IN2R') or {}).get('rsum')
    return ((LIT.get(key) or {}).get('IN2R') or {}).get('f30k', {}).get('rsum')


rows, groups = [], {}
for setk, s in D['sets'].items():
    for c in s['cells']:
        cond = c.split('_')[0]
        ds, noise = COND[cond][:2]
        d = f'{O}/{c}/{s["arm"]}'
        done = json.load(open(f'{d}/DONE.json')) if os.path.exists(f'{d}/DONE.json') else {}
        u = by_line('R31-FID', s['arm'], ds, noise).get(c)
        t = test_of(u) if u else None
        r = dict(set=setk, cell=c, arm=s['arm'], dev_best_rsum=done.get('best_rsum'), test=t, dev_gate=u.get('dev_gate') if u else None)
        rows.append(r)
        groups.setdefault((setk, cond), []).append(r)

L = ['# R31 IN²R 复现忠实度（数字表，无解读）', '',
     'TEST 为常设 TEST 工具对各单元 DEV 选出 checkpoint 的一次性评测（上游 IN²R evaluation.py；Flickr30K 1K test，CC152K test），只纳入通过 DEV 复算门禁的单元；不用于任何选择。论文值取自 arXiv:2606.04061v1 Table 1–2（LITERATURE_TABLE.json）。', '',
     '| set | cell | arm | best full-DEV rSum | TEST i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | TEST rSum |', '|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    t = r['test'] or {}
    f = lambda k: '—' if t.get(k) is None else f'{t[k]:.2f}'
    dv = 'NA' if r['dev_best_rsum'] is None else '%.2f' % r['dev_best_rsum']
    L.append(f"| {r['set']} | {r['cell']} | {r['arm']} | {dv} | " + ' | '.join(f(k) for k in KEYS) + ' |')
L += ['', '## 三 seed 汇总与对照', '', '| set | condition | n | best-DEV rSum mean | TEST rSum mean ± sd | IN²R paper TEST rSum | our earlier native reproduction TEST rSum mean (line) |', '|---|---|---|---|---|---|---|']
summ = []
for (setk, cond), rr in groups.items():
    dm = mean([r['dev_best_rsum'] for r in rr]); tm = mean([(r['test'] or {}).get('rsum') for r in rr])
    ref = None
    if cond in REF:
        refu = by_line(*REF[cond])
        ref = mean([(test_of(u) or {}).get('rsum') for u in refu.values()])
    p = paper(cond)
    summ.append(dict(set=setk, condition=cond, n_test=tm[2], dev_mean=dm[0], test_mean=tm[0], test_sd=tm[1], paper=p,
                     earlier_reproduction=dict(line=REF.get(cond, [None])[0], mean=ref[0] if ref else None, n=ref[2] if ref else 0)))
    fm = lambda x: 'NA' if x is None else f'{x:.2f}'
    L.append(f"| {setk} | {cond} | {tm[2]} | {fm(dm[0])} | {fm(tm[0])}{'' if tm[1] is None else ' ± ' + fm(tm[1])} | {fm(p)} | "
             + (f"{fm(ref[0])} ({REF[cond][0]} {REF[cond][1]}, n={ref[2]})" if ref else '—') + ' |')
open(os.path.join(HERE, 'R31_FIDELITY.md'), 'w').write('\n'.join(L) + '\n')
json.dump(dict(note=__doc__, rows=rows, summary=summ), open(os.path.join(HERE, 'R31_FIDELITY.json'), 'w'), indent=1, ensure_ascii=False)
print('\n'.join(L[-(len(summ) + 2):]))
