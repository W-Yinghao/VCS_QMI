"""Experiment list for R30 / R31 / R32 / R33 (numbers only, no interpretation; no TEST read here — TEST values are read
from the standing TEST tool's results). For every registered unit: design (condition, seed, arm, epochs), status
(DONE / RUNNING / PENDING / NOT_STARTED, from DONE files and squeue), the line's own DEV endpoint, and the one-shot TEST
rSum when the TEST tool has evaluated it. Writes EXPERIMENT_LIST_R30_R33.json and EXPERIMENT_LIST_R30_R33.md.
Usage: build_list.py"""
import csv, glob, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
H = '/home/infres/yinwang/CS_QMI/handoff'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9'
q = subprocess.run(['squeue', '-u', os.environ.get('USER', 'yinwang'), '-h', '-o', '%j|%T|%i'], capture_output=True, text=True).stdout
QUEUE = {}
for l in q.splitlines():
    n, s, j = l.split('|')
    QUEUE.setdefault(n, (s, j))
TEST = {}
tr = f'{H}/test_eval_20261101/TEST_RESULTS.json'
if os.path.exists(tr):
    U = json.load(open(tr))['units']
    for u in (U.values() if isinstance(U, dict) else U):
        t = u.get('test') or {}
        r = t.get('rsum') if isinstance(t, dict) else None
        if r is None and isinstance(t, dict):
            r = (t.get('fold5_1k') or t.get('5fold_1k') or {}).get('rsum')
        TEST[os.path.realpath(u['run_dir'])] = {'rsum': r, 'dev_gate': u.get('dev_gate')}


def status(done, jobname):
    if done:
        return 'DONE'
    if jobname in QUEUE:
        return QUEUE[jobname][0]
    return 'NOT_STARTED'


def test_of(d):
    t = TEST.get(os.path.realpath(d)) if d and os.path.exists(d) else None
    if not t:
        return None, None
    return t['rsum'], t['dev_gate']


rows = []
# ---- R30 ----
R = f'{O}/r30_pairwise_20261101/runs'
S30 = json.load(open(f'{H}/r30_pairwise_20261101/R30_SUMMARY.json'))['units'] if os.path.exists(f'{H}/r30_pairwise_20261101/R30_SUMMARY.json') else {}
for r in csv.DictReader(open(f'{H}/r30_pairwise_20261101/MATRIX_A1.csv')):
    u = r['unit_id']; d = f'{R}/{u}'; e = S30.get(u, {})
    tv, tg = test_of(d)
    rows.append(dict(line='R30', unit=u, condition=f"Flickr {r['noise_percent']}%", seed=r['seed'],
                     arm=f"{r['warmup_source']}→{r['continuation_objective']}" + (' (static scale)' if r['scale_rule'] == 'fit_static' else ''),
                     epochs=int(r['co_train_epochs']), status=status(os.path.exists(f'{d}/DONE.json'), u),
                     dev_label='E10 rSum / best-40 rSum', dev=(f"{e['E10']:.2f}" if e.get('E10') is not None else 'NA') + ' / ' + (f"{e['full']['best']:.2f}" if e.get('full') else ('NA' if int(r['co_train_epochs']) == 40 else '—')),
                     test=tv, test_gate=tg))
# ---- R31 ----
R = f'{O}/r31_in2r_fidelity_20261101/runs'
D31 = json.load(open(f'{H}/r31_in2r_fidelity_20261101/DESIGN.json'))
for k, s in D31['sets'].items():
    for c in s['cells']:
        d = f'{R}/{c}/{s["arm"]}'; jn = f'r31_{c}_{s["arm"]}'
        dev = None
        if os.path.exists(f'{d}/DONE.json'):
            dev = json.load(open(f'{d}/DONE.json')).get('best_rsum')
        tv, tg = test_of(d)
        cond = {'f20': 'Flickr 20% (official NCR index)', 'f40': 'Flickr 40%', 'cc': 'CC152K'}[c.split('_')[0]]
        rows.append(dict(line='R31', unit=f'{c}/{s["arm"]}', condition=cond, seed=c.split('_s')[1], arm=f'set {k}: {s["arm"]}',
                         epochs=40, status=status(dev is not None, jn), dev_label='best full-DEV rSum',
                         dev=dev if dev is not None else 'NA', test=tv, test_gate=tg))
# ---- R32 ----
R = f'{O}/r32_native_ratio_20261101/runs'
for r in csv.DictReader(open(f'{H}/r32_native_ratio_20261101/package/configs/MATRIX.csv')) if os.path.exists(f'{H}/r32_native_ratio_20261101/package/configs/MATRIX.csv') else csv.DictReader(open('/home/infres/yinwang/CS_QMI/r32_native_ratio/VCS_NATIVE_RATIO_v1_Server_20261101/configs/MATRIX.csv')):
    u = r['unit_id']; d = f'{R}/{u}'
    done = os.path.exists(f'{d}/DONE.json') or os.path.exists(f'{d}/execution_receipt.json') or os.path.exists(f'{d}/all_seed_metrics.json')
    dev = 'NA'
    m = f'{d}/metrics.json'
    if done and os.path.exists(m):
        mm = json.load(open(m)); last = mm.get('last') or {}
        ret = (last.get('dev') or {}).get('retrieval') or {}
        if isinstance(ret, dict) and 'rsum' in ret:
            dev = round(ret['rsum'], 2)
        elif 'J' in last:
            dev = f"J {last['J']:.4f}"
    tv, tg = test_of(d)
    st = 'DONE' if done else ('CHAIN_QUEUED' if r['stage'] in ('W2', 'E2', 'F1', 'C2') else 'NOT_STARTED')
    rows.append(dict(line='R32', unit=u, condition=r.get('condition') or f"noise {r['noise_percent']}%", seed=r['seed'], arm=f"{r['stage']}: {r['arm']}",
                     epochs=r.get('epochs') or r.get('steps'), status=st, dev_label='epoch-20 DEV rSum (E2/C2) / J (S0)', dev=dev, test=tv, test_gate=tg))
# ---- R33 ----
R = f'{O}/r33_baselines_20261102/runs'
C33 = json.load(open(f'{H}/r33_baselines_20261102/CELLS.json'))
for c, s in [(k, v) for k, v in C33.items() if not k.startswith('_')]:
    for a in s['arms']:
        d = f'{R}/{c}/{a}'; dev = None
        if os.path.exists(f'{d}/DONE.json'):
            dev = json.load(open(f'{d}/DONE.json')).get('best_rsum')
        jn = [n for n in QUEUE if n.startswith('r33_') and c in n and a in n]
        st = 'DONE' if dev is not None else (QUEUE[jn[0]][0] if jn else ('PREFIX_PENDING' if any(n == f'r33_prefix_{c}' for n in QUEUE) else 'NOT_STARTED'))
        tv, tg = test_of(d)
        ds = {'coco_precomp': 'COCO', 'f30k_precomp': 'Flickr'}[s['data_name']]
        rows.append(dict(line='R33', unit=f'{c}/{a}', condition=f"{ds} {int(round(s['noise_ratio'] * 100))}%", seed=s['seed'],
                         arm={'H0': 'native IN²R', 'a1_iface': 'IN²R + A1'}.get(a, a), epochs=40, status=st,
                         dev_label='best full-DEV rSum', dev=dev if dev is not None else 'NA', test=tv, test_gate=tg))

json.dump({'note': __doc__, 'rows': rows}, open(os.path.join(HERE, 'EXPERIMENT_LIST_R30_R33.json'), 'w'), indent=1, ensure_ascii=False)
from collections import Counter
L = ['# R30–R33 实验清单（自动生成；只列设计、状态与数字，不含解读）', '',
     '状态来自各单元 DONE 文件与当时的 squeue；DEV 为各线自身登记的终点；TEST 为常设 TEST 工具对 DEV（或登记的固定 epoch）选出 checkpoint 的一次性评测 rSum（COCO 为 5-fold 1K），未评测记 —。TEST 不用于任何选择。', '']
for line, title in [('R30', 'R30 v1.1 + 修订 A1：共同原生 warmup 后的直接竞争学习（阶段交叉与六臂完整比较）'),
                    ('R31', 'R31：IN²R 复现忠实度（(a) 官方 NCR 20% 噪声；(b) 作者 run.py 默认；(b′) 默认 + i2t/t2i）'),
                    ('R32', 'R32：VCS_NATIVE_RATIO_v1（GPO 双塔 + 有边缘约束的 ratio critic；S0/W2/F1/E2/C2）'),
                    ('R33', 'R33：基准基线行（原生 IN²R 与 IN²R + A1；COCO 60%、Flickr 80%、COCO 20%/80%、Flickr 20% A1）')]:
    R_ = [r for r in rows if r['line'] == line]
    c = Counter(r['status'] for r in R_)
    L += [f'## {title}', '', f"单元数 {len(R_)}；" + '，'.join(f'{k} {v}' for k, v in sorted(c.items())), '',
          '| 单元 | 条件 | seed | 臂 | 轮/步 | 状态 | DEV（' + (R_[0]['dev_label'] if R_ else '') + '） | TEST rSum |', '|---|---|---|---|---|---|---|---|']
    for r in R_:
        t = '—' if r['test'] is None else (f"{r['test']:.2f}" if isinstance(r['test'], float) else str(r['test']))
        if r['test_gate'] not in (None, 'PASS'):
            t += f" (DEV gate {r['test_gate']})"
        dv = r['dev'] if not isinstance(r['dev'], float) else f'{r["dev"]:.2f}'
        L.append(f"| {r['unit']} | {r['condition']} | {r['seed']} | {r['arm']} | {r['epochs']} | {r['status']} | {dv} | {t} |")
    L.append('')
open(os.path.join(HERE, 'EXPERIMENT_LIST_R30_R33.md'), 'w').write('\n'.join(L) + '\n')
print({l: Counter(r['status'] for r in rows if r['line'] == l) for l in ('R30', 'R31', 'R32', 'R33')})
