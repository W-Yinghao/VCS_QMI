"""Audit A1's six fixed-budget pilots and their six reused A0 controls; stdlib only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import statistics as S

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PACK = BASE / 'handoff/r22_mechanism_review_20261024'
ARMS = ('vcs_exact', 'bce', 'nce')
STEPS = (10000, 20000, 30000)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with Path(p).open() as f:
        for line in f:
            yield json.loads(line)


def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def identities(rr):
    return [(r['core'], r['scene'], r['perturbation'], r['E']['COS']['theta'],
             r['E']['COS']['t'], r['E']['COS']['n_grid']) for r in rr]


def metrics(rr):
    scenes = defaultdict(list)
    for r in rr:
        scenes[r['scene']].append(r['E']['COS']['err_full'])
    ee = sorted(x for v in scenes.values() for x in v)
    assert all(math.isfinite(x) and x >= 0 for x in ee)
    return dict(success9=S.mean(S.mean(x < 9 for x in v) for v in scenes.values()),
                success42=S.mean(S.mean(x < 42 for x in v) for v in scenes.values()),
                scene_mean_error=S.mean(S.mean(v) for v in scenes.values()),
                median_err=S.median(ee), p95_err=ee[math.ceil(.95*len(ee))-1],
                failed42=sum(x >= 42 for x in ee), failed100=sum(x >= 100 for x in ee),
                n=len(ee), scenes=len(scenes))


def evaluate(p, role, expected):
    rr = list(rows(p))
    assert len(rr) == expected and all(r['role'] == role and set(r['E']) == {'COS'} for r in rr)
    assert len({(r['core'], r['perturbation']) for r in rr}) == expected
    assert all({r['perturbation'] for r in rr if r['core'] == c} == set(range(10))
               for c in {r['core'] for r in rr})
    return rr, metrics(rr)


def difference(a, b):
    return {k: a[k]-b[k] for k in ('success9', 'success42', 'scene_mean_error',
                                  'median_err', 'p95_err', 'failed42', 'failed100')}


def main(check_only=False):
    jobs = read(PACK/'JOBS_a1_full.json')
    assert len(jobs) == len({j['name'] for j in jobs}) == 6
    missing = [j['name'] for j in jobs if not Path(j['completion']).exists()]
    if missing and not check_only:
        print(json.dumps(dict(status='WAITING_FOR_COMPLETE_SERIES', missing=missing)))
        return
    freeze = sha(PACK/'FREEZE.json')
    for p, h in read(PACK/'FREEZE.json')['files'].items():
        assert sha(BASE/p) == h, p
    cpu = read(PACK/'checks/a1.json')
    assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    gates = []
    for ds in ('bf', 'epfl'):
        gp = PACK/'runs'/f'a1_gate_{ds}'
        g = read(gp/'COMPLETE.json')
        assert g['status'] == 'COMPLETE' and g['freeze_sha256'] == freeze and g['ds'] == ds
        assert sha(gp/'INPUT_TRACE.json') == g['input_trace_sha256']
        assert {x['role'] for x in read(gp/'INPUT_TRACE.json')} == set(g['roles_read']) <= {'FIT', 'CAL'}
        assert {x['arm'] for x in g['results']} == set(ARMS)
        assert all(x['roundtrip'] and x['cal_gradient'] and len(x['steps']) == 3 for x in g['results'])
        gates.append(dict(job=g['job'], seconds=g['seconds']))
    locks = read(PACK/'SOURCE_LOCK.json')['geometry']
    signatures, eval_ids, units, curves, per_core = {}, {}, {}, [], []
    for j in jobs:
        p = Path(j['completion']).parent
        if not (p/'COMPLETE.json').exists():
            continue
        r = read(p/'COMPLETE.json'); ds, arm = r['ds'], r['arm']; key = f'{ds}_{arm}_1001'
        assert r['status'] == 'COMPLETE' and r['freeze_sha256'] == freeze and r['seed'] == 1001
        assert r['identity'] == dict(ds=ds, arm=arm, seed=1001, adaptation='A1_from_step3001', freeze_sha256=freeze)
        assert r['final_steps'] == 30000 and r['requested'] == 216000 and r['baseline'] == locks[key]
        b = locks[key]; bp = Path(b['completion']).parent; br = read(b['completion'])
        assert sha(b['checkpoint']) == b['sha256'] and sha(b['completion']) == b['completion_sha256']
        assert br['status'] == 'COMPLETE' and br['final_steps'] == 30000 and r['source'] == br['source']
        src = r['source']
        for k in ('checkpoint', 'metadata', 'completion', 'source_train'):
            assert sha(src[k]) == src[k+'_sha256'], (key, k)
        assert sha(p/'TRAIN.jsonl') == r['train_sha256'] and sha(bp/'TRAIN.jsonl') == br['train_sha256']
        active, requested, count, compute = 0, 0, 0, 0.
        geometry_hash, batch_hash = hashlib.sha256(), hashlib.sha256()
        loss_bins = defaultdict(list); grad_bins = defaultdict(list)
        baseline_loss_bins = defaultdict(list); baseline_grad_bins = defaultdict(list)
        for step, (a, z) in enumerate(zip_longest(rows(p/'TRAIN.jsonl'), rows(bp/'TRAIN.jsonl')), 3001):
            assert a and z and a['step'] == z['step'] == step
            assert a['batch_sha256'] == z['batch_sha256'] and a['lr'] == z['lr']
            assert a['lr'] == (3e-4 if step <= 15000 else 3e-5 if step <= 24000 else 3e-6)
            assert all(math.isfinite(v[k]) for v in (a,z) for k in ('loss','grad_norm_preclip','synchronized_step_seconds'))
            g = a['geometry']; sc = g['scenes']
            assert len(sc) == 16 and sorted(x['row'] for x in sc) == list(range(16))
            assert g['requested'] == sum(x['requested'] for x in sc) == 8
            assert g['active'] == sum(x['active'] for x in sc) <= 8
            assert all(not x['active'] or x['requested'] for x in sc)
            assert all(math.isfinite(x['theta_deg']) and all(math.isfinite(t) for t in x['translation_yx_px']) for x in sc)
            geometry_hash.update(json.dumps(g, sort_keys=True, separators=(',', ':')).encode())
            batch_hash.update(a['batch_sha256'].encode())
            active += g['active']; requested += 8; count += 1; compute += a['synchronized_step_seconds']
            bin_stop = ((step-1)//1000+1)*1000
            loss_bins[bin_stop].append(a['loss']); grad_bins[bin_stop].append(a['grad_norm_preclip'])
            baseline_loss_bins[bin_stop].append(z['loss']); baseline_grad_bins[bin_stop].append(z['grad_norm_preclip'])
        assert count == 27000 and active == r['active'] and requested == r['requested']
        sig = (geometry_hash.hexdigest(), batch_hash.hexdigest())
        if ds in signatures: assert sig == signatures[ds], (key,'cross-arm input stream')
        signatures[ds] = sig
        trace = read(p/'INPUT_TRACE.json')
        assert {x['role'] for x in trace} == {'FIT','CAL','DEV'}
        attempts = list(p.glob('ATTEMPT_*.json'))
        assert len(attempts) == 1 and attempts[0].stem == f'ATTEMPT_{r["job"]}'
        assert read(attempts[0])['status'] == 'STARTED' and not (p/'FAILURE.json').exists()
        env, benv = r['environment'], br['environment']
        assert env['gpu_name'] == benv['gpu_name'] == 'NVIDIA L40S'
        precision_diffs = {k:[env['precision'].get(k),benv['precision'].get(k)]
                           for k in set(env['precision'])|set(benv['precision'])
                           if env['precision'].get(k) != benv['precision'].get(k)}
        u = dict(ds=ds, arm=arm, seed=1001, job=r['job'], seconds=r['seconds'],
                 baseline_seconds=br['seconds'], prefix_seconds=read(src['completion'])['seconds'],
                 synchronized_training_seconds=compute, active=active, requested=requested,
                 active_fraction_all_rows=active/(27000*16), input_stream_hashes=sig,
                 a1_environment=env, a0_environment=benv, precision_differences=precision_diffs,
                 attempt_record=read(attempts[0]), completion_sha256=sha(p/'COMPLETE.json'),
                 input_trace_sha256=sha(p/'INPUT_TRACE.json'), metrics={}, cal={})
        for stop in STEPS:
            ck = p/'checkpoints'/f'step_{stop}.pt'; meta = read(ck.with_suffix('.json'))
            assert meta['step'] == stop and meta['identity'] == r['identity'] and meta['sha256'] == sha(ck)
        for label, root, receipt in [('A1',p,r),('A0',bp,br)]:
            devp = root/'dev_30000/DEV.jsonl'
            assert sha(devp) == receipt['dev_sha256' if label == 'A1' else 'dev_long_sha256']
            rr, m = evaluate(devp, 'DEV', 320 if ds == 'bf' else 740)
            assert m['scenes'] == (32 if ds == 'bf' else 72)
            assert len({x['core'] for x in rr}) == (32 if ds == 'bf' else 74)
            for k in ('success9','success42','median_err'):
                assert abs(m[k]-receipt['dev' if label == 'A1' else 'dev_long']['COS'][k]) < 1e-12
            ids = identities(rr)
            if (ds,'DEV') in eval_ids: assert ids == eval_ids[(ds,'DEV')]
            eval_ids[(ds,'DEV')] = ids
            u['metrics'][label] = m
            for core in sorted({x['core'] for x in rr}):
                cr = [x for x in rr if x['core'] == core]
                per_core.append(dict(ds=ds, arm=arm, seed=1001, condition=label,core=core,
                                     scene=cr[0]['scene'], **metrics(cr)))
            u['cal'][label] = {}
            for stop in STEPS:
                cr, cm = evaluate(root/f'cal_{stop}/CAL.jsonl','CAL',80 if ds == 'bf' else 100)
                assert cm['scenes'] == (8 if ds == 'bf' else 9)
                ids = identities(cr)
                if (ds,'CAL') in eval_ids: assert ids == eval_ids[(ds,'CAL')]
                eval_ids[(ds,'CAL')] = ids
                for k in ('success9','success42','median_err'):
                    assert abs(cm[k]-receipt['cal'][str(stop)]['COS'][k]) < 1e-12
                u['cal'][label][str(stop)] = cm
                curves.append(dict(ds=ds, arm=arm, seed=1001, condition=label,step=stop, **cm))
        u['A1_minus_A0'] = difference(u['metrics']['A1'], u['metrics']['A0'])
        d = u['A1_minus_A0']
        u['registered_descriptive_signal'] = d['success9'] >= .01 or (d['success42'] >= .01 and d['success9'] >= -.01)
        u['fit_curves'] = {label:{str(k):dict(loss=S.mean(lb[k]),grad=S.mean(gb[k])) for k in lb}
                           for label,lb,gb in [('A1',loss_bins,grad_bins),('A0',baseline_loss_bins,baseline_grad_bins)]}
        units[key] = u
    if check_only:
        out = dict(status='COMPLETED_UNITS_AUDIT_PASS',utc=datetime.now(timezone.utc).isoformat(),
                   complete=list(units),missing=missing,freeze_sha256=freeze,analyzer_sha256=sha(Path(__file__)))
        dump(HERE/'A1_ANALYZER_CHECK_20261025.json',out); print(json.dumps(out)); return
    assert len(units) == 6
    interactions = {}
    for ds in ('bf','epfl'):
        interactions[ds] = {}
        for arm in ('bce','nce'):
            exact, ctrl = units[f'{ds}_vcs_exact_1001'], units[f'{ds}_{arm}_1001']
            diffs = {c:difference(exact['metrics'][c],ctrl['metrics'][c]) for c in ('A0','A1')}
            diffs['interaction'] = difference(exact['A1_minus_A0'],ctrl['A1_minus_A0'])
            interactions[ds][arm] = diffs
    out = dict(status='AUDIT_PASS',utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=freeze,
        analyzer_sha256=sha(Path(__file__)),units=units,exact_minus_control=interactions,
        technical_gates=gates,input_matching='27000 original metadata batches/LR matched A0; resampled tensors intentionally differ. Across objectives each dataset has identical geometry plans.',
        units_note='Success rates are scene-equal fractions; error columns use err_full. EPFL native error is err_full/4. Pooled medians and p95 are descriptive, not scene-equal means.',
        cost=dict(new_receipt_gpu_hours=sum(u['seconds'] for u in units.values())/3600,
            shared_gate_gpu_hours=sum(g['seconds'] for g in gates)/3600,
            reused_a0_gpu_hours=sum(u['baseline_seconds'] for u in units.values())/3600,
            common_prefix_gpu_hours=sum(u['prefix_seconds'] for u in units.values())/3600,
            note='A0/prefix previously paid, not new cost; gates counted once. STARTED attempt markers are not failure: frozen runner only writes COMPLETE separately. CPU/scheduler startup and unrelated historical failed KD diagnostics excluded, never assumed zero.'),
        limitations=['One pilot seed per dataset/objective, not cross-seed confirmation.',
            'A0/A1 evaluations are separate jobs with nondeterministic pose search; environment differences recorded.',
            'Common unadapted3000 prefix; only subsequent27000 updates adapted.',
            'Repeated DEV/CAL development data; no TEST access or new confirmatory claim.',
            'Threshold is descriptive and not permanent prohibition; no in-flight or post-hoc recipe change.'])
    dump(HERE/'ANALYSIS_A1_ALL6_20261025.json',out)
    for name,data in [('A1_ALL6_CAL_CURVES_20261025.csv',curves),('A1_ALL6_PER_CORE_20261025.csv',per_core)]:
        with (HERE/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['A1六项完整配对审计（seed1001；固定30000步；开发证据）',
           'A0复用六项完成的干净长训练；共同3000步前缀，A1只改后27000步输入重采样。',
           'CPU、两GPU门禁、源/冻结/27000批元数据与LR、跨臂几何计划、CAL/DEV真值与搜索预算均通过。',
           '以下严格/宽松差值单位pp；误差为协议err_full，EPFL原生误差需除4。']
    for k,u in units.items():
        d=u['A1_minus_A0'];a=u['metrics']['A1'];b=u['metrics']['A0']
        lines.append(f'{k}: A0 {100*b["success9"]:.4f}/{100*b["success42"]:.4f}%, A1 {100*a["success9"]:.4f}/{100*a["success42"]:.4f}%; Δ {100*d["success9"]:+.4f}/{100*d["success42"]:+.4f}pp; medianΔ {d["median_err"]:+.4f}; signal={u["registered_descriptive_signal"]}.')
    lines += ['所有逐core尾部及CAL曲线另附CSV；不同数据集不合并。',
              '仅一个seed；需要独立匹配扩种才能评估稳定性。没有基于途中结果停止、调λ或改终点。',
              '成本：'+json.dumps(out['cost'],ensure_ascii=False)]
    (HERE/'ANALYSIS_A1_ALL6_20261025.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=out['status'],deltas={k:u['A1_minus_A0'] for k,u in units.items()},cost=out['cost']),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-completed',action='store_true')
    main(parser.parse_args().check_completed)
