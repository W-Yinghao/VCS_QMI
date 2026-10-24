"""Explicitly registered duration validation, independent of the short-pilot gate."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OLD = BASE / 'handoff/r22_other_apps_20261023'
VERIFIED = False
HISTORIES = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def validate():
    global VERIFIED
    if not VERIFIED:
        for rel, digest in read(HERE / 'FREEZE.json')['files'].items():
            assert sha(BASE / rel) == digest, rel
        protocol = read(HERE / 'PROTOCOL.json')
        original = protocol['original_training_freeze_sha256']
        assert sha(OLD / 'FREEZE.json') == original
        jobs = read(HERE / 'JOBS.json')
        assert len(jobs) == len({j['name'] for j in jobs}) == 24
        for anchor in ('ce', 'dkd'):
            decision = read(OLD / f'DECISION_KD_{anchor.upper()}.json')
            assert decision['eligible'] is False  # Historical decision remains unchanged.
            for seed in (31415, 16180, 57721):
                group = [j for j in jobs if j['anchor'] == anchor and j['seed'] == seed]
                assert {j['method'] for j in group} == {'none', 'cosine', 'vcs', 'bce'}
                assert len({j['partition_locked'] for j in group}) == 1
                for j in group:
                    assert j['weight'] == decision['selected'][j['method']]['weight']
                    assert j['freeze_sha256'] == original
                    cfg = read(OLD / 'configs' / (j['config_name'] + '.json'))
                    assert cfg['training']['epochs'] == 240
                    assert cfg['optimizer']['milestones'] == [150, 180, 210]
                    assert cfg['loss']['dkd_warmup'] == 20
                    assert '--partition=' + j['partition_locked'] in j['args']
        VERIFIED = True
    return read(HERE / 'JOBS.json')


def admit(add_manifest):
    jobs = validate()
    add_manifest(HERE / 'JOBS.json')
    path = HERE / 'ADMISSION.json'
    if not path.exists():
        write(path, dict(status='ADMITTED', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         controller_job=os.environ.get('SLURM_JOB_ID'), units=24,
                         amendment_freeze_sha256=sha(HERE / 'FREEZE.json'),
                         scientific_training_freeze_sha256=jobs[0]['freeze_sha256'],
                         reason='User-requested long-training validation after observing short-pilot outcomes; not a passed short-pilot gate.'))
    return jobs


def endpoints(history, best):
    assert len(history) == 240 and [r['epoch'] for r in history] == list(range(240))
    return dict(final_top1=history[-1]['cal']['top1'],
                mean_last10_top1=statistics.mean(r['cal']['top1'] for r in history[-10:]),
                best_top1=best['top1'], final_nll=history[-1]['cal']['nll'],
                curve={str(e): history[e-1]['cal']['top1'] for e in (40, 80, 120, 150, 180, 210, 240)})


def summarize(done):
    jobs = validate()
    units = {}
    for j in jobs:
        if j['name'] not in done:
            continue
        r = done[j['name']]
        folder = Path(j['completion']).parent
        key = (j['name'], r['history_sha256'])
        if key not in HISTORIES:
            assert sha(folder / 'history.json') == r['history_sha256']
            HISTORIES[key] = read(folder / 'history.json')
        units[j['name']] = dict(anchor=j['anchor'], method=j['method'], seed=j['seed'],
                               weight=j['weight'], gpu=r['gpu'], seconds=r['seconds'],
                               endpoints=endpoints(HISTORIES[key], r['metrics']))
    pairs = {}
    for anchor in ('ce', 'dkd'):
        pairs[anchor] = {}
        for control in ('none', 'cosine', 'bce'):
            compared = []
            for seed in (31415, 16180, 57721):
                group = [j for j in jobs if j['anchor'] == anchor and j['seed'] == seed]
                v = next(j for j in group if j['method'] == 'vcs')
                b = next(j for j in group if j['method'] == control)
                if v['name'] not in units or b['name'] not in units:
                    continue
                vr, br = done[v['name']], done[b['name']]
                assert vr['student_initial_sha256'] == br['student_initial_sha256']
                assert vr['sample_order_sha256'] == br['sample_order_sha256']
                assert vr['gpu'] == br['gpu'], 'Within-seed comparison must use the same GPU model'
                ve, be = units[v['name']]['endpoints'], units[b['name']]['endpoints']
                compared.append(dict(seed=seed, gpu=vr['gpu'],
                    difference_pp={k: (ve[k]-be[k])*100 for k in ('final_top1', 'mean_last10_top1', 'best_top1')},
                    curve_difference_pp={e: (ve['curve'][e]-be['curve'][e])*100 for e in ve['curve']}))
            stats = {}
            for endpoint in ('final_top1', 'mean_last10_top1', 'best_top1'):
                values = [r['difference_pp'][endpoint] for r in compared]
                stats[endpoint] = dict(mean_pp=statistics.mean(values) if values else None,
                                       sd_pp=statistics.stdev(values) if len(values) > 1 else None,
                                       positive=sum(v > 0 for v in values), n=len(values))
            pairs[anchor]['vcs-minus-' + control] = dict(per_seed=compared, statistics=stats)
    result = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), completed=len(units), expected=24,
                  label='Training-duration validation; exploratory CAL evidence after short-pilot selection',
                  primary='Fixed epoch240 top1; paired differences over all3 seeds',
                  secondary=['mean last10epochs', 'best CAL checkpoint', 'learning curves', 'teacher transfer diagnostics in parent SUMMARY'],
                  units=units, pairs=pairs, amendment_freeze_sha256=sha(HERE / 'FREEZE.json'),
                  limitations=['Lambdas selected using the prior40epoch pilot, not a search for the long-training optimum.',
                               'Long runs use new seeds; duration and seed effects are not isolated by the short-vs-long comparison.',
                               'Only FIT/CAL;3seeds and repeated CAL selection do not constitute independent confirmation.'])
    write(HERE / 'SUMMARY.json', result)
    (HERE / 'LATEST_CN.txt').write_text(
        f"更新：{result['utc']}\n240epoch验证完成 {len(units)}/24；原40epoch筛选结论保留为未通过。\n"
        "两种anchor×基线/余弦/VCS/BCE×3seed；每seed内固定同型GPU，从头训练，λ不再调。\n"
        "主指标为第240轮CAL准确率，附最后10轮、最佳checkpoint、完整曲线与配对差值。\n")
    return result
