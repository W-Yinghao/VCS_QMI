"""KD relation study: 9 new DKD+relation runs vs reused DKD-only and DKD+logit-cosine controls (TRAINING_PROTOCOL.json)."""
import hashlib, json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent; BASE = HERE.parents[1]; OLD = BASE / 'handoff/r22_other_apps_20261023/runs'
SEEDS = (31415, 16180, 57721)
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze = sha(HERE / 'FREEZE.json'); arms = {}
for s in SEEDS:
    for m in ('direct', 'vcs', 'bce'):
        d = HERE / 'runs' / f'kd_rel2_{m}_dkd_s{s}'; c = read(d / 'COMPLETE.json')
        assert c['status'] == 'COMPLETE' and c['freeze_sha256'] == freeze and c['epochs'] == 240 and not c['test_read']
        assert sha(d / 'history.json') == c['history_sha256']
        arms[(m, s)] = dict(hist=read(d / 'history.json'), gpu=c['gpu'], seconds=c['seconds'], job=c['job'])
    for m, name in (('dkd', f'kd_full_dkd_none_l0_s{s}'), ('dkd_cosine', f'kd_full_dkd_cosine_l10_s{s}')):
        c = read(OLD / name / 'COMPLETE.json'); arms[(m, s)] = dict(hist=read(OLD / name / 'history.json'), gpu=c['gpu'])
for s in SEEDS:
    ref = [h['sample_order_sha256'] for h in arms[('dkd', s)]['hist']]
    for m in ('direct', 'vcs', 'bce', 'dkd_cosine'): assert [h['sample_order_sha256'] for h in arms[(m, s)]['hist']] == ref
    assert len({arms[(m, s)]['gpu'] for m in ('direct', 'vcs', 'bce', 'dkd', 'dkd_cosine')}) == 1
def metrics(h):
    t = [x['cal']['top1'] * 100 for x in h]
    return dict(final=t[-1], last10=st.mean(t[-10:]), best=max(t))
M = {k: metrics(v['hist']) for k, v in arms.items()}
contrasts = {}
for a, b in (('direct', 'dkd'), ('vcs', 'dkd'), ('bce', 'dkd'), ('vcs', 'direct'), ('vcs', 'bce'), ('vcs', 'dkd_cosine'), ('direct', 'dkd_cosine'), ('bce', 'dkd_cosine')):
    contrasts[f'{a}-{b}'] = {k: dict(values=[round(M[(a, s)][k] - M[(b, s)][k], 4) for s in SEEDS]) for k in ('final', 'last10', 'best')}
    for k, v in contrasts[f'{a}-{b}'].items():
        v['mean'] = round(st.mean(v['values']), 4); v['positive'] = sum(x > 0 for x in v['values']); v['signal'] = v['mean'] >= .2 and v['positive'] >= 2
means = {m: {k: round(st.mean(M[(m, s)][k] for s in SEEDS), 4) for k in ('final', 'last10', 'best')} for m in ('dkd', 'dkd_cosine', 'direct', 'vcs', 'bce')}
diag = {f'{m}_{s}': dict(relation_positive_cosine=arms[(m, s)]['hist'][-1]['dependence'].get('relation_positive_cosine'),
                         raw_j=arms[(m, s)]['hist'][-1]['dependence'].get('raw_j')) for m in ('direct', 'vcs', 'bce') for s in SEEDS}
cost = sum(arms[(m, s)]['seconds'] for m in ('direct', 'vcs', 'bce') for s in SEEDS) / 3600
out = dict(status='COMPLETE', freeze_sha256=freeze, means=means, contrasts=contrasts, final_epoch_relation_diagnostics=diag,
           new_receipt_gpu_hours=round(cost, 3), gpus={s: arms[('dkd', s)]['gpu'] for s in SEEDS},
           checks='9 COMPLETE records/freeze/history hashes; per seed identical 240-epoch sample order and GPU model across all 5 arms')
(HERE / 'ANALYSIS.json').write_text(json.dumps(out, indent=1))
print(json.dumps(dict(means=means, gpus=out['gpus'], cost=out['new_receipt_gpu_hours']), indent=0))
for n, c in contrasts.items(): print(n, {k: (v['mean'], v['positive'], v['values'], 'SIGNAL' if v['signal'] else '') for k, v in c.items()})
