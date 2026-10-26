"""A2 SHARP completion analysis: 3 SHARP arms x seeds 901 (existing) + 902/903 (new), against original A arms and hinge controls."""
import hashlib, json, statistics as st
from pathlib import Path
B = Path('/home/infres/yinwang/CS_QMI'); P = B / 'handoff/r22_a2_complete_v3_20261025'
NEW = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r22_a2_complete_v3_20261025/runs')
lock = json.loads((P / 'CONTROL_LOCK.json').read_text()); SEEDS = (901, 902, 903)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def curve(log):
    e = log['epochs']; return [x['dev']['rsum'] for x in e]
def metrics(c):
    best = max(c); return dict(best=best, best_epoch=c.index(best) + 1, final=c[-1], last5=st.mean(c[-5:]), last10=st.mean(c[-10:]), n_epochs=len(c))
runs = {}
for s in SEEDS:
    for arm in ('VCS_SHARP', 'VCSW_SHARP', 'BCE_SHARP'):
        d = NEW / f's{s}__{arm}' if s != 901 else Path(lock['units'][f'901_{arm}']['directory'])
        u = json.loads((d / 'unit_result.json').read_text()); assert u['status'] == 'COMPLETE_VERIFIED' and not u['test_read']
        if s != 901: assert (d / 'COMPLETE.json').exists()
        runs[(arm, s)] = dict(c=curve(json.loads((d / 'train_log.json').read_text())), env=u['environment'], init=u['initial_model_sha256'], order=u['data_order_sha256'])
    for arm in ('VCS', 'VCSW', 'BCE', 'NCE'):
        d = Path(lock['units'][f'{s}_{arm}']['directory']); runs[(arm, s)] = dict(c=curve(json.loads((d / 'train_log.json').read_text())))
    for key, name in (('N2M_ConstantU', 'hinge_ConstantU'), ('N2M_VCS', 'hinge_VCSweight_N2M'), ('N2F_Unweighted_A', 'hinge_Unweighted_A'), ('N2F_VCS_Weight', 'hinge_VCSweight_N2F')):
        files = lock['hinge_metadata'][f'{key}_{s}']; tl = [f for f in files if f.endswith('train_log.json')][0]
        assert sha(tl) == files[tl]; runs[(name, s)] = dict(c=curve(json.loads(Path(tl).read_text())))
for s in SEEDS:  # same seed: identical initialization and data order across the three SHARP arms
    assert len({runs[(a, s)]['init'] for a in ('VCS_SHARP', 'VCSW_SHARP', 'BCE_SHARP')}) == 1
    assert len({runs[(a, s)]['order'] for a in ('VCS_SHARP', 'VCSW_SHARP', 'BCE_SHARP')}) == 1
M = {k: metrics(v['c']) for k, v in runs.items()}
def contrast(a, b):
    out = {}
    for k in ('best', 'final', 'last5', 'last10'):
        v = [M[(a, s)][k] - M[(b, s)][k] for s in SEEDS]
        out[k] = dict(values=[round(x, 3) for x in v], mean=round(st.mean(v), 3), positive=sum(x > 0 for x in v),
                      new_902_903_same_sign=(v[1] > 0) == (v[2] > 0))
    b = out['best']; v = b['values']; sign = b['mean'] > 0
    # protocol: |mean| >= 3 rSum, >= 2/3 seeds in the mean's direction, and both new seeds 902/903 in that direction
    b['descriptive_signal'] = abs(b['mean']) >= 3 and sum((x > 0) == sign for x in v) >= 2 and all((x > 0) == sign for x in v[1:])
    return out
res = {}
for a, b_name in (('VCS_SHARP', 'BCE_SHARP'), ('VCSW_SHARP', 'VCS_SHARP'), ('VCSW_SHARP', 'BCE_SHARP'), ('VCS_SHARP', 'VCS'), ('VCSW_SHARP', 'VCSW'), ('BCE_SHARP', 'BCE'),
                  ('VCS_SHARP', 'hinge_ConstantU'), ('VCSW_SHARP', 'hinge_VCSweight_N2M'), ('BCE_SHARP', 'hinge_ConstantU'), ('VCS_SHARP', 'NCE')):
    res[f'{a}-{b_name}'] = contrast(a, b_name)
means = {a: {k: round(st.mean(M[(a, s)][k] for s in SEEDS), 3) for k in ('best', 'final', 'last5')} for a in sorted({k[0] for k in M})}
envs = {f'{a}_{s}': {k: runs[(a, s)]['env'].get(k) for k in ('gpu', 'driver', 'host')} for a in ('VCS_SHARP', 'VCSW_SHARP', 'BCE_SHARP') for s in SEEDS}
out = dict(status='COMPLETE', means=means, per_unit={f'{a}_{s}': M[(a, s)] for (a, s) in M}, contrasts=res, environments=envs,
           primary=['VCS_SHARP-BCE_SHARP', 'VCSW_SHARP-VCS_SHARP'])
(P / 'analysis/ANALYSIS_A2_SHARP_ALL3.json').write_text(json.dumps(out, indent=1))
for a, v in means.items(): print(f'{a:22s}', v)
for n in res: print(n, {k: (res[n][k]['mean'], res[n][k]['positive'], res[n][k]['values']) for k in ('best', 'last5')}, 'signal' if res[n]['best']['descriptive_signal'] else '')
print({k: v for k, v in envs.items()})
