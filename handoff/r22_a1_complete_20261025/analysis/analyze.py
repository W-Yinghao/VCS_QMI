"""A1 completion analysis: new matched seeds 1002/1003 (12 A1 vs reused A0 long-v2 30k) and a separate 3-seed descriptive table with pilot 1001."""
import hashlib, json, statistics as st
from pathlib import Path
B = Path('/home/infres/yinwang/CS_QMI'); P = B / 'handoff/r22_a1_complete_20261025'
R = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/r22_a1_complete_20261025/runs')
lock = json.loads((P / 'SOURCE_LOCK.json').read_text())['units']
pilot = json.loads((B / 'handoff/r22_watch10m_20261024/ANALYSIS_A1_ALL6_20261025.json').read_text())['units']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
U = {}
for ds in ('bf', 'epfl'):
    for arm in ('vcs_exact', 'bce', 'nce'):
        for s in (1002, 1003):
            c = json.loads((R / f'{ds}_{arm}_{s}' / 'COMPLETE.json').read_text())
            assert c['status'] == 'COMPLETE' and c['final_steps'] == 30000 and set(c['roles_read']) <= {'FIT', 'CAL', 'DEV'}
            assert sha(R / f'{ds}_{arm}_{s}' / 'dev_30000' / 'DEV.jsonl') == c['dev_sha256'] if (R / f'{ds}_{arm}_{s}' / 'dev_30000' / 'DEV.jsonl').exists() else True
            a0 = json.loads(Path(lock[f'{ds}_{arm}_{s}']['a0_completion']).read_text())['dev_long']['COS']
            U[(ds, arm, s)] = dict(A1=c['dev']['COS'], A0=a0, env_matched=c.get('environment_matched'), host=c['environment'].get('host'))
        m = pilot[f'{ds}_{arm}_1001']['metrics']; U[(ds, arm, 1001)] = dict(A1=m['A1'], A0=m['A0'], env_matched=None, host=pilot[f'{ds}_{arm}_1001']['a1_environment']['host'])
def diff(ds, arm, s, k): return (U[(ds, arm, s)]['A1'][k] - U[(ds, arm, s)]['A0'][k]) * 100
out = {'new_matched_1002_1003': {}, 'all3_descriptive': {}, 'interaction_new': {}}
for ds in ('bf', 'epfl'):
    for arm in ('vcs_exact', 'bce', 'nce'):
        d9 = [diff(ds, arm, s, 'success9') for s in (1002, 1003)]; d42 = [diff(ds, arm, s, 'success42') for s in (1002, 1003)]
        sig = (all(x > 0 for x in d9) and st.mean(d9) >= 1) or (all(x > 0 for x in d42) and st.mean(d42) >= 1 and min(d9) >= -1)
        out['new_matched_1002_1003'][f'{ds}_{arm}'] = dict(strict=[round(x, 3) for x in d9], strict_mean=round(st.mean(d9), 3), loose=[round(x, 3) for x in d42],
                                                           loose_mean=round(st.mean(d42), 3), descriptive_signal=sig, env_matched=[U[(ds, arm, s)]['env_matched'] for s in (1002, 1003)])
        a9 = [diff(ds, arm, s, 'success9') for s in (1001, 1002, 1003)]; a42 = [diff(ds, arm, s, 'success42') for s in (1001, 1002, 1003)]
        out['all3_descriptive'][f'{ds}_{arm}'] = dict(strict=[round(x, 3) for x in a9], strict_mean=round(st.mean(a9), 3), loose=[round(x, 3) for x in a42], loose_mean=round(st.mean(a42), 3),
                                                      A1_strict_mean=round(st.mean(U[(ds, arm, s)]['A1']['success9'] * 100 for s in (1001, 1002, 1003)), 3),
                                                      A0_strict_mean=round(st.mean(U[(ds, arm, s)]['A0']['success9'] * 100 for s in (1001, 1002, 1003)), 3))
    for ctl in ('bce', 'nce'):
        v = [diff(ds, 'vcs_exact', s, 'success9') - diff(ds, ctl, s, 'success9') for s in (1002, 1003)]
        out['interaction_new'][f'{ds}_vcs_minus_{ctl}'] = dict(strict=[round(x, 3) for x in v], mean=round(st.mean(v), 3))
(P / 'analysis/ANALYSIS_A1_COMPLETE.json').write_text(json.dumps(out, indent=1))
for sec, d in out.items():
    print('==', sec)
    for k, v in d.items(): print(' ', k, v)
