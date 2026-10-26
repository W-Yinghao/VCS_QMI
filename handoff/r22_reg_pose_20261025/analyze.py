"""Registration pose-term analysis (PROTOCOL.json): 18 runs vs long-v2 30k references, DEV COS at step 30000."""
import glob, hashlib, json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent; LONG = HERE.parents[0] / 'r22_geometry_long_v2_20261024/runs'
SEEDS = (1001, 1002, 1003); R = {}
def dev_rows(p): return [json.loads(l) for l in open(p)]
def attribution(rows):
    f9 = [r['E']['COS'] for r in rows if r['E']['COS']['err_full'] >= 9]
    return dict(fail9=len(f9), energy_prefers_wrong=sum(e['F_hat'] >= e['F_true'] for e in f9))
for ds in ('bf', 'epfl'):
    for s in SEEDS:
        for arm in ('pose_only', 'pose_vcs', 'pose_bce'):
            d = HERE / 'runs' / f'{ds}_{arm}_{s}'; c = json.loads((d / 'COMPLETE.json').read_text())
            assert c['status'] == 'COMPLETE' and c['steps'] == 30000 and set(c['roles_read']) <= {'FIT', 'CAL', 'DEV'}
            devp = glob.glob(str(d / 'dev_30000' / 'DEV.jsonl'))[0]; assert hashlib.sha256(open(devp, 'rb').read()).hexdigest() == c['dev_sha256']
            R[(ds, arm, s)] = dict(**c['dev_30000']['COS'], **attribution(dev_rows(devp)), gpu=c['gpu'])
        for ref in ('vcs_exact', 'bce', 'nce'):
            c = json.loads((LONG / f'{ds}_{ref}_{s}' / 'COMPLETE.json').read_text())
            R[(ds, 'ref_' + ref, s)] = dict(**c['dev_long']['COS'], **attribution(dev_rows(LONG / f'{ds}_{ref}_{s}' / 'dev_30000/DEV.jsonl')), gpu=c['gpu'])
def contrast(ds, a, b):
    out = {}
    for k in ('success9', 'success42', 'median_err'):
        v = [(R[(ds, a, s)][k] - R[(ds, b, s)][k]) * (100 if k != 'median_err' else 1) for s in SEEDS]
        out[k] = dict(values=[round(x, 3) for x in v], mean=round(st.mean(v), 3), positive=sum(x > 0 for x in v))
    out['signal'] = out['success9']['mean'] >= 1 and out['success9']['positive'] >= 2 and out['success42']['mean'] >= -.5
    return out
res = {ds: {f'{a}-{b}': contrast(ds, a, b) for a, b in (('pose_only', 'ref_vcs_exact'), ('pose_vcs', 'pose_only'), ('pose_vcs', 'pose_bce'),
           ('pose_bce', 'ref_bce'), ('pose_vcs', 'ref_vcs_exact'), ('pose_only', 'ref_nce'))} for ds in ('bf', 'epfl')}
means = {ds: {a: {k: round(st.mean(R[(ds, a, s)][k] for s in SEEDS) * (100 if k != 'median_err' else 1), 3) for k in ('success9', 'success42', 'median_err')} | {'fail9': sum(R[(ds, a, s)]['fail9'] for s in SEEDS), 'energy_prefers_wrong': sum(R[(ds, a, s)]['energy_prefers_wrong'] for s in SEEDS)}
         for a in ('pose_only', 'pose_vcs', 'pose_bce', 'ref_vcs_exact', 'ref_bce', 'ref_nce')} for ds in ('bf', 'epfl')}
cost = sum(json.loads((HERE / 'runs' / f'{ds}_{a}_{s}' / 'COMPLETE.json').read_text())['seconds'] for ds in ('bf', 'epfl') for a in ('pose_only', 'pose_vcs', 'pose_bce') for s in SEEDS) / 3600
gpus = sorted({R[k]['gpu'] for k in R if not k[1].startswith('ref')})
(HERE / 'ANALYSIS.json').write_text(json.dumps(dict(status='COMPLETE', means=means, contrasts=res, receipt_gpu_hours=round(cost, 3), new_run_gpus=gpus), indent=1))
for ds in means:
    print('==', ds)
    for a, v in means[ds].items(): print(f'  {a:14s}', v)
    for n, c in res[ds].items(): print(f'  {n:26s} s9 {c["success9"]["mean"]:+7.2f} ({c["success9"]["positive"]}/3) {c["success9"]["values"]}  s42 {c["success42"]["mean"]:+6.2f}  {"SIGNAL" if c["signal"] else ""}')
print('cost GPUh', round(cost, 2), gpus)
