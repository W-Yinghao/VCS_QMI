"""Pose v2 analysis (PROTOCOL.json): 18 runs vs long-v2 30k references and v1 pose runs, DEV COS at step 30000. WAITING until all 18 COMPLETE."""
import hashlib, json, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; H = HERE.parents[0]
LONG, V1 = H / 'r22_geometry_long_v2_20261024/runs', H / 'r22_reg_pose_20261025/runs'
SEEDS, ARMS = (1001, 1002, 1003), ('pose2_only', 'pose2_vcs', 'pose2_bce')
V1_ARM = {'pose2_only': 'pose_only', 'pose2_vcs': 'pose_vcs', 'pose2_bce': 'pose_bce'}
done = [(ds, a, s) for ds in ('bf', 'epfl') for a in ARMS for s in SEEDS if (HERE / 'runs' / f'{ds}_{a}_{s}' / 'COMPLETE.json').exists()]
if len(done) < 18:
    print(json.dumps(dict(status='WAITING', complete=len(done), expected=18))); sys.exit(0)


def attribution(p):
    f9 = [r['E']['COS'] for r in map(json.loads, open(p)) if r['E']['COS']['err_full'] >= 9]
    return dict(fail9=len(f9), energy_prefers_wrong=sum(e['F_hat'] >= e['F_true'] for e in f9))


R = {}; COST = 0.; GPUS = set(); RATES = {}
for ds in ('bf', 'epfl'):
    for s in SEEDS:
        for a in ARMS:
            d = HERE / 'runs' / f'{ds}_{a}_{s}'; c = json.loads((d / 'COMPLETE.json').read_text()); p = d / 'dev_30000/DEV.jsonl'
            assert c['status'] == 'COMPLETE' and c['steps'] == 30000 and set(c['roles_read']) <= {'FIT', 'CAL', 'DEV'}
            assert hashlib.sha256(p.read_bytes()).hexdigest() == c['dev_sha256']
            R[(ds, a, s)] = dict(**c['dev_30000']['COS'], **attribution(p)); COST += c['seconds'] / 3600; GPUS.add(c['gpu']); RATES[(ds, a, s)] = c['last_rates']
            v = V1 / f'{ds}_{V1_ARM[a]}_{s}'; cv = json.loads((v / 'COMPLETE.json').read_text())
            R[(ds, 'v1_' + V1_ARM[a], s)] = dict(**cv['dev_30000']['COS'], **attribution(v / 'dev_30000/DEV.jsonl'))
        for ref in ('vcs_exact', 'bce', 'nce'):
            c = json.loads((LONG / f'{ds}_{ref}_{s}' / 'COMPLETE.json').read_text())
            R[(ds, 'ref_' + ref, s)] = dict(**c['dev_long']['COS'], **attribution(LONG / f'{ds}_{ref}_{s}' / 'dev_30000/DEV.jsonl'))


def contrast(ds, a, b):
    out = {}
    for k in ('success9', 'success42', 'median_err'):
        v = [(R[(ds, a, s)][k] - R[(ds, b, s)][k]) * (100 if k != 'median_err' else 1) for s in SEEDS]
        out[k] = dict(values=[round(x, 3) for x in v], mean=round(st.mean(v), 3), positive=sum(x > 0 for x in v))
    out['signal'] = out['success9']['mean'] >= 1 and out['success9']['positive'] >= 2 and out['success42']['mean'] >= -.5
    return out


PAIRS = [('pose2_vcs', 'ref_vcs_exact'), ('pose2_vcs', 'pose2_bce'), ('pose2_only', 'v1_pose_only'), ('pose2_vcs', 'v1_pose_vcs'),
         ('pose2_bce', 'v1_pose_bce'), ('pose2_only', 'pose2_vcs'), ('pose2_bce', 'ref_bce'), ('pose2_only', 'ref_nce'), ('pose2_vcs', 'ref_nce')]
GROUPS = list(ARMS) + ['v1_pose_only', 'v1_pose_vcs', 'v1_pose_bce', 'ref_vcs_exact', 'ref_bce', 'ref_nce']
res = {ds: {f'{a}-{b}': contrast(ds, a, b) for a, b in PAIRS} for ds in ('bf', 'epfl')}
means = {ds: {g: {k: round(st.mean(R[(ds, g, s)][k] for s in SEEDS) * (100 if k != 'median_err' else 1), 3) for k in ('success9', 'success42', 'median_err')}
              | {'fail9': sum(R[(ds, g, s)]['fail9'] for s in SEEDS), 'energy_prefers_wrong': sum(R[(ds, g, s)]['energy_prefers_wrong'] for s in SEEDS)}
              for g in GROUPS} for ds in ('bf', 'epfl')}
rates = {ds: {a: {k: round(st.mean(RATES[(ds, a, s)][k] for s in SEEDS), 4) for k in ('truth_beats_ring', 'basin_beats_all')} for a in ARMS} for ds in ('bf', 'epfl')}
(HERE / 'ANALYSIS.json').write_text(json.dumps(dict(status='COMPLETE', means=means, contrasts=res, train_last1000_rates=rates,
                                                    receipt_gpu_hours=round(COST, 3), new_run_gpus=sorted(GPUS)), indent=1))
for ds in means:
    print('==', ds)
    for g, v in means[ds].items(): print(f'  {g:14s}', v)
    for n, c in res[ds].items(): print(f'  {n:28s} s9 {c["success9"]["mean"]:+7.2f} ({c["success9"]["positive"]}/3) {c["success9"]["values"]}  s42 {c["success42"]["mean"]:+6.2f}  {"SIGNAL" if c["signal"] else ""}')
    print('  train rates', rates[ds])
print('cost GPUh', round(COST, 2), sorted(GPUS))
