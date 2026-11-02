#!/usr/bin/env python3
"""R37S smoke check: reads the smoke directory written by sbatch/r37s_smoke.sbatch; writes <smoke>/SMOKE_CHECK.json and a copy
SMOKE_CHECK_<job>.json in the line directory."""
from pathlib import Path
import glob, hashlib, json, os, sys, time
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code_r37s'))
import r30_state as ST

S = Path(sys.argv[1])
res = {}


def ok(name, cond, **info):
    res[name] = dict(ok=bool(cond), **info)
    print(('PASS ' if cond else 'FAIL ') + name, json.dumps(info, default=str)[:400], flush=True)


def jl(p):
    return json.loads(Path(p).read_text())


def run_json(d):
    return [jl(p) for p in sorted(glob.glob(str(d / 'RUN_*.json')))]


P = S / 'parents' / 'wp_cc_s1001_H_e5'
pr = jl(P / 'PARENT.json')
w = pr['warmup_dev']
ok('parent_boundary', pr['epoch'] == -1 and len(w) == 2 and all(x['n_images'] == 1000 and x['n_captions'] == 1000 for x in w)
   and sorted(pr['data']['files']) == ['dev_caps.tsv', 'dev_ims.npy', 'train_caps.tsv', 'train_ims.npy'] and pr['data']['noise_digest'] is None
   and pr['identity']['data_name'] == 'cc152k_precomp' and pr['identity']['k'] == 5 and pr['identity']['i2t'] and pr['identity']['t2i']
   and pr['identity']['warmup_epoch'] == 2 and pr['identity']['noise_ratio'] == 0,
   warmup_rsum=[x['rsum'] for x in w], gpu=pr['gpu'], sha256=pr['sha256'])
ro = (os.stat(P / 'parent_state.pt').st_mode & 0o222) == 0
ok('parent_readonly', ro)

C, X = S / 'runs' / 'cont', S / 'runs' / 'split'
for name, d in (('cont', C), ('split', X)):
    fr = jl(d / 'FORK_REPORT.json')
    ok(f'{name}_fork_identity', fr['ok'] and fr['components_equal'] and fr['parent_fingerprint'] == pr['training_state_fingerprint'],
       strict=fr['strict_differences'], unclassified=fr['unclassified_differences'])
    e1 = jl(d / 'E1_RECORD.json')
    ok(f'{name}_milestone_E1', e1['recomputed_equal'] and e1['rng_unchanged_by_rank_pass'] and (d / 'state_completed1.pt').exists()
       and (d / 'RANKS_completed1.npz').exists(), dev=e1['dev']['rsum'], recomputed=e1['recomputed']['rsum'])

h = jl(C / 'EPOCHS.json')
co = h['cotrain']
ok('cont_two_cotrain_epochs_done', len(co) == 2 and jl(C / 'DONE.json')['status'] == 'COMPLETE' and (C / 'model_best.pth.tar').exists()
   and co[0].get('completed_cotrain_epochs') == 1 and co[1].get('completed_cotrain_epochs') == 2,
   rsum=[e['dev']['rsum'] for e in co], is_best=[e['is_best'] for e in co])
br = {f'e{e["epoch"]}_{p}': {k: e['train'][p].get(k, 0) for k in ('i2t_usable_steps', 't2i_usable_steps', 'i2t_no_proto', 't2i_no_proto', 'steps')}
      for e in co for p in ('A', 'B')}
ok('a1_branches_i2t_t2i_used', all(br[f'e1_{p}']['i2t_usable_steps'] > 0 and br[f'e1_{p}']['t2i_usable_steps'] > 0 for p in ('A', 'B'))
   and br['e0_B']['i2t_usable_steps'] > 0 and br['e0_B']['t2i_usable_steps'] > 0, **br)
best = torch.load(C / 'model_best.pth.tar', map_location='cpu', weights_only=False)
bi = max(range(len(co)), key=lambda i: (co[i]['dev']['rsum'], -i))
ok('dev_best_rule', best['epoch'] == co[bi]['epoch'] and best['best_rsum'] == co[bi]['dev']['rsum'], best_epoch=best['epoch'])

pk = {Path(p).parent.name + '/' + Path(p).name: jl(p) for d in (C, X) for p in glob.glob(str(d / 'PEAK_*.json'))}
gm = {k: v.get('img_gather') for k, v in pk.items()}
ok('gather_modes', any(k.startswith('cont') and v in ('gpu', 'cpu') for k, v in gm.items())
   and any(k.startswith('split') and v == 'cpu' for k, v in gm.items()), modes=gm,
   peak_alloc_GiB={k: round(v['peak_mem_allocated_GiB'], 2) for k, v in pk.items()})
hx = jl(X / 'EPOCHS.json')
ok('split_resumed_own_state', (hx.get('resumed_from') or {}).get('epoch') == 0 and len(hx['cotrain']) == 2
   and jl(X / 'DONE.json')['status'] == 'COMPLETE', resumed_from=hx.get('resumed_from'))

fa = ST.fingerprint(torch.load(C / 'state_last.pt', map_location='cpu', weights_only=False))
fb = ST.fingerprint(torch.load(X / 'state_last.pt', map_location='cpu', weights_only=False))
cmp = ST.compare(fa, fb)
rs = ([e['dev']['rsum'] for e in co], [e['dev']['rsum'] for e in hx['cotrain']])
ok('resume_and_host_gather_bitwise_vs_continuous', cmp['equal'] and rs[0] == rs[1], differing=cmp['differing'], rsum_cont=rs[0], rsum_split=rs[1])

dp = (S / 'DRY_parent.txt').read_text()
du = (S / 'DRY_unit.txt').read_text()
ok('runner_dry_parent', 'r30_run.py --r30_stage warmup' in dp and 'dry parent rc=0' in dp and '--data_name cc152k_precomp' in dp)
ok('runner_dry_unit_refuses_smoke_parent', 'Parent is not a full 5-epoch warmup' in du and 'dry unit rc=1' in du,
   note='PARENT.json read + sha256 re-check passed; the smoke parent (2 x 30 it) is correctly refused as a formal parent')

envs = run_json(C)
out = dict(all_pass=all(v['ok'] for v in res.values()), checks=res, smoke_dir=str(S), job=os.environ.get('SLURM_JOB_ID'),
           gpu=envs[0]['env']['gpu'] if envs else None, code_files_sha256=envs[0]['env']['code_sha256'] if envs else None,
           protocol_sha256=hashlib.sha256((ROOT / 'PROTOCOL.json').read_bytes()).hexdigest(),
           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
code = json.loads((ROOT / 'code_r37s' / 'INSTALL_MANIFEST.json').read_text())['files_sha256']
out['code_eq_manifest'] = out['code_files_sha256'] == code
out['all_pass'] = out['all_pass'] and out['code_eq_manifest']
(S / 'SMOKE_CHECK.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
(ROOT / f"SMOKE_CHECK_{out['job']}.json").write_text(json.dumps(out, indent=1, default=str) + '\n')
print('ALL_PASS' if out['all_pass'] else 'SOME_FAIL')
sys.exit(0 if out['all_pass'] else 1)
