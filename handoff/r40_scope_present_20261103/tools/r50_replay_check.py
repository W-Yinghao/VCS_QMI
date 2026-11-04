"""R50 replay verdict (reuse evidence for the R47 G / R48 N references): for each R50_REPLAY_<ref> unit in the gate dir, the batch-index hash chain at the
last gate step equals the reference unit's chain at the same step, and the head / null / JOINT-student state at that step equals the reference checkpoint
(bitwise expected when the GPU type equals the reference's; max |delta| recorded otherwise). Writes <gate>/REPLAY_R50.json, appends ROOT/REPLAY_R50.json.
Usage: python -I -B tools/r50_replay_check.py <gate_dir> <units> <rc>"""
import json, os, sys, time
from pathlib import Path
import torch
W = Path(__file__).resolve().parents[1]; G = Path(sys.argv[1]); units = sys.argv[2].split(','); rc = int(sys.argv[3])
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
out = dict(job=os.environ.get('SLURM_JOB_ID'), gate_dir=str(G), rc=rc, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), units={})
for u in units:
    cfg = rj(W / 'configs' / f'{u}.json'); ref = cfg['replay_of']['unit_id']; rd = OUT / cfg['replay_of']['runs'] / ref; d = G / u
    r = rj(d / 'RESULT.json') or {}; th = rj(d / 'TRAIN_HISTORY.json') or {}; rr = rj(rd / 'RESULT.json'); rth = rj(rd / 'TRAIN_HISTORY.json')
    rec = dict(reference=ref, done=(d / 'DONE.json').exists(), failed=[f.name for f in d.glob('FAILED_*.json')])
    if rec['done']:
        step = r['phases'][-1]['steps']; k = rr['config']['ckpt_steps'].index(step)   # gate steps (RESULT config holds the formal 8000)
        rec.update(step=step, chain_equal=th['batch_hash'][-1] == rth['batch_hash'][k], init_equal=r['phases'][0]['identity']['init_state_sha256'] == rr['phases'][0]['identity']['init_state_sha256'],
                   gpu=r['env']['gpu'], reference_gpu=rr['env']['gpu'])
        a = torch.load(d / f'ckpt_step{step}.pt', map_location='cpu', weights_only=False); b = torch.load(rd / f'ckpt_step{step}.pt', map_location='cpu', weights_only=False)
        mx, neq, nt = 0.0, 0, 0
        for part in ('heads', 'nulls', 'student'):
            if a.get(part) is None and b.get(part) is None:
                continue
            assert set(a[part]) == set(b[part]), part
            for kk in a[part]:
                x, y = a[part][kk], b[part][kk]; nt += 1
                if not torch.equal(x, y):
                    neq += 1; mx = max(mx, float((x.double() - y.double()).abs().max()))
        rec.update(state_tensors=nt, state_tensors_not_bitwise=neq, state_max_abs=mx, state_bitwise=neq == 0,
                   rsum_replay_last=r['final']['dev_retrieval']['rsum'], step0_native_check=(rj(d / 'eval_step0.json') or {}).get('step0_native_check', {}).get('passed'))
        rec['passed'] = bool(rec['chain_equal'] and rec['init_equal'] and (rec['state_bitwise'] or (rec['gpu'] != rec['reference_gpu'] and mx <= 1e-4)))
    else:
        rec['passed'] = False
    out['units'][u] = rec
out['ALL_PASS'] = rc == 0 and all(v['passed'] for v in out['units'].values())
(G / 'REPLAY_R50.json').write_text(json.dumps(out, indent=1, default=str))
agg = rj(W / 'REPLAY_R50.json') or {}; agg[out['job']] = out; (W / 'REPLAY_R50.json').write_text(json.dumps(agg, indent=1, default=str))
print('R50_REPLAY ALL_PASS', out['ALL_PASS'], json.dumps({u: (v['passed'], v.get('state_bitwise'), v.get('state_max_abs')) for u, v in out['units'].items()}))
