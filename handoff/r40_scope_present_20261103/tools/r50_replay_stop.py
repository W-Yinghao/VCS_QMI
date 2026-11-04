"""R50 JOINT replay (reuse evidence): the R47 G N-JOINT reference config on the R50 FIT-fixed cond, trained by the production run_unit with the reference's
full 8000-update schedule (cosine over 8000 steps, readouts 0/250) and stopped after the step-250 checkpoint (run_unit stop_at); then the batch-index hash
chain at 250 and the head / null / student state at 250 are compared with the reference unit's ckpt_step250.pt (bitwise expected on the reference GPU type).
Gate-mode replays with --gate-steps 250 (jobs 1086228 / 1086282 / 1086300 / 1086281) compress the cosine schedule to 250 updates and are not valid replays
(kept on record). Writes <out>/<unit>/REPLAY_STOP.json and appends ROOT/REPLAY_R50.json under key '<job>_stop'.
Usage: python -B tools/r50_replay_stop.py <tree> <replay units> <out_dir>"""
import json, os, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
TREE, UNITS, OD = sys.argv[1], sys.argv[2].split(','), Path(sys.argv[3])
sys.path.insert(0, str(W / TREE))
import torch
import common as C
C.install_guards(role='trainer')
import train_unit as TU
TU.set_determinism()
dev = torch.device('cuda'); ctxs = {}; out = dict(job=os.environ.get('SLURM_JOB_ID'), mode='stop_at_250_full_schedule', units={})
for u in UNITS:
    cfg = TU.cfg_of(u); cfg['_config_sha256'] = C.sha256(W / 'configs' / f'{u}.json'); ref = cfg['replay_of']['unit_id']
    rd = C.OUT / cfg['replay_of']['runs'] / ref; o = OD / u
    if cfg['cond'] not in ctxs:
        ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], dev, need_local=False)}
    ctx = ctxs[cfg['cond']]
    if cfg.get('encoder') == 'joint_global':
        ctx.ensure_joint()
    t0 = time.time()
    res = TU.run_unit(dict(cfg, phase=1), ctx, pool=TU.pool_of(cfg, ctx), out_dir=o, stop_at=250)
    rth = json.loads((rd / 'TRAIN_HISTORY.json').read_text()); rr = json.loads((rd / 'RESULT.json').read_text())
    a = torch.load(o / 'ckpt_step250.pt', map_location='cpu', weights_only=False); b = torch.load(rd / 'ckpt_step250.pt', map_location='cpu', weights_only=False)
    neq = nt = 0; mx = 0.0
    for part in ('heads', 'nulls', 'student'):
        if a.get(part) is None and b.get(part) is None:
            continue
        for k in a[part]:
            nt += 1
            if not torch.equal(a[part][k], b[part][k]):
                neq += 1; mx = max(mx, float((a[part][k].double() - b[part][k].double()).abs().max()))
    h = res['hist']; k = rr['config']['ckpt_steps'].index(250)
    rec = dict(reference=ref, step=250, chain_equal=h['batch_hash'][-1] == rth['batch_hash'][k], init_equal=res['init_hash'] == rr['phases'][0]['identity']['init_state_sha256'],
               state_tensors=nt, state_tensors_not_bitwise=neq, state_max_abs=mx, state_bitwise=neq == 0, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
               reference_gpu=rr['env']['gpu'], reference_host=rr['env'].get('host'), lr_at_250=h['r38']['lr_at_readout'].get('250'), reference_lr_at_250=rth['r38']['lr_at_readout'].get('250'),
               rsum_step250_primary=json.loads((o / 'eval_step250.json').read_text())['dev_retrieval']['rsum'], seconds=time.time() - t0)
    rec['passed'] = bool(rec['chain_equal'] and rec['init_equal'] and rec['state_bitwise'])
    C.write_json(o / 'REPLAY_STOP.json', rec); out['units'][u] = rec
    print('R50_REPLAY_STOP', u, json.dumps({k_: rec[k_] for k_ in ('chain_equal', 'init_equal', 'state_bitwise', 'state_max_abs', 'host', 'reference_host', 'lr_at_250', 'reference_lr_at_250')}), flush=True)
out['ALL_PASS'] = all(v['passed'] for v in out['units'].values())
p = W / 'REPLAY_R50.json'; agg = json.loads(p.read_text()) if p.exists() else {}; agg[f"{out['job']}_stop"] = out; p.write_text(json.dumps(agg, indent=1, default=str))
print('R50_REPLAY ALL_PASS', out['ALL_PASS'])
