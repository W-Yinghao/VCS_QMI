#!/usr/bin/env python3
"""FROZEN reuse receipt: rerun the NR4 trainer (scope FROZEN) for the full 4000 added steps on the NR3-B reference unit's GPU type
and compare with the stored NR3-B run: every saved checkpoint (stage 0/250/500/1000/2000/4000) model state + optimizer state, the
sampler RNG state, and the logged loss/grad-norm trajectory. Bitwise equality => the 12 NR3-B endpoints are reused as FROZEN."""
import json, sys, time
import numpy as np, torch
from nr1.io import read_json, write_json, sha256
from nr4 import train
from nr4.paths import paths, OUT
H = '/home/infres/yinwang/CS_QMI/handoff/nr4_20261104'


def cmp_state(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a) if set(a) == set(b) else float('inf')


for unit in sys.argv[1:]:
    cfg = read_json(f'{H}/configs_production/{unit}.json'); ref = cfg['reference_unit']; P = paths(cfg['condition'], ref)
    od = OUT/'reuse_check'/unit
    if (od/'RECEIPT.json').exists(): continue
    t = time.perf_counter()
    if not (od/'DONE.json').exists(): train.run(cfg, P['fit'], P['observed'], P['calibration'], P['prefix'], P['tables'], od, None, 'cuda')
    sec = time.perf_counter() - t
    nr3cfg = read_json(f'/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/configs_production/{ref}.json')
    same_fields = {k: nr3cfg[k] == cfg[k] for k in nr3cfg if k in cfg and k not in ('unit_id', 'protocol_id')}
    rec = dict(unit=unit, reference_unit=ref, gpu=torch.cuda.get_device_name(), reference_gpu=read_json(P['nr3_run']/'TRAIN_COST.json')['gpu'],
               config_fields_equal=all(same_fields.values()), config_fields_differing=[k for k, v in same_fields.items() if not v],
               nr3_reference_config_sha256=sha256(f'/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/configs_production/{ref}.json'), rerun_seconds=sec, checkpoints=[])
    for s in (0, 250, 500, 1000, 2000, 4000):
        a = torch.load(od/f'step_{8000+s:06d}.pt', map_location='cpu', weights_only=False); b = torch.load(P['nr3_run']/f'step_{8000+s:06d}.pt', map_location='cpu', weights_only=False)
        oa, ob = a['optimizer']['state'], b['optimizer']['state']
        opt_d = max([float((oa[k][n].double() - ob[k][n].double()).abs().max()) for k in ob for n in ('exp_avg', 'exp_avg_sq')] or [0.0]) if set(oa) == set(ob) else float('inf')
        rec['checkpoints'].append(dict(stage_step=s, model_max_abs=cmp_state(a['model'], b['model']), optimizer_moment_max_abs=opt_d,
                                       sampler_state_equal=json.dumps(a['rng_sampler'], default=str) == json.dumps(b['rng_sampler'], default=str)))
    la = [json.loads(l) for l in open(od/'TRAIN.jsonl')]; lb = [json.loads(l) for l in open(P['nr3_run']/'TRAIN.jsonl')]
    rec['logged_steps'] = [len(la), len(lb)]
    rec['loss_max_abs'] = max(abs(x['loss'] - y['loss']) for x, y in zip(la, lb)); rec['grad_norm_max_abs'] = max(abs(x['grad_norm'] - y['grad_norm']) for x, y in zip(la, lb))
    rec['bitwise_equal'] = rec['loss_max_abs'] == 0 and all(c['model_max_abs'] == 0 and c['optimizer_moment_max_abs'] == 0 and c['sampler_state_equal'] for c in rec['checkpoints'])
    rec['decision'] = 'REUSE_NR3B_ENDPOINT' if rec['bitwise_equal'] and rec['config_fields_equal'] else 'RERUN_FROZEN_CONTROL'
    write_json(od/'RECEIPT.json', rec); print(json.dumps({k: rec[k] for k in ('unit', 'gpu', 'bitwise_equal', 'loss_max_abs', 'decision')}))
