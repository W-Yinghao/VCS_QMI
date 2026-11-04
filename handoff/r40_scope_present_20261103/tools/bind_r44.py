"""R44 DEDUP_R44.json + PARENT_BINDING_R44.json (exact binding of parents / ALL references / R41-A paired controls; missing -> reported, never substituted).
Usage: python -I -B tools/bind_r44.py (v9_sgr env)"""
import csv, glob, hashlib, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rj = lambda p: json.loads(Path(p).read_text())
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
units = rj(W / 'configs' / 'UNITS_R44.json')
key = lambda c: (c.get('cond'), c.get('supervision'), c.get('routing'), c.get('adaptation') or ('both_joint' if c.get('encoder') == 'joint_global' else 'frozen'), c.get('objective'), c.get('steps'),
                 str(c.get('init_seed')), str(c.get('sampler_seed')), (c.get('optimizer') or {}).get('schedule'), (c.get('optimizer') or {}).get('lr_encoder'), c.get('architecture'))
prior = {}
for d in [W / 'configs', Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs'), Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs'),
          Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/configs'), Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/configs')]:
    for f in glob.glob(str(d / '*.json')):
        try:
            c = rj(f)
        except Exception:
            continue
        if isinstance(c, dict) and 'unit_id' in c and not c['unit_id'].startswith('R44_'):
            prior.setdefault(key(c), []).append(c['unit_id'])
runs = {Path(p).name for p in glob.glob(str(OUT / 'runs*' / 'R44_*'))}
keys = [key(rj(W / 'configs' / f'{u}.json')) for u in units]
dd = dict(n_units=len(units), n_prior_configs=sum(map(len, prior.values())), unique_within=len(set(keys)) == len(keys),
          exact_prior_matches={u: prior[k] for u, k in zip(units, keys) if k in prior}, existing_R44_run_dirs=sorted(runs),
          key='(cond, supervision, routing, adaptation, objective, steps, init_seed, sampler_seed, schedule, lr_encoder, architecture)',
          scope='R37/R37C/R38/R39/R39X/R40/R41/R42/R43 configs (this handoff + R39/R38/R37/R37C trees); no R44 unit matches a completed run -> all 39 submitted')
(W / 'DEDUP_R44.json').write_text(json.dumps(dd, indent=1))
bind, missing = {}, []
for u in units:
    c = rj(W / 'configs' / f'{u}.json'); e = dict(cond=c['cond'], tree=c['code_tree'], parent_H=c.get('parent_H'))
    pm = (c.get('parent_H') or {}).get('model_best')
    if pm and Path(pm).exists():
        e['parent_model_best_sha256'] = sh(pm)
    elif pm:
        missing.append((u, 'parent model_best', pm))
    r = c.get('reference_joint')
    if r:
        e['in_unit_reference'] = dict(unit_id=r['unit_id'], mode=r.get('mode'), result_sha_ok=sh(r['result_json']) == r['result_json_sha256'], ckpt_step0_sha_ok=sh(r['ckpt_step0']) == r['ckpt_step0_sha256'],
                                      done=rj(Path(r['result_json']).parent / 'DONE.json')['status'])
    if c['block'] == 'A':
        pk = '_'.join(u.split('_')[2:5]); d = OUT / 'runs_r41' / f"R41_A_{pk}_N_joint_16k_{c['objective']}"
        e['paired_control_seed1001'] = dict(unit_id=d.name, result_json_sha256=sh(d / 'RESULT.json') if (d / 'RESULT.json').exists() else None,
                                            done=rj(d / 'DONE.json')['status'] if (d / 'DONE.json').exists() else None, has_12000_readout=(d / 'eval_step12000.json').exists())
        if not (d / 'RESULT.json').exists():
            missing.append((u, 'R41-A control', str(d)))
    bind[u] = e
(W / 'PARENT_BINDING_R44.json').write_text(json.dumps(dict(units=bind, missing=missing, note='R41-A seed-1001 controls have no 12000 readout (left empty, not imputed)'), indent=1))
print('R44 dedup exact matches', len(dd['exact_prior_matches']), 'existing runs', len(runs), 'missing', missing)
