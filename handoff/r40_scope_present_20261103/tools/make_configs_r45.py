"""R45 configs (main session, same user authorisation 「…只要你觉得有用的就提交，不用问我」): COCO N x both_joint 16000 updates on the IN2R-native R33 H0
parents c60 s1001 / c80 s1001 x J/BCE/NCE = 6, tree r45o (= r43o + R45 common section). Exact R41-A 16k recipe: stretched cosine over 16000
from the original H + fresh zero-output head, training seed 1001 (sampler [1001, 3]); readouts 0/250/500/1000/2000/4000/8000/12000/16000;
primary 16000 (8000 also reported). In-unit prefix reference = the R43-A 8000-update N-JOINT unit of the same parent / stream (head / null init,
student init, batch chain at its 7 readouts). Writes configs/R45_*.json, configs/UNITS_R45.json, RESOLVED_MATRIX_R45.csv."""
import copy, csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
from common import sha256, write_json, read_json
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
QUOTE = '你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我'
units, rows = [], []
for pk in ('C60_IN2R_H1001', 'C80_IN2R_H1001'):
    for o in M.OBJ:
        b = read_json(W / 'configs' / f'R43_A_{pk}_N_joint_{o}.json'); a41 = read_json(W / 'configs' / f'R41_A_F60_R30_H1001_N_joint_16k_{o}.json')
        assert b['steps'] == 8000 and b['init_seed'] == 1001 and b['sampler_seed'] == [1001, 3] and b['optimizer'] == a41['optimizer']
        cfg = copy.deepcopy(b)
        cfg.update(unit_id=f'R45_{pk}_N_joint_16k_{o}', block='R45', line='R45', code_tree='r45o', runner='r45o/train_unit.py', steps=16000,
                   ckpt_steps=[0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000], primary_endpoint='fixed_step_16000 (step 8000 also reported)',
                   schedule_rule=a41['schedule_rule'], budget_rule='R41-A 16000-update stretched cosine from the original H + fresh zero-output head (never resumed from the 8000-update unit); same init / sampler stream as R43-A',
                   reference_unit=None, reference_note=None,
                   reference_joint=M.ref(OUT / 'runs_r43' / f'R43_A_{pk}_N_joint_{o}', 'R45 prefix reference (R43-A 8000-update N-JOINT, same parent / init / stream): head / null init, student init, batch chain at its 7 readouts', True, 'prefix'),
                   authorisation=dict(user_quote=QUOTE, relayed_by='main session', reason='COCO has ~4x the Flickr training images; R41-A showed +3-8 rSum from 16k on Flickr60; COCO JOINT gains survive at 80%'))
        write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
        rows.append(dict(unit_id=cfg['unit_id'], cond=cfg['cond'], objective=o, steps=16000, init_seed=1001, sampler_seed='[1001, 3]', ckpt_steps=' '.join(map(str, cfg['ckpt_steps'])),
                         prefix_reference=cfg['reference_joint']['unit_id'], config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))
write_json(W / 'configs' / 'UNITS_R45.json', units)
with open(W / 'RESOLVED_MATRIX_R45.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R45 configs', units)
