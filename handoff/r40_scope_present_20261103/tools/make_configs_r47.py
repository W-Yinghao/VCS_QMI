"""R47 configs (main session, standing authorisation 「…只要你觉得有用的就提交，不用问我」; formal submission only after the main session confirms):
N x both_joint x J/BCE/NCE x {Flickr60 R30-H s1001 (n60), Flickr80 IN2R f80 s1001 (f80in), Flickr0% R30-H f00 s1001 (n00), COCO60 R33 c60 s1001,
CC152K CC31} x {G: fixed global parent means, S: own training-batch means} = 30 units, 8000 updates, the condition's N-JOINT recipe (R39 / R39X /
R41-C / R43-A) with the cond swapped to the layout-free condition (layout-free parent cache, prep and JOINT prep); no in-unit reference.
Writes configs/R47_*.json, configs/UNITS_R47.json, RESOLVED_MATRIX_R47.csv. Usage: python -I -B tools/make_configs_r47.py"""
import copy, csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
from common import sha256, write_json, read_json
QUOTE = '你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我'
BASE = {'F60_R30_H1001': ('n60', 'r47f', M.R39 / 'configs' / 'R39_Flickr60_N_joint_global_{o}.json'), 'F80_IN2R_H1001': ('f80in', 'r47f', W / 'configs' / 'R41_C_F80_IN2R_H1001_N_joint_{o}.json'),
        'F00_R30_H1001': ('n00', 'r47f', M.R39 / 'configs' / 'R39X_Flickr00_R30H_s1001_N_joint_global_{o}.json'), 'C60_IN2R_H1001': ('c60in_s1001', 'r47o', W / 'configs' / 'R43_A_C60_IN2R_H1001_N_joint_{o}.json'),
        'CC31_H1001': ('cc', 'r47c', M.R39 / 'configs' / 'R39_CC152K_N_joint_global_{o}.json')}
RULE = {'G': 'R47 G: JOINT student and frozen parent cache centred by ONE parent mean per split / modality / peer (layout-free); DEV readout with the same centring (primary)',
        'S': 'R47 S: JOINT student centred by its own mean over the unique items of each training batch (evaluation: fixed random 512-item batches); frozen parent cache layout-free (global means); DEV readout with the same centring (primary)'}
units, rows = [], []
for o in M.OBJ:
    for pk, (cond, tree, cp) in BASE.items():
        b = read_json(Path(str(cp).format(o=o))); assert b['cond'] == cond and b['steps'] == 8000
        for v in ('G', 'S'):
            cfg = M.joint_fields(M.finish(copy.deepcopy(b), f'R47_{v}_{pk}_N_joint_{o}', v, 'both_joint'), b, 'both_joint')
            cfg.update(cond=f'{cond}_lf{v}', code_tree=tree, runner=f'{tree}/train_unit.py', line='R47', reference_unit=None, reference_joint=None, centring_variant=v, centring_rule=RULE[v],
                       reference_note='layout-free parent cache and student centring: no in-unit reference; native-layout readout reported post hoc (secondary)',
                       authorisation=dict(user_quote=QUOTE, relayed_by='main session', note='formal submission only after the main session confirms'))
            write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
            rows.append(dict(unit_id=cfg['unit_id'], variant=v, tree=tree, cond=cfg['cond'], base_cond=cond, objective=o, steps=8000, base_config=Path(str(cp).format(o=o)).name, config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))
write_json(W / 'configs' / 'UNITS_R47.json', units)
with open(W / 'RESOLVED_MATRIX_R47.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R47 configs', len(units))
