"""R42 (main session relaying the user authorisation 「…只要你觉得有用的就提交，不用问我」): 48 units on the r42f tree, R41 C recipe (8000 updates, N route,
matched J/BCE/NCE, step-0 native reproduction, R40 one-step rule, TEST not read).
  N  noise curve on IN2R-native Flickr parents x {N-frozen-shared-cosine, N-both_joint}: f20 R31 H0_codecfg s1001/s1002/s1003 (official NCR 0.2
     index), f40 R31 H0_default s1002/s1003, f80 R33 H0 s1002/s1003 (R33 NOISE_LOCK 0.8 map) -> 42; configs = the R41 C configs with the parent swapped.
  O  ORACLE_MISSING (R39 rule, isolated process) x {frozen, both_joint} on the 80% parent R33 H0 f80_s1001 (R41 parents, read-only) -> 6;
     references: R41 C N-frozen f80 (frozen: init/chain; joint: init/chain) and R41 C N-JOINT f80 (joint: student init, chain).
Writes configs/R42_*.json, configs/UNITS_R42.json, RESOLVED_MATRIX_R42.csv. Usage: python -I -B tools/make_configs_r42.py (v9_sgr env)"""
import csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
sys.path.insert(0, str(W / 'r42f')); sys.modules.pop('common', None)   # r42f common (new conds), not the r41f one imported by make_configs_r41
from common import sha256, write_json, read_json, CONDS
OUTR41 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/runs_r41')
NEW = [('F20_IN2R_H1001', 'f20in_s1001'), ('F20_IN2R_H1002', 'f20in_s1002'), ('F20_IN2R_H1003', 'f20in_s1003'), ('F40_IN2R_H1002', 'f40in_s1002'),
       ('F40_IN2R_H1003', 'f40in_s1003'), ('F80_IN2R_H1002', 'f80in_s1002'), ('F80_IN2R_H1003', 'f80in_s1003')]
DESC = {20: 'Flickr30K 20% noise (official NCR noise_index f30k_precomp_0.2)', 40: 'Flickr30K 40% injected noise (R25 noise map 0.4)', 80: 'Flickr30K 80% injected noise (R33 NOISE_LOCK map 0.8)'}
units, rows = [], []


def put(cfg, refs):
    cfg.update(code_tree='r42f', runner=cfg['runner'].replace('r41f/', 'r42f/').replace('r41g/', 'r42f/'), line='R42')
    write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
    rows.append(dict(unit_id=cfg['unit_id'], block=cfg['block'], cond=cfg['cond'], noise_percent=cfg['noise_percent'], parent_seed=cfg['parent_seed'], supervision=cfg['supervision'],
                     adaptation=cfg['adaptation'], objective=cfg['objective'], steps=cfg['steps'], oracle=cfg['oracle'], runner=cfg['runner'], refs=refs,
                     config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))


def r41ref(uid, student=False):
    d = OUTR41 / uid
    e = M.ref(d, 'R42 reused reference (R41 C, same parent): init / chain' + (' / student init' if student else '') + ' asserted in-unit (read-only)', student)
    e['config_sha256'] = sha256(W / 'configs' / f'{uid}.json')
    return e


for o in M.OBJ:
    for pk, cond in NEW:
        c = CONDS[cond]
        for kind in ('frozen', 'joint'):
            base = read_json(W / 'configs' / f"R41_C_F40_IN2R_H1001_N_{kind}_{o}.json")
            ph = dict(unit=c['h_unit'], model_best=str(Path(c['parent_dir']) / 'model_best.pth.tar'), seed=c['parent_seed'], description=f"{c['h_unit']} (IN2R-native)")
            cfg = dict(base, unit_id=f'R42_N_{pk}_N_{kind}_{o}', block='N', cond=cond, noise_percent=c['noise_percent'], parent_seed=c['parent_seed'], parent_H=ph, parent_key=pk,
                       dataset=DESC[c['noise_percent']], reference_unit=None, reference_note='new parent: no prior unit; frozen/joint pair shares init and sampler stream (compared in the summary)')
            put(cfg, '')
    for kind in ('frozen', 'joint'):
        base = read_json(W / 'configs' / f'R41_C_F80_IN2R_H1001_N_{kind}_{o}.json')
        oc = read_json(M.R39 / 'configs' / f'R39_Flickr60_ORACLE_MISSING_frozen_{o}.json')
        cfg = dict(base, unit_id=f'R42_O_F80_IN2R_H1001_ORACLE_{kind}_{o}', block='O', oracle=True, runner=base['runner'].replace('train_unit.py', 'train_oracle.py'),
                   supervision='ORACLE_MISSING', supervision_label='ORACLE_MISSING', supervision_rule=oc['supervision_rule'] + ' (R42 O: 80% IN2R-native parent R33 H0 f80_s1001; bit-identical R39 oracle_truth.py)',
                   reference_unit=r41ref(f'R41_C_F80_IN2R_H1001_N_frozen_{o}'), reference_note='R41 C N-frozen of the same parent (R38 rule: head init + batch chain)')
        refs = f'n_frozen=R41_C_F80_IN2R_H1001_N_frozen_{o}'
        if kind == 'joint':
            cfg['reference_joint'] = r41ref(f'R41_C_F80_IN2R_H1001_N_joint_{o}', True); refs += f';n_joint=R41_C_F80_IN2R_H1001_N_joint_{o}'
        put(cfg, refs)
write_json(W / 'configs' / 'UNITS_R42.json', units)
with open(W / 'RESOLVED_MATRIX_R42.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R42 configs', len(units))
