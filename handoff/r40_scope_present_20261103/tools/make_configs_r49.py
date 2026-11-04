"""R49 configs (user 2026-11-04 「好，准备R49吧，smoke过了就提交，TEST也再读一次」): the R48 layout-free N-frozen extended to the remaining parents
(Flickr60 R30-H s1002/s1003, Flickr20 IN2R s1001-3, Flickr40 IN2R s1001-3, Flickr80 IN2R s1002/s1003, COCO60 IN2R s1002/s1003, COCO80 IN2R s1001-3,
COCO20 IN2R s1001, CC152K R31-H0 s1002/s1003, CC152K R37S) x J/BCE/NCE = 57 units, 8000 updates, on each parent's R47-style layout-free G cache
(tools/r47_build_lf.py, cond <base>_lfG). Each config = that parent's existing N-frozen config (R41-C / R42 / R43-A) with only cond -> <base>_lfG and
the code tree / bookkeeping changed; for the five parents whose N-frozen ran only in the R38 / R37S trees (Flickr60 s1002/s1003, CC s1002/s1003, R37S)
the R48 config of the same dataset is used with the parent fields (cond, parent_H, parent_key, parent_seed) of the parent's R39X JOINT config and the
frozen fields of its R38 / R37S N-frozen config. Code trees r49f/r49o/r49c (r48f/r48o/r47cca byte copies + R49 block in common.py).
Writes configs/R49_*.json, configs/UNITS_R49.json, RESOLVED_MATRIX_R49.csv. Usage: python -I -B tools/make_configs_r49.py"""
import copy, csv, hashlib, json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
R39C = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs')
R38C = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
R37SC = Path('/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/configs')
QUOTE = '好，准备R49吧，smoke过了就提交，TEST也再读一次'
OBJ = ('J', 'BCE', 'NCE')
FROZEN_FIELDS = ('adaptation', 'architecture_rule', 'encoder', 'evaluation', 'joint_rule', 'optimizer', 'student_modalities')
PARENT_FIELDS = ('cond', 'parent_H', 'parent_key', 'parent_seed')
RULE = ('R49: frozen parent encoders; parent cache centred by ONE parent mean per split / modality / peer (R47-style layout-free G cache of this parent, '
        'no sequential-batch grouping); shared G head trained on it; DEV two-stage readout with the same centring (primary).')
rd = lambda p: json.loads(Path(p).read_text()); sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
# key: (tree, base cond, direct frozen template | None, (R48 template, R39X JOINT parent config, R38/R37S frozen template))
DIRECT = {
    'F20_IN2R_H1001': ('r49f', 'f20in_s1001', W / 'configs/R42_N_F20_IN2R_H1001_N_frozen_{o}.json'),
    'F20_IN2R_H1002': ('r49f', 'f20in_s1002', W / 'configs/R42_N_F20_IN2R_H1002_N_frozen_{o}.json'),
    'F20_IN2R_H1003': ('r49f', 'f20in_s1003', W / 'configs/R42_N_F20_IN2R_H1003_N_frozen_{o}.json'),
    'F40_IN2R_H1001': ('r49f', 'f40in', W / 'configs/R41_C_F40_IN2R_H1001_N_frozen_{o}.json'),
    'F40_IN2R_H1002': ('r49f', 'f40in_s1002', W / 'configs/R42_N_F40_IN2R_H1002_N_frozen_{o}.json'),
    'F40_IN2R_H1003': ('r49f', 'f40in_s1003', W / 'configs/R42_N_F40_IN2R_H1003_N_frozen_{o}.json'),
    'F80_IN2R_H1002': ('r49f', 'f80in_s1002', W / 'configs/R42_N_F80_IN2R_H1002_N_frozen_{o}.json'),
    'F80_IN2R_H1003': ('r49f', 'f80in_s1003', W / 'configs/R42_N_F80_IN2R_H1003_N_frozen_{o}.json'),
    'C60_IN2R_H1002': ('r49o', 'c60in_s1002', W / 'configs/R43_A_C60_IN2R_H1002_N_frozen_{o}.json'),
    'C60_IN2R_H1003': ('r49o', 'c60in_s1003', W / 'configs/R43_A_C60_IN2R_H1003_N_frozen_{o}.json'),
    'C80_IN2R_H1001': ('r49o', 'c80in_s1001', W / 'configs/R43_A_C80_IN2R_H1001_N_frozen_{o}.json'),
    'C80_IN2R_H1002': ('r49o', 'c80in_s1002', W / 'configs/R43_A_C80_IN2R_H1002_N_frozen_{o}.json'),
    'C80_IN2R_H1003': ('r49o', 'c80in_s1003', W / 'configs/R43_A_C80_IN2R_H1003_N_frozen_{o}.json'),
    'C20_IN2R_H1001': ('r49o', 'c20in_s1001', W / 'configs/R43_A_C20_IN2R_H1001_N_frozen_{o}.json'),
}
VIA_R48 = {
    'F60_R30_H1002': ('r49f', 'n60s1002', W / 'configs/R48_F60_R30_H1001_N_frozen_lfG_{o}.json', R39C / 'R39X_Flickr60_R30H_s1002_N_joint_global_{o}.json', R38C / 'r38_B_f60_R30_s1002_N_shared_{o}_cosine.json'),
    'F60_R30_H1003': ('r49f', 'n60s1003', W / 'configs/R48_F60_R30_H1001_N_frozen_lfG_{o}.json', R39C / 'R39X_Flickr60_R30H_s1003_N_joint_global_{o}.json', R38C / 'r38_B_f60_R30_s1003_N_shared_{o}_cosine.json'),
    'CC_R31_H1002': ('r49c', 'cc_s1002', W / 'configs/R48_CC31_H1001_N_frozen_lfG_{o}.json', R39C / 'R39X_CC152K_R31H0_s1002_N_joint_global_{o}.json', R38C / 'r38_B_cc_R31_s1002_N_shared_{o}_cosine.json'),
    'CC_R31_H1003': ('r49c', 'cc_s1003', W / 'configs/R48_CC31_H1001_N_frozen_lfG_{o}.json', R39C / 'R39X_CC152K_R31H0_s1003_N_joint_global_{o}.json', R38C / 'r38_B_cc_R31_s1003_N_shared_{o}_cosine.json'),
    'CC_R37S_H1001': ('r49c', 'cc_r37s', W / 'configs/R48_CC31_H1001_N_frozen_lfG_{o}.json', R39C / 'R39X_CC152K_R37S_s1001_N_joint_global_{o}.json', R37SC / 'r37s_cc_r30h_s1001_N_shared_{o}_cosine.json'),
}
DROP = ('paired_joint_unit', 'reference_unit', 'reference_joint', 'frozen_template')
units, rows = [], []
for o in OBJ:
    for pk, (tree, base, tfmt) in DIRECT.items():
        tp = Path(str(tfmt).format(o=o)); t = rd(tp); assert t['cond'] == base and t['encoder'] == 'frozen' and t['objective'] == o and t['steps'] == 8000
        cfg = copy.deepcopy(t); src = dict(kind='direct', frozen_config=tp.stem)
        cfg['cond'] = f'{base}_lfG'
        units.append((pk, o, tree, cfg, src, tp))
    for pk, (tree, base, r48fmt, jfmt, ffmt) in VIA_R48.items():
        r48 = rd(Path(str(r48fmt).format(o=o))); jp = Path(str(jfmt).format(o=o)); j = rd(jp); fp = Path(str(ffmt).format(o=o)); f = rd(fp)
        assert j['cond'] == base and f['objective'] == o and r48['objective'] == o
        cfg = copy.deepcopy(r48)
        for k in PARENT_FIELDS:
            cfg[k] = copy.deepcopy(j[k])
        cfg['cond'] = f'{base}_lfG'
        for k in FROZEN_FIELDS:
            if k in f: cfg[k] = copy.deepcopy(f[k])
        cfg.update(adaptation='frozen', encoder='frozen', student_modalities=[], joint_rule=None)
        src = dict(kind='via_R48', r48_config=Path(str(r48fmt).format(o=o)).stem, parent_fields_from=jp.stem, frozen_fields_from=fp.stem)
        units.append((pk, o, tree, cfg, src, fp))
out = []
for pk, o, tree, cfg, src, tp in units:
    for k in DROP: cfg.pop(k, None)
    uid = f'R49_{pk}_N_frozen_lfG_{o}'
    cfg.update(unit_id=uid, line='R49', block='R49', code_tree=tree, runner=f'{tree}/train_unit.py', oracle=False, test_read=False, centring_variant='G', centring_rule=RULE,
               reference_unit=None, reference_note='layout-free G parent cache of this parent (R47 build tool); no in-unit reference; the native-centring N-frozen of the same parent is listed post hoc',
               native_frozen_unit=src.get('frozen_config') or src.get('frozen_fields_from'), derivation=src,
               authorisation=dict(user_quote=QUOTE, relayed_by='main session', note='formal submission after CPU check and GPU smoke pass'))
    p = W / 'configs' / f'{uid}.json'; assert not p.exists() or rd(p) == cfg, f'{p} exists with different content'
    p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); out.append(uid)
    rows.append(dict(unit_id=uid, tree=tree, cond=cfg['cond'], parent_key=pk, objective=o, steps=cfg['steps'], init_seed=cfg['init_seed'], derivation=src['kind'],
                     native_frozen_unit=cfg['native_frozen_unit'], config_sha256=sh(p)))
(W / 'configs' / 'UNITS_R49.json').write_text(json.dumps(out, indent=1))
with open(W / 'RESOLVED_MATRIX_R49.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R49 configs', len(out))
