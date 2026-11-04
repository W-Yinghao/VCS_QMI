"""R48 configs (user 2026-11-04 「好，准备R48吧，smoke过了就提交」): N-frozen (shared G head, cosine schedule, encoders frozen) trained and read out
on the R47 layout-free G parent cache, x J/BCE/NCE x {Flickr60 R30-H s1001, Flickr80 IN2R f80 s1001, Flickr0 R30-H f00 s1001, COCO60 IN2R c60 s1001,
CC152K R31-H0 s1001} = 15 units, 8000 updates. Each config = the R47 G N-JOINT config of the same condition / objective (same cond *_lfG, init seed,
sampler stream, steps, readouts) with only the seven fields that separate frozen from JOINT configs (adaptation, architecture_rule, encoder, evaluation,
joint_rule, optimizer, student_modalities) taken from that condition's existing N-frozen config (R38 / R39X / R41-C / R43-A). Code trees r48f/r48o/r48c
(byte copies of r47f/r47o/r47c + R48 block in common.py). Writes configs/R48_*.json, configs/UNITS_R48.json, RESOLVED_MATRIX_R48.csv.
Usage: python -I -B tools/make_configs_r48.py"""
import copy, csv, hashlib, json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
R39C = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs')
R38C = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
QUOTE = '好，准备R48吧，smoke过了就提交'
OBJ = ('J', 'BCE', 'NCE')
FROZEN_FIELDS = ('adaptation', 'architecture_rule', 'encoder', 'evaluation', 'joint_rule', 'optimizer', 'student_modalities')
CONDS = {  # R47 key: (tree, frozen template format, template unit)
    'F60_R30_H1001': ('r48f', R38C / 'r38_B_f60_R30_s1001_N_shared_{o}_cosine.json'),
    'F80_IN2R_H1001': ('r48f', W / 'configs' / 'R41_C_F80_IN2R_H1001_N_frozen_{o}.json'),
    'F00_R30_H1001': ('r48f', R39C / 'R39X_Flickr00_R30H_s1001_N_frozen_shared_{o}.json'),
    'C60_IN2R_H1001': ('r48o', W / 'configs' / 'R43_A_C60_IN2R_H1001_N_frozen_{o}.json'),
    'CC31_H1001': ('r48c', R38C / 'r38_B_cc_R31_s1001_N_shared_{o}_cosine.json'),
}
RULE = ('R48: frozen parent encoders; parent cache centred by ONE parent mean per split / modality / peer (the R47 G layout-free cache, no sequential-batch '
        'grouping); shared G head trained on it; DEV two-stage readout with the same centring (primary). Paired with R47 G N-JOINT (same init / sampler stream).')
rd = lambda p: json.loads(Path(p).read_text())
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
units, rows = [], []
for o in OBJ:
    for pk, (tree, tfmt) in CONDS.items():
        g = rd(W / 'configs' / f'R47_G_{pk}_N_joint_{o}.json')
        tp = Path(str(tfmt).format(o=o)); t = rd(tp)
        assert g['centring_variant'] == 'G' and g['cond'].endswith('_lfG') and g['steps'] == 8000 and g['objective'] == o
        cfg = copy.deepcopy(g)
        for k in FROZEN_FIELDS:
            cfg[k] = copy.deepcopy(t.get(k))
        cfg.update(adaptation='frozen', encoder='frozen', student_modalities=[], joint_rule=None)
        if not cfg.get('architecture_rule'):
            cfg['architecture_rule'] = 'shared G (R35/R36 GHead): g = a s_H + b + w^T h(phi(u, v, s_H)) + c'
        uid = f'R48_{pk}_N_frozen_lfG_{o}'
        cfg.update(unit_id=uid, line='R48', block='R48', code_tree=tree, runner=f'{tree}/train_unit.py', centring_variant='G', centring_rule=RULE,
                   reference_unit=None, reference_joint=None, paired_joint_unit=g['unit_id'],
                   reference_note='layout-free G parent cache (R47 build); no in-unit reference; paired with the R47 G N-JOINT unit (same init / sampler stream)',
                   frozen_template=dict(unit=tp.stem, config_sha256=sh(tp), fields=list(FROZEN_FIELDS)),
                   authorisation=dict(user_quote=QUOTE, relayed_by='main session', note='formal submission after CPU check and GPU smoke pass'))
        p = W / 'configs' / f'{uid}.json'; assert not p.exists() or rd(p) == cfg, f'{p} exists with different content'
        p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); units.append(uid)
        rows.append(dict(unit_id=uid, tree=tree, cond=cfg['cond'], objective=o, steps=8000, paired_joint=g['unit_id'], frozen_template=tp.stem, config_sha256=sh(p)))
(W / 'configs' / 'UNITS_R48.json').write_text(json.dumps(units, indent=1))
with open(W / 'RESOLVED_MATRIX_R48.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R48 configs', len(units))
