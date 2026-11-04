"""R43 configs (main session relaying the user authorisation 「你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我」).
R41 C / R42 recipe unchanged (8000 updates, shared G + cosine, matched J/BCE/NCE, step-0 native reproduction, R40 one-step rule, TEST not read).
  A  MS-COCO (second injected-noise dataset): IN2R-native R33 H0 parents c20 s1001, c60 s1001/s1002/s1003, c80 s1001/s1002/s1003 (main session added
     c80 s1002/s1003 after they finished) x {N-frozen-shared-cosine, N-both_joint} x J/BCE/NCE = 42, tree r43o; configs = the R41 C f80 configs with the parent and
     the dataset swapped (DEV = native 1K-image split, 1000 images x 5000 captions).
  B  deployable cleaner supervision under JOINT: R37 TRUST_NATIVE_NULL (TRUST+N: anchors from the parent's native GMM B-TRUST subset, full N
     lists / labels from the observed relation) x {frozen, both_joint} x J/BCE/NCE on the 80% Flickr parents R33 H0 f80 s1001/s1002/s1003 (tree
     r43f; their N-frozen / N-JOINT cells are R41 C / R42) and on CC152K CC31 R31 H0_default cc_s1001 (tree r43c; N-frozen R38, N-JOINT R39) = 24.
  No in-unit reference unit (TRUST changes the anchor stream; COCO parents are new); frozen/joint pairs share head init and sampler stream.
Writes configs/R43_*.json, configs/UNITS_R43.json, RESOLVED_MATRIX_R43.csv. Usage: python -I -B tools/make_configs_r43.py (v9_sgr env)"""
import copy, csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
from common import sha256, write_json, read_json   # r41f common (helpers only)
sys.path.insert(0, str(W / 'r43o')); sys.modules.pop('common', None)
import common as CO
sys.modules.pop('common', None); sys.path.remove(str(W / 'r43o')); sys.path.insert(0, str(W / 'r43f'))
import common as CF
R38C = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
R37C_ = Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/configs')
QUOTE = '你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我'
units, rows = [], []


def put(cfg, tree, refs):
    cfg.update(code_tree=tree, runner=f'{tree}/train_unit.py', line='R43', oracle=False, test_read=False, label='development experiment (no TEST read in this line)',
               authorisation=dict(user_quote=QUOTE, relayed_by='main session'))
    write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
    rows.append(dict(unit_id=cfg['unit_id'], block=cfg['block'], tree=tree, cond=cfg['cond'], dataset=cfg['dataset'], noise_percent=cfg.get('noise_percent'), parent_seed=cfg.get('parent_seed', cfg.get('parent_H', {}).get('seed')),
                     supervision=cfg['supervision'], adaptation=cfg['adaptation'], objective=cfg['objective'], steps=cfg['steps'], compare_with=refs,
                     config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))


COCO = [('C20_IN2R_H1001', 'c20in_s1001'), ('C60_IN2R_H1001', 'c60in_s1001'), ('C60_IN2R_H1002', 'c60in_s1002'), ('C60_IN2R_H1003', 'c60in_s1003'), ('C80_IN2R_H1001', 'c80in_s1001'), ('C80_IN2R_H1002', 'c80in_s1002'), ('C80_IN2R_H1003', 'c80in_s1003')]
CDESC = {20: 'MS-COCO 20% noise (official NCR noise_index coco_precomp_0.2)', 60: 'MS-COCO 60% injected noise (R25 noise lock coco_precomp_0.6)', 80: 'MS-COCO 80% injected noise (R33 NOISE_LOCK coco_precomp_0.8)'}
FL80 = [('F80_IN2R_H1001', 'f80in', 'R41_C_F80_IN2R_H1001_N_{k}_{o}'), ('F80_IN2R_H1002', 'f80in_s1002', 'R42_N_F80_IN2R_H1002_N_{k}_{o}'), ('F80_IN2R_H1003', 'f80in_s1003', 'R42_N_F80_IN2R_H1003_N_{k}_{o}')]
for o in M.OBJ:
    trust37 = read_json(R37C_ / f'r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_{o}.json')
    trust_fields = dict(supervision='TRUST_NATIVE_NULL', supervision_label='TRUST_NATIVE_NULL',
                        supervision_rule=trust37['supervision_rule'] + ' (R43 B: R37 B1 definition reused exactly; B-TRUST subset of the bound parent, R37 prep.trust_cond rule)',
                        reference_unit=None, reference_joint=None,
                        reference_note='TRUST draws anchors from the B-TRUST pool, so the sampler stream differs from the N units: no in-unit reference; the frozen/joint pair shares head init and stream')
    for kind in ('frozen', 'joint'):
        base = read_json(W / 'configs' / f'R41_C_F80_IN2R_H1001_N_{kind}_{o}.json')
        # ---------------- A: MS-COCO
        for pk, cond in COCO:
            c = CO.CONDS[cond]
            ph = dict(unit=c['h_unit'], model_best=str(Path(c['parent_dir']) / 'model_best.pth.tar'), seed=c['parent_seed'],
                      description=f"{c['h_unit']} (IN2R-native, R33 R25-code configuration)")
            cfg = dict(copy.deepcopy(base), unit_id=f'R43_A_{pk}_N_{kind}_{o}', block='A', cond=cond, noise_percent=c['noise_percent'], parent_seed=c['parent_seed'], parent_H=ph,
                       parent_key=pk, dataset=CDESC[c['noise_percent']], reference_unit=None,
                       evaluation=base['evaluation'].replace('DEV two-stage', 'DEV (native 1K-image split: 1000 images x 5000 captions, per_captions 5; 5K = TEST, never opened) two-stage'),
                       reference_note='new parent: no prior unit; the frozen/joint pair shares init and sampler stream (compared in the summary)')
            put(cfg, 'r43o', '')
        # ---------------- B: TRUST+N on the 80% Flickr parents
        for pk, cond, nfmt in FL80:
            nb = read_json(W / 'configs' / f"{nfmt.format(k=kind, o=o)}.json")
            assert nb['cond'] == cond and nb['supervision'] == 'OBS_MISSING_NULL' and nb['adaptation'] == ('frozen' if kind == 'frozen' else 'both_joint')
            cfg = dict(copy.deepcopy(nb), unit_id=f'R43_B_{pk}_TRUST_{kind}_{o}', block='B', **trust_fields)
            put(cfg, 'r43f', f"n={nfmt.format(k=kind, o=o)}")
    # ---------------- B: TRUST+N on CC152K CC31
    fz = read_json(R38C / f'r38_B_cc_R31_s1001_N_shared_{o}_cosine.json')
    jt = read_json(M.R39 / 'configs' / f'R39_CC152K_N_joint_global_{o}.json')
    cf = M.finish(copy.deepcopy(fz), f'R43_B_CC31_H1001_TRUST_frozen_{o}', 'B', 'frozen')
    cf.update(joint_rule=None, **trust_fields)
    put(cf, 'r43c', f'n=r38_B_cc_R31_s1001_N_shared_{o}_cosine')
    cj = M.joint_fields(M.finish(copy.deepcopy(jt), f'R43_B_CC31_H1001_TRUST_joint_{o}', 'B', 'both_joint'), jt, 'both_joint')
    cj.update(**trust_fields)
    put(cj, 'r43c', f'n=R39_CC152K_N_joint_global_{o}')
write_json(W / 'configs' / 'UNITS_R43.json', units)
with open(W / 'RESOLVED_MATRIX_R43.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R43 configs', len(units), {b: sum(r['block'] == b for r in rows) for b in 'AB'})
