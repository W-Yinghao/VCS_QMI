"""R41 configs (main session relaying the user: 「…我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我」). 39 units, all on the r41f tree
(R40 recipe unless stated; matched J/BCE/NCE per cell):
  A  budget: N x both_joint x 16000 updates (same cosine shape stretched to 16000; readouts 0/250/.../8000/16000; primary 16000, 8000 also
     reported) on Flickr60 R30-H s1001/s1002/s1003 and Flickr 0% R30-H f00_s1001 (12). Prefix reference = the 8000-update R39/R39X N-JOINT
     unit (same init, same sampler stream): head/null init, student init and the 7 shared batch-chain readouts asserted in-unit.
  B  run-to-run variance: N x both_joint x 8000 on Flickr60 R30-H s1001 with training seed 2002 (head init seed 2002, sampler [2002, 3]) (3).
  C  IN2R-native Flickr parents R31 H0_default f40_s1001 and R33 H0 f80_s1001 x {N-frozen-shared-cosine, N-both_joint} (12); configs
     = the R38 / R39 Flickr60 s1001 configs with the parent swapped; no in-unit reference (new parents). Frozen later (PROTOCOL_FROZEN_R41C).
  D  ORACLE_MISSING (R39 rule, isolated process) x {frozen, both_joint} on Flickr60 R30-H s1002 (6); frozen reference: R38 N-frozen s1002
     (init/chain); joint reference: R39X N-JOINT s1002.
  E  image_only on Flickr60 R30-H s1002/s1003 (6); references R38 N-frozen + R39X N-JOINT of the parent.
Writes configs/R41_*.json, configs/UNITS_R41.json (part 1: A/B/D/E), configs/UNITS_R41C.json (part 2: C), RESOLVED_MATRIX_R41.csv,
REUSE_REFERENCES_R41.json. Usage: /projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -I -B tools/make_configs_r41.py"""
import copy, csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r41f'))
import torch
from common import sha256, write_json, read_json, utc, ADAPTATIONS, CONDS

R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
R38C = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
R38R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
R39R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
OBJ = ('J', 'BCE', 'NCE')
PAR = {  # key: (cond, R39/R39X N-JOINT unit format, R38 N-frozen unit format or None)
    'F60_R30_H1001': ('n60', 'runs/R39_Flickr60_N_joint_global_{o}', 'r38_B_f60_R30_s1001_N_shared_{o}_cosine'),
    'F60_R30_H1002': ('n60s1002', 'runs_r39x/R39X_Flickr60_R30H_s1002_N_joint_global_{o}', 'r38_B_f60_R30_s1002_N_shared_{o}_cosine'),
    'F60_R30_H1003': ('n60s1003', 'runs_r39x/R39X_Flickr60_R30H_s1003_N_joint_global_{o}', 'r38_B_f60_R30_s1003_N_shared_{o}_cosine'),
    'F00_R30_H1001': ('n00', 'runs_r39x/R39X_Flickr00_R30H_s1001_N_joint_global_{o}', None),
}
NEWP = {'F40_IN2R_H1001': ('f40in', 'Flickr30K 40% injected noise (R25 noise map 0.4)', 'R31 H0_default f40_s1001 (IN2R author run.py defaults; DEV 514.18)'),
        'F80_IN2R_H1001': ('f80in', 'Flickr30K 80% injected noise (R33 NOISE_LOCK map 0.8)', 'R33 H0 f80_s1001 (R25 IN2R code configuration; DEV 464.93)')}
JT = {'image_only': 'image fc, MLP (BN affine; running stats frozen), GPO GRU+linear (text encoder = parent, requires_grad False, not in the optimizer)',
      'both_joint': 'image fc, MLP (BN affine; running stats frozen), GPO GRU+linear; text BiGRU, fc, MLP, GPO GRU+linear (R39 N-JOINT)'}


def ref(unit_dir, use, student=False, mode=None):
    unit_dir = Path(unit_dir); assert read_json(unit_dir / 'DONE.json')['status'] == 'COMPLETE'
    e = dict(unit_id=unit_dir.name, result_json=str(unit_dir / 'RESULT.json'), result_json_sha256=sha256(unit_dir / 'RESULT.json'),
             train_history_json=str(unit_dir / 'TRAIN_HISTORY.json'), train_history_json_sha256=sha256(unit_dir / 'TRAIN_HISTORY.json'), use=use)
    if mode:
        e['mode'] = mode
    if student:
        ck = torch.load(unit_dir / 'ckpt_step0.pt', map_location='cpu', weights_only=False)
        assert ck['step'] == 0 and ck['unit_id'] == unit_dir.name
        e.update(ckpt_step0=str(unit_dir / 'ckpt_step0.pt'), ckpt_step0_sha256=sha256(unit_dir / 'ckpt_step0.pt'), student_init_state_sha256=ck['lineage']['student']['init_state_sha256'])
    return e


def r38ref(uid):
    return dict(unit_id=uid, result_json=str(R38R / uid / 'RESULT.json'), result_json_sha256=sha256(R38R / uid / 'RESULT.json'),
                train_history_json=str(R38R / uid / 'TRAIN_HISTORY.json'), train_history_json_sha256=sha256(R38R / uid / 'TRAIN_HISTORY.json'),
                config_sha256=sha256(R38C / f'{uid}.json'), use='R41 reused N-FROZEN reference (R38): init-state hash (head) and batch-index hash chain asserted equal in-unit (read-only)')


def finish(cfg, uid, block, ad, oracle=False):
    cfg = {k: v for k, v in cfg.items() if k not in ('package_unit', 'package_block', 'job_bundle', 'matrix_source', 'extension', 'teacher_parents', 'r39_template')}
    cfg.update(unit_id=uid, package_unit=None, block=block, line='R41', code_tree='r41f', runner=f"r41f/{'train_oracle.py' if oracle else 'train_unit.py'}", oracle=oracle,
               adaptation=ad, student_modalities=list(ADAPTATIONS[ad]), encoder='frozen' if ad == 'frozen' else 'joint_global', test_read=False,
               label='development experiment (no TEST read in this line)', job_bundle=None)
    return cfg


def joint_fields(cfg, tmpl, ad):
    cfg['optimizer'] = copy.deepcopy(tmpl['optimizer'])
    cfg['joint_rule'] = dict(tmpl['joint_rule'], trainable=JT[ad], adaptation=ad,
                             clip=tmpl['joint_rule']['clip'].split('; R40')[0] + '; R40/R41 also record head / null / image / text pre-clip norms per peer and the shared clip multiplier')
    if ad == 'image_only':
        cfg['joint_rule']['frozen'] = tmpl['joint_rule']['frozen'] + '; the text encoder (whole modality, parent weights)'
    if 'JOINT:' not in cfg['evaluation']:
        cfg['evaluation'] = cfg['evaluation'] + '; JOINT: phi on the current student vectors, s_H frozen-H native score'
    return cfg


def main():
    t39 = {o: read_json(R39 / 'configs' / f'R39_Flickr60_N_joint_global_{o}.json') for o in OBJ}
    units1, units2, rows, reuse = [], [], [], {}
    def put(cfg, part, refs):
        write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); (units1 if part == 1 else units2).append(cfg['unit_id']); reuse[cfg['unit_id']] = refs
        rows.append(dict(unit_id=cfg['unit_id'], block=cfg['block'], cond=cfg['cond'], parent_key=cfg.get('parent_key'), supervision=cfg['supervision'], adaptation=cfg['adaptation'],
                         objective=cfg['objective'], steps=cfg['steps'], init_seed=cfg['init_seed'], sampler_seed=str(cfg['sampler_seed']), oracle=cfg['oracle'], runner=cfg['runner'],
                         freeze_part=part, refs=';'.join(f'{k}={v}' for k, v in refs.items()), config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))
    for o in OBJ:
        # ---------------- A: 16000-update budget
        for pk, (cond, jfmt, ffmt) in PAR.items():
            jdir = R39R / jfmt.format(o=o); base = read_json(R39 / 'configs' / f'{jdir.name}.json')
            cfg = finish(base, f'R41_A_{pk}_N_joint_16k_{o}', 'A', 'both_joint')
            cfg = joint_fields(cfg, t39[o], 'both_joint')
            assert cfg['optimizer'] == t39[o]['optimizer'] and cfg['cond'] == cond and cfg['steps'] == 8000
            cfg.update(steps=16000, ckpt_steps=[0, 250, 500, 1000, 2000, 4000, 8000, 16000], primary_endpoint='fixed_step_16000 (step 8000 also reported)',
                       schedule_rule='common cosine stretched to 16000 updates: lr(t) = base*[0.1+0.45*(1+cos(pi*t/15999))], t = 0..15999; head 1e-4 -> 1e-5, student 1e-5 -> 1e-6',
                       reference_unit=None,
                       reference_joint=ref(jdir, 'R41 prefix reference (8000-update N-JOINT of the same parent / stream): head/null init, student init, batch chain at its 7 readouts', True, 'prefix'),
                       budget_rule='same init / sampler stream as the 8000-update unit; only the update budget (and hence the stretched cosine) changes')
            put(cfg, 1, dict(n_joint_prefix=jdir.name))
        # ---------------- B: training-seed replicate (s1001)
        jdir = R39R / PAR['F60_R30_H1001'][1].format(o=o); base = read_json(R39 / 'configs' / f'{jdir.name}.json')
        cfg = joint_fields(finish(base, f'R41_B_F60_R30_H1001_seed2002_N_joint_{o}', 'B', 'both_joint'), t39[o], 'both_joint')
        cfg.update(init_seed=2002, sampler_seed=[2002, 3], reference_unit=None, seed_rule='R41 B: training seed 2002 (head init build_heads(..., 2002), sampler [2002, 3]); parent, prep, recipe unchanged',
                   head_init='fresh R35 build_heads(G, a0, b0, 2002): FIT-only calibration a0*s+b0 + zero-init residual; null slots init 0')
        put(cfg, 1, dict(compare_with=jdir.name))
        # ---------------- C: IN2R-native parents
        fz38 = read_json(R38C / f'r38_B_f60_R30_s1001_N_shared_{o}_cosine.json')
        for pk, (cond, dsdesc, pdesc) in NEWP.items():
            c = CONDS[cond]
            ph = dict(unit=c['h_unit'], model_best=str(Path(c['parent_dir']) / 'model_best.pth.tar'), seed=1001, description=pdesc)
            common_ = dict(cond=cond, noise_percent=c['noise_percent'], parent_seed=1001, parent_H=ph, parent_key=pk, dataset=dsdesc, reference_unit=None,
                           reference_note='new parent: no prior unit; the frozen/joint pair shares init and sampler stream (compared in the summary)')
            fr = finish(dict(fz38, **common_), f'R41_C_{pk}_N_frozen_{o}', 'C', 'frozen')
            fr['joint_rule'] = None; fr['architecture_rule'] = 'shared G (R35/R36 GHead): g = a s_H + b + w^T h(phi(u, v, s_H)) + c'
            put(fr, 2, {})
            jt = joint_fields(finish(dict(fz38, **common_), f'R41_C_{pk}_N_joint_{o}', 'C', 'both_joint'), t39[o], 'both_joint')
            put(jt, 2, {})
        # ---------------- D: ORACLE_MISSING x {frozen, joint} on s1002
        oc = read_json(R39 / 'configs' / f'R39_Flickr60_ORACLE_MISSING_frozen_{o}.json')
        f38 = read_json(R38C / f'r38_B_f60_R30_s1002_N_shared_{o}_cosine.json')
        ofr = finish(f38, f'R41_D_F60_R30_H1002_ORACLE_frozen_{o}', 'D', 'frozen', oracle=True)
        ofr.update(supervision='ORACLE_MISSING', supervision_label='ORACLE_MISSING', supervision_rule=oc['supervision_rule'] + ' (R41 D: parent R30-H s1002; bit-identical R39 oracle_truth.py)',
                   joint_rule=None, reference_unit=r38ref(f'r38_B_f60_R30_s1002_N_shared_{o}_cosine'))
        put(ofr, 1, dict(n_frozen=f'r38_B_f60_R30_s1002_N_shared_{o}_cosine'))
        jdir = R39R / PAR['F60_R30_H1002'][1].format(o=o); xb = read_json(R39 / 'configs' / f'{jdir.name}.json')
        oj = joint_fields(finish(xb, f'R41_D_F60_R30_H1002_ORACLE_joint_{o}', 'D', 'both_joint', oracle=True), t39[o], 'both_joint')
        oj.update(supervision='ORACLE_MISSING', supervision_label='ORACLE_MISSING', supervision_rule=ofr['supervision_rule'],
                  reference_unit=r38ref(f'r38_B_f60_R30_s1002_N_shared_{o}_cosine'),
                  reference_joint=ref(jdir, 'R41 reused N-JOINT reference (R39X s1002): head/null init, student init, batch chain asserted in-unit', True))
        put(oj, 1, dict(n_frozen=f'r38_B_f60_R30_s1002_N_shared_{o}_cosine', n_joint=jdir.name))
        # ---------------- E: image_only on s1002 / s1003
        for pk in ('F60_R30_H1002', 'F60_R30_H1003'):
            cond, jfmt, ffmt = PAR[pk]; jdir = R39R / jfmt.format(o=o)
            xb = read_json(R39 / 'configs' / f'{jdir.name}.json')
            e = joint_fields(finish(xb, f'R41_E_{pk}_image_only_{o}', 'E', 'image_only'), t39[o], 'image_only')
            e.update(reference_unit=r38ref(ffmt.format(o=o)), reference_joint=ref(jdir, 'R41 reused N-JOINT reference (R39X): head/null init, student init, batch chain asserted in-unit', True))
            put(e, 1, dict(n_frozen=ffmt.format(o=o), n_joint=jdir.name))
    write_json(W / 'configs' / 'UNITS_R41.json', units1); write_json(W / 'configs' / 'UNITS_R41C.json', units2)
    with open(W / 'RESOLVED_MATRIX_R41.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(W / 'REUSE_REFERENCES_R41.json', dict(utc=utc(), by_unit=reuse))
    print('R41 configs', len(units1), '+', len(units2))


if __name__ == '__main__':
    main()
