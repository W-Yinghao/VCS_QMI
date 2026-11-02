"""R39X (JOINT confirmation extension; main session relaying the user 「把你觉得值得提交的都提交了吧」): 15 configs = N route x JOINT
(joint_global) x {J, BCE, NCE} on five additional parents: Flickr60 R30-H s1002/s1003, CC152K R31-H0 s1002/s1003, CC152K R37S
R30-style s1001. Each config = the parent's existing N-shared-cosine reference config (R38 / R37S; reused read-only, not rerun) +
exactly the R39 JOINT fields (copied from the frozen R39 N_joint_global config of the same dataset/objective). Writes
configs/R39X_*.json, configs/UNITS_R39X.json, RESOLVED_MATRIX_R39X.csv, REUSE_REFERENCES_R39X.json. The R39 21-unit configs and
protocol are not touched. Usage: python3 -I -B tools/make_configs_x.py"""
import csv, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r39xc'))
from common import sha256, write_json, read_json, utc

R38_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
R38_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
R37S_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/configs')
R37S_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102/critic/runs')
PARENTS = [  # (key, dataset, tree, cond, reference cfg dir, runs dir, reference id format, R39 template dataset)
    ('Flickr60_R30H_s1002', 'Flickr60', 'r39xf', 'n60s1002', R38_CFG, R38_RUNS, 'r38_B_f60_R30_s1002_N_shared_{o}_cosine', 'Flickr60'),
    ('Flickr60_R30H_s1003', 'Flickr60', 'r39xf', 'n60s1003', R38_CFG, R38_RUNS, 'r38_B_f60_R30_s1003_N_shared_{o}_cosine', 'Flickr60'),
    ('CC152K_R31H0_s1002', 'CC152K', 'r39xc', 'cc_s1002', R38_CFG, R38_RUNS, 'r38_B_cc_R31_s1002_N_shared_{o}_cosine', 'CC152K'),
    ('CC152K_R31H0_s1003', 'CC152K', 'r39xc', 'cc_s1003', R38_CFG, R38_RUNS, 'r38_B_cc_R31_s1003_N_shared_{o}_cosine', 'CC152K'),
    ('CC152K_R37S_s1001', 'CC152K', 'r39xc', 'cc_r37s', R37S_CFG, R37S_RUNS, 'r37s_cc_r30h_s1001_N_shared_{o}_cosine', 'CC152K'),
]
JOINT_KEYS = ('encoder', 'joint_rule', 'architecture', 'architecture_rule', 'label')


def main():
    units, rows, refs = [], [], []
    for key, ds, tree, cond, cdir, rdir, fmt, tds in PARENTS:
        for o in ('J', 'BCE', 'NCE'):
            ref_id = fmt.format(o=o); base = read_json(cdir / f'{ref_id}.json'); rd = rdir / ref_id
            assert base['cond'] == cond and base['structure'] == 'G' and base['supervision'] == 'OBS_MISSING_NULL' and base['objective'] == o
            assert base['optimizer']['schedule'] == 'cosine' and base['steps'] == 8000 and base['init_seed'] == 1001 and base['sampler_seed'] == [1001, 3]
            assert read_json(rd / 'DONE.json')['status'] == 'COMPLETE'
            tmpl = read_json(W / 'configs' / f'R39_{tds}_N_joint_global_{o}.json')
            uid = f'R39X_{key}_N_joint_global_{o}'
            ref = dict(unit_id=ref_id, result_json=str(rd / 'RESULT.json'), result_json_sha256=sha256(rd / 'RESULT.json'),
                       train_history_json=str(rd / 'TRAIN_HISTORY.json'), train_history_json_sha256=sha256(rd / 'TRAIN_HISTORY.json'),
                       config_sha256=sha256(cdir / f'{ref_id}.json'),
                       use='R39X reused N-FROZEN reference of this parent: init-state hash (head) and batch-index hash chain asserted equal in-unit (read-only)')
            cfg = {k: v for k, v in base.items() if k not in ('unit_id', 'package_unit', 'package_block', 'job_bundle', 'block', 'reference_unit', 'architecture_rule', 'matrix_source')}
            cfg.update({k: tmpl[k] for k in JOINT_KEYS})
            cfg.update(unit_id=uid, package_unit=None, block='R39X', line='R39X', code_tree=tree, runner=f'{tree}/train_unit.py',
                       supervision='OBS_MISSING_NULL', supervision_label='N', oracle=False, teacher_parents=None, reference_unit=ref, job_bundle=None,
                       optimizer=dict(base['optimizer'], lr_encoder=tmpl['optimizer']['lr_encoder']),
                       evaluation=base['evaluation'] + '; JOINT: phi on the current student vectors, s_H frozen-H native score',
                       extension=dict(name='R39X', parent_key=key, r39_template=f'R39_{tds}_N_joint_global_{o}',
                                      reason='confirm whether the Flickr60 JOINT gain and the CC152K JOINT loss of R39 hold on other parents / the R30-style CC recipe'))
            assert cfg['optimizer'] == dict(tmpl['optimizer']), (uid, cfg['optimizer'], tmpl['optimizer'])
            write_json(W / 'configs' / f'{uid}.json', cfg); units.append(uid)
            rows.append(dict(unit=uid, dataset=ds, parent=key, tree=tree, cond=cond, supervision='OBS_MISSING_NULL', encoder='joint_global', objective=o,
                             reference_unit=ref_id, config_sha256=sha256(W / 'configs' / f'{uid}.json')))
            res = read_json(rd / 'RESULT.json')
            refs.append(dict(r39x_unit=uid, reference_unit=ref_id, result_json_sha256=ref['result_json_sha256'], train_history_sha256=ref['train_history_json_sha256'],
                             config_sha256=ref['config_sha256'], rsum_step8000=res['final']['dev_retrieval']['rsum'], native_H_rsum=res['native_H_dev']['rsum'],
                             init_state_sha256=res['phases'][0]['identity']['init_state_sha256'], batch_index_sha256=res['phases'][0]['identity']['batch_index_sha256'],
                             action='REUSED as the N-FROZEN cell of this parent (not rerun)'))
    # ---------------- R39X addition (main session, same user authorisation): Flickr 0% R30-H f00 s1001 capability diagnostic.
    # No R38 N reference exists at 0%: 3 N-frozen-shared-cosine units (exact R38 recipe, R38 s1001 config with cond n00) + 3 JOINT units;
    # no in-unit reference (the frozen partner is new); init-state / batch-chain identity of each pair is compared in the summary.
    PH = dict(unit='r30_f00_s1001_H0', model_best='/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101/runs/r30_f00_s1001_H0/model_best.pth.tar', seed=1001)
    for o in ('J', 'BCE', 'NCE'):
        base = read_json(R38_CFG / f'r38_B_f60_R30_s1001_N_shared_{o}_cosine.json'); tmpl = read_json(W / 'configs' / f'R39_Flickr60_N_joint_global_{o}.json')
        assert base['cond'] == 'n60' and base['optimizer']['schedule'] == 'cosine'
        common_ = {k: v for k, v in base.items() if k not in ('unit_id', 'package_unit', 'package_block', 'job_bundle', 'block', 'reference_unit', 'architecture_rule', 'matrix_source')}
        common_.update(cond='n00', noise_percent=0, dataset='Flickr30K 0% (identity observed map)', parent_H=PH, parent_key='f00_R30_s1001', package_unit=None, block='R39X',
                       line='R39X', code_tree='r39xf', runner='r39xf/train_unit.py', supervision='OBS_MISSING_NULL', supervision_label='N', oracle=False, teacher_parents=None,
                       reference_unit=None, job_bundle=None, reference_note='no prior N-frozen unit at 0%: R39X_Flickr00_R30H_s1001_N_frozen_shared_<obj> is the matched frozen cell')
        fr = dict(common_, unit_id=f'R39X_Flickr00_R30H_s1001_N_frozen_shared_{o}', encoder='frozen', architecture='shared',
                  architecture_rule='shared G (R35/R36 GHead): g = a s + b + w^T h(phi) + c',
                  extension=dict(name='R39X', parent_key='Flickr00_R30H_s1001', role='N-frozen-shared-cosine reference (R38 recipe) for the 0% capability diagnostic'))
        jt = dict(common_, **{k: tmpl[k] for k in JOINT_KEYS}, unit_id=f'R39X_Flickr00_R30H_s1001_N_joint_global_{o}', optimizer=dict(base['optimizer'], lr_encoder=tmpl['optimizer']['lr_encoder']),
                  evaluation=base['evaluation'] + '; JOINT: phi on the current student vectors, s_H frozen-H native score',
                  extension=dict(name='R39X', parent_key='Flickr00_R30H_s1001', r39_template=f'R39_Flickr60_N_joint_global_{o}',
                                 reason='capability diagnostic: does the Flickr JOINT gain also appear without noise'))
        assert jt['optimizer'] == tmpl['optimizer']
        for cfg in (fr, jt):
            write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
            rows.append(dict(unit=cfg['unit_id'], dataset='Flickr00', parent='Flickr00_R30H_s1001', tree='r39xf', cond='n00', supervision='OBS_MISSING_NULL',
                             encoder=cfg['encoder'], objective=o, reference_unit='(new frozen cell in R39X)' if cfg is jt else '(none: this is the frozen cell)',
                             config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))
    write_json(W / 'configs' / 'UNITS_R39X.json', units)
    with open(W / 'RESOLVED_MATRIX_R39X.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(W / 'REUSE_REFERENCES_R39X.json', dict(utc=utc(), references=refs))
    print('R39X configs', len(units)); assert len(units) == 21


if __name__ == '__main__':
    main()
