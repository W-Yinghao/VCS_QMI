"""R41-F (amendment to R41; main session relaying the same user authorisation 「…只要你觉得有用的就提交，不用问我」): CC152K CC31 parent (R31 H0_default
cc_s1001), N x both_joint x J/BCE/NCE, identical to the R39 CC N-JOINT recipe except the student-encoder lr is 10x lower (1e-6 -> 1e-7 on the
common cosine; head lr 1e-4 -> 1e-5 unchanged). One variant only (not a grid). In-unit references: R38 CC N-FROZEN (init + chain, R38 rule) and
R39 CC N-JOINT (head/null init, student init, chain; optimizer compared except lr_encoder). Writes configs/R41_F_*.json, configs/UNITS_R41F.json,
RESOLVED_MATRIX_R41F.csv. Usage: /projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -I -B tools/make_configs_r41f.py"""
import copy, csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r41c'))
import torch
from common import sha256, write_json, read_json, ADAPTATIONS
R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
R39R = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/runs')
units, rows = [], []
for o in ('J', 'BCE', 'NCE'):
    tid = f'R39_CC152K_N_joint_global_{o}'; t = read_json(R39 / 'configs' / f'{tid}.json'); jd = R39R / tid
    assert t['optimizer']['lr_encoder'] == 1e-5 and t['cond'] == 'cc'
    ck = torch.load(jd / 'ckpt_step0.pt', map_location='cpu', weights_only=False)
    cfg = {k: v for k, v in copy.deepcopy(t).items() if k not in ('package_unit', 'teacher_parents', 'job_bundle')}
    uid = f'R41_F_CC31_H1001_N_joint_enclr0p1_{o}'
    cfg.update(unit_id=uid, package_unit=None, block='F', line='R41', code_tree='r41c', runner='r41c/train_unit.py', oracle=False, adaptation='both_joint',
               student_modalities=list(ADAPTATIONS['both_joint']), encoder='joint_global', test_read=False, job_bundle=None, r39_template=tid,
               optimizer=dict(t['optimizer'], lr_encoder=1e-6),
               joint_rule=dict(t['joint_rule'], lr='head 1e-4 -> 1e-5, student 1e-6 -> 1e-7 (R41-F: 10x lower than R39), common cosine multiplier, no warmup, optimizer never reset; AdamW wd 1e-4 both groups'),
               amendment=dict(name='R41-F', reason='on CC152K every encoder-update arm (R39, R39X x3 parents, R40 image/text/present) is <= N-frozen while Flickr gains; '
                                                    'tests whether the CC drop is an encoder step-size effect (one variant, not a grid)',
                              user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session'),
               reference_joint=dict(unit_id=tid, result_json=str(jd / 'RESULT.json'), result_json_sha256=sha256(jd / 'RESULT.json'), train_history_json=str(jd / 'TRAIN_HISTORY.json'),
                                    train_history_json_sha256=sha256(jd / 'TRAIN_HISTORY.json'), mode='encoder_lr_variant', ckpt_step0=str(jd / 'ckpt_step0.pt'),
                                    ckpt_step0_sha256=sha256(jd / 'ckpt_step0.pt'), student_init_state_sha256=ck['lineage']['student']['init_state_sha256'],
                                    use='R41-F reused N-JOINT reference (R39 CC): head/null init, student init and batch chain asserted in-unit; optimizer compared except lr_encoder'))
    write_json(W / 'configs' / f'{uid}.json', cfg); units.append(uid)
    rows.append(dict(unit_id=uid, block='F', cond='cc', supervision=cfg['supervision'], adaptation='both_joint', objective=o, steps=cfg['steps'], lr_head=cfg['optimizer']['lr_head'],
                     lr_encoder=cfg['optimizer']['lr_encoder'], ref_n_frozen=cfg['reference_unit']['unit_id'], ref_n_joint=tid, config_sha256=sha256(W / 'configs' / f'{uid}.json')))
write_json(W / 'configs' / 'UNITS_R41F.json', units)
with open(W / 'RESOLVED_MATRIX_R41F.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R41-F configs', units)
