"""R40: resolve the package MATRIX.csv (27 rows) into configs/<unit>.json, RESOLVED_MATRIX.csv, PARENT_BINDING.json and REUSE_REFERENCES.json.
Every config is the frozen R39 N-JOINT config of the same parent/objective (itself = the R38 N-shared-cosine config + the R39 JOINT fields),
so head init seed, sampler stream, n_queries, steps, checkpoint steps, clip, AdamW, both lr groups and the cosine schedule are byte-for-byte
those of R39; R40 changes only:
  A: adaptation image_only / text_only (student modalities; the other modality stays the parent, requires_grad False, not in the optimizer)
  B: supervision N_PRESENT_ONLY x {frozen (head only: encoder frozen, no lr_encoder, R39 frozen evaluation string), both_joint}
  C: Flickr ORACLE_MISSING (R39 oracle rule, bit-identical module) x both_joint, oracle runner
and bookkeeping. In-unit read-only references: R38 N-FROZEN (cfg reference_unit, R38 rule), R39 N-JOINT (reference_joint: head/null init,
student init state from its ckpt_step0 lineage, batch chain), R39 ORACLE-FROZEN (reference_oracle, C only).
Usage: /projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -I -B tools/make_configs.py"""
import csv, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r40c'))
import torch
from common import sha256, write_json, read_json, utc, R40_PACKAGE, ADAPTATIONS

R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
R39_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')
DS = {'Flickr60': dict(tree='r40f', cond='n60', parent='F60_R30_H1001', r39='Flickr60', parent_desc='R30-H f60 s1001 (r30_f60_s1001_H0), 60% injected noise; native H DEV rSum 505.2071'),
      'CC152K': dict(tree='r40c', cond='cc', parent='CC31_H1001', r39='CC152K', parent_desc='R31 H0_default cc_s1001 (IN2R author defaults), CC152K real observed relation; native H DEV rSum 389.70')}
SUP = {'N': 'OBS_MISSING_NULL', 'N_PRESENT_ONLY': 'N_PRESENT_ONLY', 'ORACLE_MISSING': 'ORACLE_MISSING'}
PRESENT_RULE = ('R40 B N_PRESENT_ONLY: the R38/R39 N lists unchanged (target natural top-32 + null, 33 slots, all permuted, Q = Uniform(33)); a row enters the loss only '
                'if >= 1 real slot carries observed-positive mass; absent rows (A_null = 1) are masked entirely (null slot included), never redrawn; per direction '
                'L_d = mean_i[m_i L_i] / p_d with p_d = Pr(observed partner present) fixed once from the full FIT observed relation and the fixed candidate table '
                '(data36.presence_probs, recomputed in-unit), never from the batch; conditional-on-present P_d/Q_d, J form unchanged; DEV inference identical to N')
JOINT_TRAINABLE = {'image_only': 'image fc, MLP (BN affine; running stats frozen), GPO GRU+linear (text encoder = parent, requires_grad False, not in the optimizer)',
                   'text_only': 'text BiGRU, fc, MLP (BN affine; running stats frozen), GPO GRU+linear (image encoder = parent, requires_grad False, not in the optimizer)',
                   'both_joint': 'image fc, MLP (BN affine; running stats frozen), GPO GRU+linear; text BiGRU, fc, MLP, GPO GRU+linear (R39 N-JOINT)'}


def ref_entry(unit_dir, unit_id, use, with_student=False):
    e = dict(unit_id=unit_id, result_json=str(unit_dir / 'RESULT.json'), result_json_sha256=sha256(unit_dir / 'RESULT.json'),
             train_history_json=str(unit_dir / 'TRAIN_HISTORY.json'), train_history_json_sha256=sha256(unit_dir / 'TRAIN_HISTORY.json'), use=use)
    assert read_json(unit_dir / 'DONE.json')['status'] == 'COMPLETE'
    if with_student:
        ck = torch.load(unit_dir / 'ckpt_step0.pt', map_location='cpu', weights_only=False)
        assert ck['step'] == 0 and ck['unit_id'] == unit_id
        e.update(ckpt_step0=str(unit_dir / 'ckpt_step0.pt'), ckpt_step0_sha256=sha256(unit_dir / 'ckpt_step0.pt'),
                 student_init_state_sha256=ck['lineage']['student']['init_state_sha256'], student_parent_sha256=ck['lineage']['student']['parent_sha256'])
    return e


def main():
    rows = list(csv.DictReader(open(R40_PACKAGE / 'MATRIX.csv')))
    assert len(rows) == 27 and len({r['unit_id'] for r in rows}) == 27
    units, out_rows, reuse = [], [], {}
    for r in rows:
        P = DS[r['dataset']]; o = r['objective']; ad = r['adaptation']; sup = SUP[r['supervision']]
        assert r['parent'] == P['parent'] and r['steps'] == '8000' and r['test_read'] == 'false' and ad in ADAPTATIONS
        tmpl_id = f"R39_{P['r39']}_N_joint_global_{o}"
        tmpl = read_json(R39 / 'configs' / f'{tmpl_id}.json')
        assert sha256(R39 / 'configs' / f'{tmpl_id}.json') == read_json(R39 / 'PROTOCOL_FROZEN.json')['config_sha256'][tmpl_id]
        assert tmpl['steps'] == 8000 and tmpl['optimizer']['schedule'] == 'cosine' and tmpl['init_seed'] == 1001 and tmpl['sampler_seed'] == [1001, 3] and tmpl['n_queries'] == 16
        cfg = {k: v for k, v in tmpl.items() if k not in ('unit_id', 'package_unit', 'block', 'line', 'code_tree', 'runner', 'job_bundle')}
        joint = ad != 'frozen'; oracle = r['oracle'] == 'true'
        cfg.update(unit_id=r['unit_id'], package_unit=r['unit_id'], block=r['block'], line='R40', code_tree=P['tree'],
                   runner=f"{P['tree']}/{'train_oracle.py' if oracle else 'train_unit.py'}", supervision=sup, supervision_label=r['supervision'], oracle=oracle,
                   adaptation=ad, student_modalities=list(ADAPTATIONS[ad]), encoder='joint_global' if joint else 'frozen', parent_key=P['parent'],
                   label='development experiment (no TEST read in this line)', job_bundle=None, r39_template=tmpl_id)
        if sup == 'N_PRESENT_ONLY':
            cfg['supervision_rule'] = PRESENT_RULE
        elif sup == 'ORACLE_MISSING':
            oc = read_json(R39 / 'configs' / f'R39_Flickr60_ORACLE_MISSING_frozen_{o}.json')
            cfg['supervision_rule'] = oc['supervision_rule'] + ' (R40 C: bit-identical R39 oracle_truth.py; both_joint global update)'
        if joint:
            cfg['joint_rule'] = dict(tmpl['joint_rule'], trainable=JOINT_TRAINABLE[ad], adaptation=ad,
                                     frozen=tmpl['joint_rule']['frozen'] + ('' if ad == 'both_joint' else f"; the {'text' if ad == 'image_only' else 'image'} encoder (whole modality, parent weights)"),
                                     clip=tmpl['joint_rule']['clip'] + '; R40 also records head / null / image / text pre-clip norms per peer and the shared clip multiplier')
        else:
            fz = read_json(R39 / 'configs' / f"R39_{P['r39']}_COMPLETE_frozen_{o}.json")
            cfg['joint_rule'] = None
            cfg['optimizer'] = fz['optimizer']; cfg['evaluation'] = fz['evaluation']
            assert 'lr_encoder' not in cfg['optimizer'] and {k: v for k, v in tmpl['optimizer'].items() if k != 'lr_encoder'} == cfg['optimizer']
        # references
        ref38 = tmpl['reference_unit']   # R38 N-FROZEN (as in R39): head init + batch chain, R38 rule
        assert sha256(ref38['result_json']) == ref38['result_json_sha256'] and sha256(ref38['train_history_json']) == ref38['train_history_json_sha256']
        cfg['reference_unit'] = dict(ref38, use='R40 reused N-FROZEN reference (R38): init-state hash (head) and batch-index hash chain asserted equal in-unit (read-only)')
        refs = dict(n_frozen=ref38['unit_id'])
        if joint:
            cfg['reference_joint'] = ref_entry(R39_OUT / 'runs' / tmpl_id, tmpl_id, 'R40 reused N-JOINT reference (R39): head/null init hash, student init-state hash and batch chain asserted equal in-unit (read-only)', True)
            refs['n_joint'] = tmpl_id
        if oracle:
            oid = f'R39_Flickr60_ORACLE_MISSING_frozen_{o}'
            cfg['reference_oracle'] = ref_entry(R39_OUT / 'oracle_runs' / oid, oid, 'R40 reused ORACLE-FROZEN reference (R39, oracle process only): head/null init hash and batch chain asserted equal in-unit (read-only)')
            refs['oracle_frozen'] = oid
        units.append(r['unit_id']); reuse[r['unit_id']] = refs
        write_json(W / 'configs' / f"{r['unit_id']}.json", cfg)
        out_rows.append(dict(unit_id=r['unit_id'], block=r['block'], dataset=r['dataset'], parent=r['parent'], tree=P['tree'], cond=P['cond'], supervision=sup,
                             adaptation=ad, student_modalities='+'.join(ADAPTATIONS[ad]) or '-', objective=o, steps=8000, oracle=oracle, runner=cfg['runner'],
                             ref_n_frozen=ref38['unit_id'], ref_n_joint=refs.get('n_joint', ''), ref_oracle_frozen=refs.get('oracle_frozen', ''),
                             r39_template=tmpl_id, test_read=False, config_sha256=sha256(W / 'configs' / f"{r['unit_id']}.json")))
    write_json(W / 'configs' / 'UNITS.json', units)
    with open(W / 'RESOLVED_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    # parent binding
    pb = {}
    for ds, P in DS.items():
        t = read_json(R39 / 'configs' / f"R39_{P['r39']}_N_joint_global_J.json")
        jm = read_json(R39_OUT / 'prep' / ('f60' if ds == 'Flickr60' else 'cc') / 'JOINT_PREP_MANIFEST.json')
        res = read_json(R39_OUT / 'runs' / f"R39_{P['r39']}_N_joint_global_J" / 'RESULT.json')
        pb[P['parent']] = dict(dataset=ds, description=P['parent_desc'], cond=P['cond'], parent_H=t['parent_H'], model_best_sha256=jm['parent']['model_best_sha256'],
                               native_H_dev_rsum=res['native_H_dev']['rsum'], parent_record=res['parent'],
                               joint_prep_manifest=str(R39_OUT / 'prep' / ('f60' if ds == 'Flickr60' else 'cc') / 'JOINT_PREP_MANIFEST.json'),
                               joint_prep_manifest_sha256=sha256(R39_OUT / 'prep' / ('f60' if ds == 'Flickr60' else 'cc') / 'JOINT_PREP_MANIFEST.json'),
                               candidates='frozen-H natural top-32 of this parent (R35/R37C prep, read-only), H-top100 two-stage DEV, s_H native skip; same as R38/R39',
                               centring='R39 registered patch: parent per-native-batch centring vectors (R39 prep, read-only)',
                               units=[u for u in units if read_json(W / 'configs' / f'{u}.json')['cond'] == P['cond']],
                               note='fixed R39 parents (package section 2); no other parent is substituted into these cells (R39X covers other parents)')
    write_json(W / 'PARENT_BINDING.json', dict(utc=utc(), parents=pb))
    write_json(W / 'REUSE_REFERENCES.json', dict(utc=utc(), by_unit=reuse))
    print('configs', len(units))


if __name__ == '__main__':
    main()
