"""R38: resolve the package configs/MATRIX_NEW.csv (30 rows) into per-unit configs (configs/<unit_id>.json), RESOLVED_MATRIX.csv
and REUSE_REFERENCES.json. Every config starts from the config of the read-only shared-constant reference unit of the same
parent/route/objective (R36 for Flickr60 R30-H1001, R37C for CC152K R31-H1001; R37C N_G for the A block parents), so head-init
seed, sampler stream, n_queries, steps, readouts, clip, AdamW settings are unchanged; R38 changes only structure (G/GD), the
schedule, the cond/parent (A block) and bookkeeping. Usage: python -I -B tools/make_configs.py"""
import csv, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r38c'))
from common import sha256, write_json, read_json, utc, PACKAGE, CONDS as CC_CONDS

R36_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102/configs')
R36_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r36_supervision_20261102/runs')
R37_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/configs')
R37_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/runs')
R37C_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/configs')
R37C_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/runs')
PARENT = {'f60_R30_s1001': dict(tree='r38f', cond='n60', dataset='Flickr30K 60% injected noise', ref_cfg=R36_CFG, ref_runs=R36_RUNS, ref_fmt='r36_{r}_n60_G_{o}'),
          'f60_R30_s1002': dict(tree='r38f', cond='n60s1002', dataset='Flickr30K 60% injected noise', ref_cfg=R37_CFG, ref_runs=R37_RUNS, ref_fmt='r37_confirm_f60_h1002_{r}_G_{o}'),
          'f60_R30_s1003': dict(tree='r38f', cond='n60s1003', dataset='Flickr30K 60% injected noise', ref_cfg=R37_CFG, ref_runs=R37_RUNS, ref_fmt='r37_confirm_f60_h1003_{r}_G_{o}'),
          'cc_R31_s1001': dict(tree='r38c', cond='cc', dataset='CC152K (real noise)', ref_cfg=R37C_CFG, ref_runs=R37C_RUNS, ref_fmt='r37c_pilot_cc_h1001_{r}_G_{o}'),
          'cc_R31_s1002': dict(tree='r38c', cond='cc_s1002', dataset='CC152K (real noise)', ref_cfg=R37C_CFG, ref_runs=None, ref_fmt='r37c_pilot_cc_h1001_{r}_G_{o}'),
          'cc_R31_s1003': dict(tree='r38c', cond='cc_s1003', dataset='CC152K (real noise)', ref_cfg=R37C_CFG, ref_runs=None, ref_fmt='r37c_pilot_cc_h1001_{r}_G_{o}')}
SUP = {'N': 'OBS_MISSING_NULL', 'T': 'TRUST_HARD'}


def main():
    rows = list(csv.DictReader(open(PACKAGE / 'configs' / 'MATRIX_NEW.csv')))
    assert len(rows) == 30 and len({r['unit_id'] for r in rows}) == 30
    amend = list(csv.DictReader(open(W / 'MATRIX_AMENDMENT.csv')))   # user amendment (completeness): package B/C cells on the other parents
    assert len(amend) == 54 and not {r['unit_id'] for r in rows} & {r['unit_id'] for r in amend}
    rows = rows + amend
    out, uids, refs = [], [], {}
    for r in rows:
        P = PARENT[r['parent_key']]
        assert r['steps'] == '8000' and r['encoder_update'] == 'false' and r['test_read'] == 'false' and float(r['head_lr']) == 1e-4
        assert (r['schedule'], float(r['cosine_floor'])) in (('constant', 1.0), ('cosine', 0.1))
        ref_id = P['ref_fmt'].format(r=r['route'], o=r['objective'])
        base = read_json(P['ref_cfg'] / f'{ref_id}.json')
        assert base['structure'] == 'G' and base['supervision'] == SUP[r['route']] and base['objective'] == r['objective'] and base['optimizer']['schedule'] == 'constant'
        cfg = {k: v for k, v in base.items() if k not in ('unit_id', 'package_unit', 'package_block', 'job_bundle', 'block')}
        cfg.update(unit_id=r['unit_id'], package_unit=r['unit_id'], package_block=r['block'], block=r['block'][0], line='R38',
                   cond=P['cond'], code_tree=P['tree'], dataset=P['dataset'], parent_key=r['parent_key'], route=r['route'], architecture=r['architecture'],
                   structure='GD' if r['architecture'] == 'directional' else 'G',
                   optimizer=dict(base['optimizer'], schedule=r['schedule'], cosine_floor=float(r['cosine_floor'])),
                   schedule_rule=('constant lr 1e-4 (R36/R37 protocol)' if r['schedule'] == 'constant' else
                                  'common cosine: lr(t) = 1e-4*[0.1+0.45*(1+cos(pi*t/7999))], t = 0..7999 update index; no warmup, optimizer never reset; same for J/BCE/NCE'),
                   architecture_rule=('shared G (R35/R36 GHead): g = a s + b + w^T h(phi) + c' if r['architecture'] == 'shared' else
                                      'directional G (R38 DirGHead): shared 4097-256-128 trunk, per-direction 128->1 output (zero init) and positive affine (a0/b0); +131 params/peer; i2t rows scored with g_i2t, t2i rows with g_t2i, in training, DEV inference and FIT diagnostics'),
                   evaluation=base['evaluation'] + ('; directional: i2t candidates with g_i2t, t2i candidates with g_t2i, peer mean per direction; GT pairs scored per direction' if r['architecture'] == 'directional' else ''),
                   test_read=False, label='development experiment (no TEST read in this line)', runner=f"{P['tree']}/train_unit.py",
                   job_bundle=f"{r['parent_key']}_{r['route']}_{r['architecture']}_{r['schedule']}")
        if r['status'] == 'USER_AMENDMENT_ADDED':   # (original 30 configs stay byte-identical)
            cfg.update(package_unit=None, matrix_source='R38 MATRIX_AMENDMENT.csv (user amendment before results: package B/C cells applied to every available parent)')
        if P['tree'] == 'r38f' and P['cond'] != 'n60':
            assert base['cond'] == P['cond'] and base['parent_seed'] == int(r['parent_key'][-4:])
        if P['cond'] != 'cc' and P['tree'] == 'r38c':   # CC s1002/s1003 (A block + amendment): new parent (same R37C rules)
            c = CC_CONDS[P['cond']]
            cfg['parent_seed'] = c['parent_seed']
            cfg['parent_H'] = dict(unit=c['h_unit'], series='R31 IN2R native (author defaults); not the Flickr R30-H series',
                                   model_best=str(Path(c['parent_dir']) / 'model_best.pth.tar'), seed=c['parent_seed'])
            cfg['head_init'] = cfg['head_init'].replace('parent-specific', 'parent-specific (this parent)')
            cfg['reference_unit'] = None
            cfg['reference_note'] = f'no earlier unit on this parent; config fields copied from {ref_id} (R37C, parent s1001), parent/cond replaced'
        else:
            rr = P['ref_runs'] / ref_id
            res = read_json(rr / 'RESULT.json'); hist = read_json(rr / 'TRAIN_HISTORY.json')
            assert res['status'] == 'COMPLETE' and len(hist['batch_hash']) == len(base['ckpt_steps']) and res['config']['structure'] == 'G'
            cfg['reference_unit'] = dict(unit_id=ref_id, result_json=str(rr / 'RESULT.json'), result_json_sha256=sha256(rr / 'RESULT.json'),
                                         train_history_json=str(rr / 'TRAIN_HISTORY.json'), train_history_json_sha256=sha256(rr / 'TRAIN_HISTORY.json'),
                                         use='init-state hash (shared-head keys) and batch-index hash chain asserted equal in-unit (read-only)')
            refs.setdefault(ref_id, dict(unit_id=ref_id, role='shared-constant reference (reused read-only, not re-run)' if r['route'] == 'N' else 'T shared-constant reference (reused read-only, not re-run)',
                                         parent_key=r['parent_key'], route=r['route'], objective=r['objective'], dir=str(rr),
                                         result_sha256=sha256(rr / 'RESULT.json'), dev_rsum_step8000=res['final']['dev_retrieval']['rsum'],
                                         env=res['env'], lineage_cache=res['phases'][0]['train_history'].get('lineage') if False else None,
                                         config_sha256=sha256(P['ref_cfg'] / f'{ref_id}.json')))
        write_json(W / 'configs' / f"{r['unit_id']}.json", cfg); uids.append(r['unit_id'])
        out.append({**r, 'cond': P['cond'], 'code_tree': P['tree'], 'structure': cfg['structure'], 'supervision': cfg['supervision'],
                    'reference_unit': cfg['reference_unit']['unit_id'] if cfg['reference_unit'] else '', 'config_sha256': sha256(W / 'configs' / f"{r['unit_id']}.json"),
                    'status': 'CONFIG_WRITTEN'})
    write_json(W / 'configs' / 'UNITS.json', uids)
    with open(W / 'RESOLVED_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    write_json(W / 'REUSE_REFERENCES.json', dict(created_utc=utc(), references=list(refs.values()),
               rule='shared-constant N (and T constant) arms of the B/C comparisons are the R36 (Flickr) / R37C (CC) units, read-only; every R38 unit on the same parent/route asserts the same init-state hash (shared keys) and the same batch-index hash chain at every readout; the R38 shared/constant code path is checked against R37/R37C in the CPU test (bitwise 250 steps)'))
    print(len(uids), 'configs;', len(refs), 'reference units')


if __name__ == '__main__':
    main()
