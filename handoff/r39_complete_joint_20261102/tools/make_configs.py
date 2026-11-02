"""R39: resolve the package MATRIX.csv (21 rows) into configs/<unit>.json, RESOLVED_MATRIX.csv and REUSE_REFERENCES.json.
Every config starts from the config of the R38 N-shared-cosine unit of the same parent and objective (the package's reused
N-FROZEN reference, REFERENCE_MATRIX.csv), so head init seed, sampler stream, n_queries, steps, checkpoint steps, clip, AdamW and
the cosine schedule are byte-for-byte those of the reference; R39 changes only the supervision (COMPLETE / ORACLE_MISSING), the
encoder (frozen / joint_global: + student global encoders, lr 1e-5 -> 1e-6 on the same multiplier) and bookkeeping. The R38 unit
becomes the in-unit read-only reference (init-state hash and batch-index hash chain asserted equal at every readout).
Usage: python3 -I -B tools/make_configs.py"""
import csv, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r39c'))
from common import sha256, write_json, read_json, utc, PACKAGE

R38_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
R38_RUNS = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/runs')
DS = {'Flickr60': dict(tree='r39f', cond='n60', pk='f60_R30_s1001', parent='R30_H1001',
                       teachers=['r30v11_f60_s1002_HH (R30-H f60 s1002)', 'r30v11_f60_s1003_HH (R30-H f60 s1003)']),
      'CC152K': dict(tree='r39c', cond='cc', pk='cc_R31_s1001', parent='R31_H0_default1001',
                     teachers=['r31_cc_s1002_H0_default', 'r31_cc_s1003_H0_default'])}
SUP = {'N': 'OBS_MISSING_NULL', 'COMPLETE': 'COMPLETE_XCHECK', 'ORACLE_MISSING': 'ORACLE_MISSING'}
EQ_TOL = 1e-4   # student-forward vs native-cache max |diff| on unit vectors (technical tolerance, fixed before any unit)


def main():
    rows = list(csv.DictReader(open(PACKAGE / 'MATRIX.csv'))); refs = list(csv.DictReader(open(PACKAGE / 'REFERENCE_MATRIX.csv')))
    assert len(rows) == 21 and len({r['unit'] for r in rows}) == 21 and len(refs) == 6
    units, out_rows, reuse = [], [], {}
    for r in rows:
        P = DS[r['dataset']]
        assert r['parent'] == P['parent'] and r['schedule'] == 'cosine' and r['steps'] == '8000' and r['head_seed'] == '1001'
        ref_id = f"r38_B_{P['pk']}_N_shared_{r['objective']}_cosine"
        base = read_json(R38_CFG / f'{ref_id}.json')
        assert base['structure'] == 'G' and base['supervision'] == 'OBS_MISSING_NULL' and base['objective'] == r['objective'] and base['optimizer']['schedule'] == 'cosine'
        assert base['init_seed'] == 1001 and base['steps'] == 8000 and base['cond'] == P['cond']
        rd = R38_RUNS / ref_id
        assert read_json(rd / 'DONE.json')['status'] == 'COMPLETE'
        ref = dict(unit_id=ref_id, result_json=str(rd / 'RESULT.json'), result_json_sha256=sha256(rd / 'RESULT.json'),
                   train_history_json=str(rd / 'TRAIN_HISTORY.json'), train_history_json_sha256=sha256(rd / 'TRAIN_HISTORY.json'),
                   config_sha256=sha256(R38_CFG / f'{ref_id}.json'),
                   use='R39 reused N-FROZEN reference: init-state hash (head) and batch-index hash chain asserted equal in-unit (read-only)')
        cfg = {k: v for k, v in base.items() if k not in ('unit_id', 'package_unit', 'package_block', 'job_bundle', 'block', 'reference_unit', 'architecture_rule', 'schedule_rule')}
        sup = SUP[r['supervision']]; joint = r['encoder'] == 'joint_global'
        cfg.update(unit_id=r['unit'], package_unit=r['unit'], block='R39', line='R39', code_tree=P['tree'],
                   runner=f"{P['tree']}/{'train_oracle.py' if r['oracle'] == 'True' else 'train_unit.py'}",
                   supervision=sup, supervision_label=r['supervision'], oracle=r['oracle'] == 'True', encoder=r['encoder'],
                   teacher_parents=P['teachers'] if r['supervision'] == 'COMPLETE' else None,
                   architecture='shared', schedule_rule=base.get('schedule_rule'), architecture_rule='shared G (R35/R36 GHead): g = a s_H + b + w^T h(phi(u, v, s_H)) + c',
                   reference_unit=ref, label='development experiment (no TEST read in this line)',
                   job_bundle=None)
        if sup == 'COMPLETE_XCHECK':
            cfg['supervision_rule'] = ('R38 N lists unchanged (target natural top-32 + null, 33 slots, all permuted, Q = Uniform(33)); rows whose N label is all-null are completed '
                                       'iff the two fixed same-series teachers agree on their full-FIT-gallery top-1, the agreed ID is in the target top-32 (no insertion) and both '
                                       'reverse ranks <= 5: A = 0.5 one_hot(agreed) + 0.5 null (fixed, not a probability estimate); every other row keeps its N label; no truth read')
        elif sup == 'ORACLE_MISSING':
            cfg['supervision_rule'] = ('Flickr oracle feasibility diagnostic (separate process / dir): rows whose N label is all-null: if the list holds >= 1 true slot '
                                       '(pre-injection pairing), A = 0.5 uniform(true slots) + 0.5 null, else unchanged; observed-present rows unchanged; not deployable')
        if joint:
            cfg['optimizer'] = dict(cfg['optimizer'], lr_encoder=1e-5)
            cfg['joint_rule'] = dict(student='per peer: copy of the parent global encoders (native EncoderImage fc+MLP+GPO, EncoderText BiGRU+fc+MLP+GPO; word embedding frozen), '
                                             're-encoded from region features / tokens, native per-batch centring held at the parent batch vector, L2',
                                     trainable='image fc, MLP (BN affine; running stats frozen), GPO GRU+linear; text BiGRU, fc, MLP, GPO GRU+linear',
                                     frozen='word embedding, BN running statistics, region features, frozen H (candidates, s_H skip, top-100), teachers',
                                     lr='head 1e-4 -> 1e-5, student 1e-5 -> 1e-6, common cosine multiplier, no warmup, optimizer never reset; AdamW wd 1e-4 both groups',
                                     clip='one clip per peer (head + null + student) at 2.0; pre-clip head/student norms and clip coefficient recorded',
                                     inference='frozen H top-100, reranked by the peer-mean critic on student vectors (rest in H order)',
                                     equivalence_tol_max_abs=EQ_TOL)
        cfg['evaluation'] = base['evaluation'] + ('; JOINT: phi on the current student vectors, s_H frozen-H native score' if joint else '')
        units.append(r['unit']); reuse[r['unit']] = ref
        write_json(W / 'configs' / f"{r['unit']}.json", cfg)
        out_rows.append(dict(unit=r['unit'], dataset=r['dataset'], tree=P['tree'], cond=P['cond'], supervision=sup, encoder=r['encoder'], objective=r['objective'],
                             oracle=r['oracle'], reference_unit=ref_id, config_sha256=sha256(W / 'configs' / f"{r['unit']}.json")))
    write_json(W / 'configs' / 'UNITS.json', units)
    with open(W / 'RESOLVED_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    rr = []
    for r in refs:
        P = DS[r['dataset']]; ref_id = f"r38_B_{P['pk']}_N_shared_{r['objective']}_cosine"; rd = R38_RUNS / ref_id
        res = read_json(rd / 'RESULT.json')
        rr.append(dict(package_row=r, r38_unit=ref_id, result_json_sha256=sha256(rd / 'RESULT.json'), train_history_sha256=sha256(rd / 'TRAIN_HISTORY.json'),
                       config_sha256=sha256(R38_CFG / f'{ref_id}.json'), rsum_step8000=res['final']['dev_retrieval']['rsum'], status=res['status'],
                       init_state_sha256=res['phases'][0]['identity']['init_state_sha256'], batch_index_sha256=res['phases'][0]['identity']['batch_index_sha256'],
                       action='REUSED (not resubmitted); identity (head init, batch chain, steps, sampler/init seeds, ckpt steps) asserted in every R39 unit of the same parent/objective'))
    write_json(W / 'REUSE_REFERENCES.json', dict(utc=utc(), references=rr))
    print('configs', len(units))


if __name__ == '__main__':
    main()
