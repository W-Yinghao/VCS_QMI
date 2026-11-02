"""CPU: resolve the 6 CC152K rows of the R37 package MATRIX_PROPOSED.csv (block C_real_noise; R37 resolved them as
NOT_RUN_MISSING_PARENT) into R37C per-unit JSON configs bound to the R31 CC152K parent + RESOLVED_MATRIX.csv.
Head init seed, sampler stream, steps, optimizer, clip, n_queries are those of the R36/R37 G units (only the dataset, the
parent state, its calibration, candidate tables and trust subset change). Usage: python -B make_configs.py <MATRIX_PROPOSED.csv>"""
from __future__ import annotations
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CONDS, CKPT_STEPS_ABC, write_json, read_json

OPT = dict(name='AdamW', lr_head=1e-4, weight_decay=1e-4, betas=[0.9, 0.999], eps=1e-8, schedule='constant')
ROUTE = {'T_G': 'TRUST_HARD', 'N_G': 'OBS_MISSING_NULL'}
SUP_TEXT = {
    'TRUST_HARD': "R36 T ported to CC152K: anchor records from the parent's frozen B-TRUST subset (native eval_train GMM, mean(probA,probB)>=0.5); R35 C forced lists (observed positive + 31 other frozen-H top-32); A from the full observed (identity) relation; Q = Uniform(32)",
    'OBS_MISSING_NULL': 'R36 N ported to CC152K: unmodified frozen-H top-32 real candidates + null slot (33); A = observed positive in list (identity relation: the record\'s own image/caption) or A_null=1 (= observed partner not in the native top-32); all 33 slots permuted; Q = Uniform(33); null = learnable scalar per peer/direction (init 0); inference ignores null',
}
JOB = {'TRUST_HARD': 'cc_T', 'OBS_MISSING_NULL': 'cc_N'}


def main(matrix_csv):
    rows = [r for r in csv.DictReader(open(matrix_csv)) if r['dataset'] == 'CC152K']
    assert len(rows) == 6 and {r['route'] for r in rows} == {'T_G', 'N_G'} and {r['objective'] for r in rows} == {'J', 'BCE', 'NCE'}
    c = CONDS['cc']
    out_rows, uids = [], []
    for r in rows:
        assert r['steps'] == '8000' and r['parent_seed'] == '1001'
        sup = ROUTE[r['route']]
        uid = f"r37c_{r['unit']}"
        cfg = dict(unit_id=uid, package_unit=r['unit'], package_block=r['block'], block='C', cond='cc', dataset='CC152K', noise='real (uninjected)',
                   parent_seed=1001, structure='G', objective=r['objective'], supervision=sup, supervision_rule=SUP_TEXT[sup], oracle=False,
                   steps=8000, ckpt_steps=list(CKPT_STEPS_ABC), primary_endpoint='fixed_step_8000',
                   init_seed=1001, sampler_seed=[1001, 3], n_queries=16, n_slots=33 if sup != 'TRUST_HARD' else 32,
                   optimizer=OPT, clip_per_peer=2.0,
                   head_init='fresh R35 build_heads(G, a0, b0, 1001): parent-specific FIT-only calibration a0*s+b0 + zero-init residual; null slots init 0',
                   seed_rule='head init seed 1001 and sampler stream [1001,3] identical to R36/R37 G units',
                   parent_H=dict(unit=c['h_unit'], series='R31 IN2R native (author defaults); not the Flickr R30-H series',
                                 model_best=str(Path(c['parent_dir']) / 'model_best.pth.tar'), seed=1001),
                   evaluation='DEV (1000 images x 1000 captions, per_captions 1) two-stage: frozen-H top-100 of the parent reranked by peer-mean g (rest in H order), all queries; null slot never used at inference',
                   test_read=False, label='development experiment (CC152K TEST not read in this line)',
                   precision='float32 training (TF32 off, deterministic algorithms); evaluation affine in float64 on the exact native shard score',
                   job_bundle=JOB[sup], runner='train_unit.py')
        write_json(ROOT / 'configs' / f'{uid}.json', cfg); uids.append(uid)
        out_rows.append({**r, 'unit_id': uid, 'cond': 'cc', 'supervision': sup, 'job_bundle': cfg['job_bundle'], 'resolved_status': 'CONFIG_WRITTEN (R37C, parent r31_cc_s1001_H0_default)'})
    write_json(ROOT / 'configs' / 'UNITS.json', uids)
    with open(ROOT / 'RESOLVED_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(len(uids), 'configs')


if __name__ == '__main__':
    main(sys.argv[1])
