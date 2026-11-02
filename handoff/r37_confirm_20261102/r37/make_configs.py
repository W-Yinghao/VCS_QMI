"""CPU: resolve the package MATRIX_PROPOSED.csv (24 rows) into R37 per-unit JSON configs (18 Flickr units) +
RESOLVED_MATRIX.csv. The 6 CC152K rows are resolved as NOT_RUN_MISSING_PARENT (see PARENT_BINDING.json): no config.
Reads only the package matrix (passed by path)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CONDS, CKPT_STEPS_ABC, R30_RUNS, write_json

OPT = dict(name='AdamW', lr_head=1e-4, weight_decay=1e-4, betas=[0.9, 0.999], eps=1e-8, schedule='constant')
ROUTE = {'T_G': 'TRUST_HARD', 'N_G': 'OBS_MISSING_NULL', 'TRUST_NATIVE_NULL_G': 'TRUST_NATIVE_NULL', 'N_PRESENT_ONLY_G': 'N_PRESENT_ONLY'}
COND = {'1001': 'n60', '1002': 'n60s1002', '1003': 'n60s1003'}
SUP_TEXT = {
    'TRUST_HARD': "R36 T: anchor records from the parent's frozen B-TRUST subset (native eval_train GMM, mean(probA,probB)>=0.5); R35 C forced lists (observed positive + 31 other frozen-H top-32); A from the full observed map; Q = Uniform(32)",
    'OBS_MISSING_NULL': 'R36 N: unmodified frozen-H top-32 real candidates + null slot (33); A = observed positives in list or A_null=1; all 33 slots permuted; Q = Uniform(33); null = learnable scalar per peer/direction (init 0); inference ignores null',
    'TRUST_NATIVE_NULL': "R37 B1 TRUST+N: anchor records drawn from the parent's frozen B-TRUST subset (as T) with the full N protocol (native top-32, no insertion, 33 slots incl. observed-missing null, Q = Uniform(33)); positive labels from the full-pool observed relations (N definition unchanged)",
    'N_PRESENT_ONLY': 'R37 B2 N-PRESENT_ONLY: N lists of all-pool queries; rows whose observed partner is absent from native top-32 (A_null=1) get zero loss and are not redrawn; per direction sum(mask*row)/(n_queries*p_d), p_d = fixed training-pool presence probability; same 33-slot output and Q = Uniform(33) as N; DEV inference identical to N (real slots only, no query rejected)',
}
JOB = {('n60s1002', 'TRUST_HARD'): 'h1002_T', ('n60s1002', 'OBS_MISSING_NULL'): 'h1002_N', ('n60s1003', 'TRUST_HARD'): 'h1003_T',
       ('n60s1003', 'OBS_MISSING_NULL'): 'h1003_N', ('n60', 'TRUST_NATIVE_NULL'): 'h1001_TN', ('n60', 'N_PRESENT_ONLY'): 'h1001_PO'}


def main(matrix_csv):
    rows = list(csv.DictReader(open(matrix_csv)))
    assert len(rows) == 24 and len({r['unit'] for r in rows}) == 24
    out_rows, uids = [], []
    for r in rows:
        assert r['steps'] == '8000'
        if r['dataset'] == 'CC152K':
            out_rows.append({**r, 'unit_id': f"r37_{r['unit']}", 'cond': '', 'supervision': '', 'job_bundle': '',
                             'resolved_status': 'NOT_RUN_MISSING_PARENT (no same-series R30-H CC152K parent; see PARENT_BINDING.json)'})
            continue
        assert r['dataset'] == 'Flickr60'
        cond, sup = COND[r['parent_seed']], ROUTE[r['route']]
        uid = f"r37_{r['unit']}"
        cfg = dict(unit_id=uid, package_unit=r['unit'], package_block=r['block'], block=r['block'][0], cond=cond, noise_percent=60,
                   parent_seed=int(r['parent_seed']), structure='G', objective=r['objective'], supervision=sup, supervision_rule=SUP_TEXT[sup], oracle=False,
                   steps=8000, ckpt_steps=list(CKPT_STEPS_ABC), primary_endpoint='fixed_step_8000',
                   init_seed=1001, sampler_seed=[1001, 3], n_queries=16, n_slots=33 if sup != 'TRUST_HARD' else 32,
                   optimizer=OPT, clip_per_peer=2.0,
                   head_init='fresh R35 build_heads(G, a0, b0, 1001): parent-specific FIT-only calibration a0*s+b0 + zero-init residual; null slots init 0',
                   seed_rule='head init seed 1001 and sampler stream [1001,3] identical to R36 for every parent (only the parent state, its calibration, candidate tables and trust subset change)',
                   parent_H=dict(unit=CONDS[cond]['h_unit'], model_best=str(R30_RUNS / CONDS[cond]['h_unit'] / 'model_best.pth.tar'), seed=int(r['parent_seed'])),
                   evaluation='DEV two-stage: frozen-H top-100 of the same parent reranked by peer-mean g (rest in H order), all queries; null slot never used at inference',
                   test_read=False, precision='float32 training (TF32 off, deterministic algorithms); evaluation affine in float64 on the exact native shard score',
                   job_bundle=JOB[(cond, sup)], runner='train_unit.py')
        write_json(ROOT / 'configs' / f'{uid}.json', cfg); uids.append(uid)
        out_rows.append({**r, 'unit_id': uid, 'cond': cond, 'supervision': sup, 'job_bundle': cfg['job_bundle'], 'resolved_status': 'CONFIG_WRITTEN'})
    write_json(ROOT / 'configs' / 'UNITS.json', uids)
    with open(ROOT / 'RESOLVED_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(len(uids), 'configs;', len(out_rows) - len(uids), 'rows not run')


if __name__ == '__main__':
    main(sys.argv[1])
