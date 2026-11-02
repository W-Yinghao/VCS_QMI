"""CPU: resolve the package configs/MATRIX.csv (27 units) into per-unit JSON configs + RESOLVED_MATRIX.csv.
Reads only the package matrix (passed by path); writes ROOT/configs and ROOT/RESOLVED_MATRIX.csv."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, CONDS, CKPT_STEPS_ABC, CKPT_STEPS_D, R30_RUNS, write_json

OPT = dict(name='AdamW', lr_head=1e-4, weight_decay=1e-4, betas=[0.9, 0.999], eps=1e-8, schedule='constant')
SUP_TEXT = {
    'RANDOM': 'observed partner forced + 31 identities uniform w/o replacement from the opposite FIT identity pool (excluding the forced partner) from a separate stream; A = observed positives; Q = Uniform(32)',
    'SHIELD_ORACLE': 'R35 C lists/queries/A; ORACLE: true slots with A=0 removed from Q (J/BCE: Q uniform on the rest) and from the NCE denominator; wrong observed positives keep their mass',
    'RELABEL_AVAILABLE_ORACLE': 'R35 C lists/queries; ORACLE: A = uniform over the true slots present in the list, else A unchanged; Q = Uniform(32)',
    'TRUST_HARD': 'anchor records from the frozen R35 B-TRUST subset (mean(probA,probB)>=0.5, 61588 records); R35 C forced lists; A from the full observed map; Q = Uniform(32)',
    'OBS_MISSING_NULL': 'unmodified frozen-H top-32 real candidates + null slot (33); A = observed positives in list or A_null=1; all 33 slots permuted; Q = Uniform(33); null = learnable scalar per peer/direction (init 0); inference ignores null',
    'OBS_HARD_CLEAN_REPLICATION': 'R35 C-L (0% noise) with a new head-init/sampler seed; then 4000 steps R35 D HEAD_CONTROL continuation from its own step 8000 (optimizer reset, stream [seed,4])',
}
JOBS = {
    'M_RANDOM': lambda r: r['block'] == 'M' and r['supervision'] == 'RANDOM',
    'M_ORACLE': lambda r: r['block'] == 'M' and r['oracle'] == 'True',
    'T_G': lambda r: r['block'] == 'T' and r['structure'] == 'G',
    'T_L': lambda r: r['block'] == 'T' and r['structure'] == 'L',
    'N_G': lambda r: r['block'] == 'N' and r['structure'] == 'G',
    'N_L': lambda r: r['block'] == 'N' and r['structure'] == 'L',
    'V_h2002': lambda r: r['block'] == 'V' and r['head_seed'] == '2002',
    'V_h2003': lambda r: r['block'] == 'V' and r['head_seed'] == '2003',
}


def main(matrix_csv):
    rows = list(csv.DictReader(open(matrix_csv)))
    assert len(rows) == 27 and len({r['unit_id'] for r in rows}) == 27
    out_rows, uids = [], []
    for r in rows:
        cond = 'n00' if r['noise'] == '0' else 'n60'
        hs, ss = int(r['head_seed']), int(r['sampler_seed'])
        assert int(r['parent_encoder_seed']) == 1001 and r['unfreeze_encoder'] == 'False'
        cfg = dict(unit_id=r['unit_id'], block=r['block'], cond=cond, noise_percent=int(r['noise']), structure=r['structure'], objective=r['objective'],
                   supervision=r['supervision'], supervision_rule=SUP_TEXT[r['supervision']], oracle=r['oracle'] == 'True',
                   steps=int(r['steps']), ckpt_steps=list(CKPT_STEPS_ABC), primary_endpoint=r['primary_endpoint'],
                   init_seed=hs, sampler_seed=[ss, 3], n_queries=16, n_slots=33 if r['supervision'] == 'OBS_MISSING_NULL' else 32,
                   optimizer=OPT, clip_per_peer=2.0, local_encoders='frozen (H local fc/mlp/embedding/BiGRU; no gradient)',
                   head_init='fresh R35 build_heads(structure, a0, b0, init_seed): common FIT-only calibration a0*s+b0 + zero-init residual; no C/D/A/B weights loaded',
                   parent_H=dict(unit=CONDS[cond]['h_unit'], model_best=str(R30_RUNS / CONDS[cond]['h_unit'] / 'model_best.pth.tar'), seed=1001),
                   evaluation='DEV two-stage: frozen-H top-100 reranked by peer-mean g (rest in H order), all queries; null slot never used at inference',
                   test_read=False, precision='float32 training (TF32 off, deterministic algorithms); evaluation affine in float64 on the exact native shard score')
        if r['supervision'] == 'RANDOM':
            cfg['sampler_seed2'] = [ss, 3, 7]
        if r['block'] == 'V':
            cfg.update(phase2_steps=int(r['phase2_steps']), phase2_ckpt_steps=list(CKPT_STEPS_D), phase2_sampler_seed=[ss, 4],
                       phase2_rule='R35 D HEAD_CONTROL: load own phase-1 step-8000 head, fresh AdamW (reset), new sampler stream [seed,4], head only',
                       secondary_endpoint='phase1_fixed_step_8000')
        job = [k for k, f in JOBS.items() if f(r)]; assert len(job) == 1
        cfg['job_bundle'] = job[0]; cfg['runner'] = 'train_oracle.py' if cfg['oracle'] else 'train_unit.py'
        write_json(ROOT / 'configs' / f"{r['unit_id']}.json", cfg)
        uids.append(r['unit_id'])
        out_rows.append({**r, 'cond': cond, 'job_bundle': job[0], 'runner': cfg['runner'], 'sampler_seed_resolved': json.dumps(cfg['sampler_seed']),
                         'sampler_seed2': json.dumps(cfg.get('sampler_seed2')), 'phase2_sampler_seed': json.dumps(cfg.get('phase2_sampler_seed')),
                         'ckpt_steps': ' '.join(map(str, cfg['ckpt_steps'])), 'phase2_ckpt_steps': ' '.join(map(str, cfg.get('phase2_ckpt_steps', []))),
                         'n_slots': cfg['n_slots'], 'lr_head': OPT['lr_head'], 'weight_decay': OPT['weight_decay'], 'clip_per_peer': 2.0,
                         'output_dir': ('oracle_runs/' if cfg['oracle'] else 'runs/') + r['unit_id']})
    write_json(ROOT / 'configs' / 'UNITS.json', uids)
    with open(ROOT / 'RESOLVED_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(len(uids), 'configs')


if __name__ == '__main__':
    main(sys.argv[1])
